//! termem — a workspace-organized mini terminal emulator.
//!
//! Backend : `alacritty_terminal` (PTY + VT/ANSI state machine + parser thread)
//! Frontend: `winit` (window + keyboard/mouse) + `softbuffer` (CPU framebuffer)
//!           + `fontdue` (glyph rasterization)
//!
//! Architecture is functional-core / imperative-shell:
//!
//!   * The workspace is a *rose tree* parsed from the `workspace` file. The
//!     tree and every decision over it (which rows to draw, which leaves a
//!     group contains, what to start/stop, whether something is already
//!     running) are **pure functions** — see the `core` section.
//!   * Only PTY spawning, byte I/O, `/proc` observation and drawing are
//!     effectful; those live in the `shell` section (`State` / `App`).
//!
//! Run with: `cargo run --release`

use std::collections::HashMap;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use alacritty_terminal::event::{Event as TermEvent, EventListener, WindowSize};
use alacritty_terminal::event_loop::{EventLoop as PtyEventLoop, EventLoopSender, Msg};
use alacritty_terminal::grid::Dimensions;
use alacritty_terminal::index::{Column, Line, Point, Side};
use alacritty_terminal::selection::{Selection, SelectionType};
use alacritty_terminal::sync::FairMutex;
use alacritty_terminal::term::{Config, Term};
use alacritty_terminal::tty::{self, Options as PtyOptions};
use alacritty_terminal::vte::ansi::{Color, NamedColor};

use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::{Key, ModifiersState, NamedKey};
use winit::window::{ResizeDirection, Window, WindowId};

const FONT_PX: f32 = 16.0;

// ===========================================================================
// core — the workspace as a pure rose tree
// ===========================================================================

/// Stable identifier for a position in the tree (arena index). Stable for the
/// lifetime of the process, including across spawn/exit of a leaf's session.
type NodeId = usize;

/// What a node *is*. `Leaf` carries the spec needed to run it.
#[derive(Clone)]
enum Kind {
    /// The invisible forest root (`workspaces`).
    Root,
    /// A folder. May contain groups and leaves. Has no command of its own.
    Group,
    /// A runnable session: a working directory and a default command. An
    /// empty `command` means "just a shell here" (Scratch/Transient/new tab).
    Leaf { workdir: PathBuf, command: String },
}

/// One node of the workspace tree. `expanded`/`dynamic` are the only mutable
/// bits and they are explicit, never hidden behind a traversal.
struct Node {
    parent: Option<NodeId>,
    children: Vec<NodeId>,
    name: String,
    kind: Kind,
    /// Groups only: whether children are shown. Ignored for leaves.
    expanded: bool,
    /// Created at runtime (a new scratch tab) rather than from the spec file.
    /// Dynamic leaves are removed from the tree when their session exits.
    dynamic: bool,
}

/// The workspace. An arena of `Node`s plus the root index. The arena layout is
/// the idiomatic Rust spelling of an immutable-shaped tree: structure is set
/// up once, traversals below are pure reads, mutations are localized.
struct Tree {
    nodes: Vec<Node>,
    root: NodeId,
}

impl Tree {
    fn push(&mut self, parent: Option<NodeId>, name: String, kind: Kind, dynamic: bool) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(Node {
            parent,
            children: Vec::new(),
            name,
            kind,
            expanded: true,
            dynamic,
        });
        if let Some(p) = parent {
            self.nodes[p].children.push(id);
        }
        id
    }

    fn is_group(&self, id: NodeId) -> bool {
        matches!(self.nodes[id].kind, Kind::Group | Kind::Root)
    }
    fn is_leaf(&self, id: NodeId) -> bool {
        matches!(self.nodes[id].kind, Kind::Leaf { .. })
    }

    /// Leaf spec, if `id` is a leaf.
    fn leaf_spec(&self, id: NodeId) -> Option<(&Path, &str)> {
        match &self.nodes[id].kind {
            Kind::Leaf { workdir, command } => Some((workdir.as_path(), command.as_str())),
            _ => None,
        }
    }

    /// Mutable handle to a leaf's default command (the inspector edits this).
    fn command_mut(&mut self, id: NodeId) -> Option<&mut String> {
        match &mut self.nodes[id].kind {
            Kind::Leaf { command, .. } => Some(command),
            _ => None,
        }
    }

    /// Current text of an inspector field for `id`. `Command` is `None` on a
    /// group (groups have no command — that field is shown disabled).
    fn field_text(&self, id: NodeId, f: Field) -> Option<String> {
        match f {
            Field::Title => Some(self.nodes[id].name.clone()),
            Field::Command => self.leaf_spec(id).map(|(_, c)| c.to_string()),
        }
    }

    /// DFS over visible nodes (groups gate their subtree via `expanded`). This
    /// is the catamorphism the sidebar render and hit-testing both fold over,
    /// so the picture on screen and the click map can never disagree.
    fn rows(&self) -> Vec<Row> {
        fn go(t: &Tree, id: NodeId, depth: usize, out: &mut Vec<Row>) {
            for &c in &t.nodes[id].children {
                let n = &t.nodes[c];
                let is_group = matches!(n.kind, Kind::Group);
                out.push(Row {
                    id: c,
                    depth,
                    name: n.name.clone(),
                    is_group,
                    expanded: n.expanded,
                    has_children: !n.children.is_empty(),
                });
                if is_group && n.expanded {
                    go(t, c, depth + 1, out);
                }
            }
        }
        let mut out = Vec::new();
        go(self, self.root, 0, &mut out);
        out
    }

    /// Every leaf in `id`'s subtree (a leaf yields itself). The fold that turns
    /// "Start on a group" into "start each of these leaves".
    fn leaves(&self, id: NodeId) -> Vec<NodeId> {
        let mut out = Vec::new();
        fn go(t: &Tree, id: NodeId, out: &mut Vec<NodeId>) {
            if t.is_leaf(id) {
                out.push(id);
                return;
            }
            for &c in &t.nodes[id].children {
                go(t, c, out);
            }
        }
        go(self, id, &mut out);
        out
    }

    /// The group a new session should be attached to given the current
    /// selection: a selected group is its own context; a selected leaf hands
    /// off to its enclosing group.
    fn group_for_new(&self, sel: NodeId) -> NodeId {
        if self.is_group(sel) {
            sel
        } else {
            self.nodes[sel].parent.unwrap_or(self.root)
        }
    }

    /// First leaf in the subtree, in DFS order — used to pick what terminal to
    /// show when a group (rather than a leaf) is selected.
    fn first_leaf(&self, id: NodeId) -> Option<NodeId> {
        self.leaves(id).into_iter().next()
    }

    /// `Group / Sub / Leaf` path string for the header bar.
    fn path(&self, id: NodeId) -> String {
        let mut parts = Vec::new();
        let mut cur = Some(id);
        while let Some(c) = cur {
            if c == self.root {
                break;
            }
            parts.push(self.nodes[c].name.clone());
            cur = self.nodes[c].parent;
        }
        parts.reverse();
        parts.join("  /  ")
    }
}

/// A flattened, render-ready view of one visible tree node.
struct Row {
    id: NodeId,
    depth: usize,
    name: String,
    is_group: bool,
    expanded: bool,
    has_children: bool,
}

/// Expand `~` / `~/...` to `$HOME`. Pure given `home`.
fn expand_tilde(s: &str, home: &Path) -> PathBuf {
    if s == "~" {
        home.to_path_buf()
    } else if let Some(rest) = s.strip_prefix("~/") {
        home.join(rest)
    } else {
        PathBuf::from(s)
    }
}

/// Parse the `workspace` file into a tree.
///
/// Format: indentation depth = tree depth (one `\t` per level). Fields on a
/// line are tab-separated; surrounding whitespace and `"`quotes`"` are
/// stripped. A line with a working directory + command is a `Leaf`; a bare
/// name is a `Group`. The first non-empty line (`workspaces`) is the root
/// sentinel. Two empty groups, `Scratch` and `Transient`, are appended as
/// homes for ad-hoc sessions.
fn parse_workspace(text: &str, home: &Path) -> Tree {
    let mut tree = Tree {
        nodes: Vec::new(),
        root: 0,
    };
    let root = tree.push(None, "workspaces".into(), Kind::Root, false);
    tree.root = root;

    // stack[d] = id of the most recent node opened at depth d. A child at
    // depth d attaches to stack[d-1]; depth 1 attaches to the root.
    let mut stack: Vec<NodeId> = vec![root];

    for raw in text.lines() {
        if raw.trim().is_empty() {
            continue;
        }
        let depth = raw.chars().take_while(|&c| c == '\t').count();
        let rest = raw.trim_start_matches('\t');
        let fields: Vec<String> = rest
            .split('\t')
            .map(|f| f.trim().trim_matches('"').trim().to_string())
            .filter(|f| !f.is_empty())
            .collect();
        if fields.is_empty() {
            continue;
        }
        if depth == 0 {
            // The `workspaces` root line; already created.
            continue;
        }
        let parent = *stack.get(depth - 1).unwrap_or(&root);
        let id = if fields.len() >= 3 {
            // name, workdir, command  -> a runnable leaf
            let workdir = expand_tilde(&fields[1], home);
            tree.push(
                Some(parent),
                fields[0].clone(),
                Kind::Leaf {
                    workdir,
                    command: fields[2].clone(),
                },
                false,
            )
        } else {
            // bare name -> a group
            tree.push(Some(parent), fields[0].clone(), Kind::Group, false)
        };
        stack.truncate(depth);
        stack.push(id);
    }

    for name in ["Scratch", "Transient"] {
        tree.push(Some(root), name.into(), Kind::Group, false);
    }
    tree
}

/// What the shell observed about a session's PTY, gathered from `/proc`.
/// Effectful to *produce* (`observe`), but a plain value the planner reasons
/// about purely.
#[derive(Clone, Default)]
struct Obs {
    /// The shell has a foreground child — i.e. a command is running, not just
    /// an idle prompt.
    foreground: bool,
    cmd: Option<String>,
}

/// Pure predicate: should `plan_start` *skip* running `command` here because
/// it's already running? True when the shell has a foreground job whose
/// command line mentions the configured program. Conservative: if we can't
/// tell, we don't skip (running again is recoverable with Ctrl+C; silently
/// not starting is not).
fn already_running(obs: &Obs, command: &str) -> bool {
    if !obs.foreground || command.trim().is_empty() {
        return false;
    }
    let prog = command
        .trim()
        .split_whitespace()
        .next()
        .unwrap_or("")
        .trim_start_matches("./");
    let prog = Path::new(prog)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(prog);
    !prog.is_empty() && obs.cmd.as_deref().is_some_and(|c| c.contains(prog))
}

/// An editable text field in the right-hand inspector.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Field {
    Title,
    Command,
}

/// A keystroke against a single-line text field, normalized away from winit.
enum Edit {
    Ins(char),
    Back,
    Del,
    Left,
    Right,
    Home,
    End,
}

/// Pure: apply one `Edit` to `(text, caret)` and return the new pair. `caret`
/// is a character index in `0..=len`. Single-line, UTF-8 safe (operates on
/// `char`s, not bytes). The inspector's whole editing model is this function;
/// everything else is just plumbing keys into it and the result back out.
fn apply_edit(text: &str, caret: usize, e: Edit) -> (String, usize) {
    let mut v: Vec<char> = text.chars().collect();
    let mut c = caret.min(v.len());
    match e {
        Edit::Ins(ch) => {
            v.insert(c, ch);
            c += 1;
        }
        Edit::Back if c > 0 => {
            v.remove(c - 1);
            c -= 1;
        }
        Edit::Del if c < v.len() => {
            v.remove(c);
        }
        Edit::Left => c = c.saturating_sub(1),
        Edit::Right if c < v.len() => c += 1,
        Edit::Home => c = 0,
        Edit::End => c = v.len(),
        _ => {}
    }
    (v.into_iter().collect(), c)
}

/// One lifecycle effect the pure planner emits for the shell to execute.
#[derive(Clone)]
enum Action {
    /// Spawn an idle PTY for this leaf (sets cwd, runs no command).
    Spawn(NodeId),
    /// Type the leaf's default command + Enter into its session.
    Run(NodeId),
    /// Send Ctrl+C to the leaf's session.
    Sigint(NodeId),
}

/// Pure: starting `target` means, for every leaf under it, ensuring a session
/// exists and then running its command unless it's already running. Spawn
/// precedes Run for the same leaf so a re-started (previously exited) spec
/// leaf comes back. Groups fan out to all descendant leaves — each runs in its
/// own PTY thread, so this is parallel for free.
fn plan_start(
    tree: &Tree,
    has_session: &dyn Fn(NodeId) -> bool,
    obs: &HashMap<NodeId, Obs>,
    target: NodeId,
) -> Vec<Action> {
    let mut acts = Vec::new();
    for leaf in tree.leaves(target) {
        let Some((_, command)) = tree.leaf_spec(leaf) else {
            continue;
        };
        if !has_session(leaf) {
            acts.push(Action::Spawn(leaf));
        } else if obs.get(&leaf).is_some_and(|o| already_running(o, command)) {
            continue;
        }
        if !command.trim().is_empty() {
            acts.push(Action::Run(leaf));
        }
    }
    acts
}

/// Pure: stopping `target` sends Ctrl+C to every leaf under it that has a live
/// session.
fn plan_stop(tree: &Tree, has_session: &dyn Fn(NodeId) -> bool, target: NodeId) -> Vec<Action> {
    tree.leaves(target)
        .into_iter()
        .filter(|&l| has_session(l))
        .map(Action::Sigint)
        .collect()
}

/// Walk `/proc` to see what a PTY's shell is doing. Linux-only and best-effort
/// — any failure degrades to "nothing observed" (`Obs::default`), which the
/// planner treats as "not running".
fn observe(shell_pid: u32) -> Obs {
    fn children(pid: u32) -> Vec<u32> {
        std::fs::read_to_string(format!("/proc/{pid}/task/{pid}/children"))
            .ok()
            .map(|s| s.split_whitespace().filter_map(|x| x.parse().ok()).collect())
            .unwrap_or_default()
    }
    // Descend to the deepest foreground descendant of the shell.
    let mut cur = shell_pid;
    loop {
        match children(cur).first() {
            Some(&c) => cur = c,
            None => break,
        }
    }
    if cur == shell_pid {
        return Obs::default();
    }
    let cmd = std::fs::read(format!("/proc/{cur}/cmdline")).ok().map(|b| {
        b.split(|&c| c == 0)
            .filter(|s| !s.is_empty())
            .map(|s| String::from_utf8_lossy(s).into_owned())
            .collect::<Vec<_>>()
            .join(" ")
    });
    Obs {
        foreground: true,
        cmd: cmd.filter(|s| !s.is_empty()),
    }
}

// ===========================================================================
// terminal session plumbing (unchanged core, generalized to take a cwd)
// ===========================================================================

/// Userland event from a PTY parser thread. Each event carries the id of the
/// session it originated from so the GUI thread can route it.
#[derive(Debug, Clone)]
enum UserEvent {
    Wakeup,
    Exit(u64),
    Title(u64, String),
    ResetTitle(u64),
}

/// `EventListener` impl handed to `alacritty_terminal`. Forwards the events we
/// care about onto the winit loop. `Clone + Send` (the PTY runs on its own
/// thread); `id` ties events back to the owning session.
#[derive(Clone)]
struct Listener {
    proxy: EventLoopProxy<UserEvent>,
    id: u64,
}

impl EventListener for Listener {
    fn send_event(&self, event: TermEvent) {
        let _ = match event {
            TermEvent::Wakeup => self.proxy.send_event(UserEvent::Wakeup),
            TermEvent::Exit | TermEvent::ChildExit(_) => {
                self.proxy.send_event(UserEvent::Exit(self.id))
            }
            TermEvent::Title(t) => self.proxy.send_event(UserEvent::Title(self.id, t)),
            TermEvent::ResetTitle => self.proxy.send_event(UserEvent::ResetTitle(self.id)),
            _ => Ok(()),
        };
    }
}

/// Grid geometry. `alacritty_terminal` needs this to size the terminal and the
/// PTY (`Dimensions` for `Term`, `WindowSize` for the kernel PTY ioctl).
#[derive(Clone, Copy)]
struct TermSize {
    cols: usize,
    lines: usize,
}

impl Dimensions for TermSize {
    fn total_lines(&self) -> usize {
        self.lines
    }
    fn screen_lines(&self) -> usize {
        self.lines
    }
    fn columns(&self) -> usize {
        self.cols
    }
}

/// A rasterized glyph: coverage bitmap plus placement metrics.
struct Glyph {
    w: usize,
    h: usize,
    left: i32,
    top: i32, // pixels from cell top to glyph top
    bitmap: Vec<u8>,
}

struct Renderer {
    /// `fonts[0]` is the primary monospace face (defines cell metrics); the
    /// rest are fallbacks consulted, in order, for glyphs it lacks.
    fonts: Vec<fontdue::Font>,
    cell_w: usize,
    cell_h: usize,
    ascent: f32,
    cache: HashMap<char, Glyph>,
}

impl Renderer {
    fn new() -> Self {
        let fonts: Vec<fontdue::Font> = load_fonts()
            .into_iter()
            .filter_map(|b| fontdue::Font::from_bytes(b, fontdue::FontSettings::default()).ok())
            .collect();
        assert!(!fonts.is_empty(), "no usable font found");
        let primary = &fonts[0];

        let lm = primary
            .horizontal_line_metrics(FONT_PX)
            .expect("font line metrics");
        let cell_h = lm.new_line_size.ceil() as usize;
        // Monospace: every cell is the advance width of a representative glyph.
        let cell_w = primary.metrics('M', FONT_PX).advance_width.ceil() as usize;

        Self {
            fonts,
            cell_w: cell_w.max(1),
            cell_h: cell_h.max(1),
            ascent: lm.ascent,
            cache: HashMap::new(),
        }
    }

    fn glyph(&mut self, c: char) -> &Glyph {
        let ascent = self.ascent;
        let fonts = &self.fonts;
        self.cache.entry(c).or_insert_with(|| {
            // Pick the first font that actually has this glyph; fall back to
            // the primary (renders .notdef) if none do.
            let font = fonts
                .iter()
                .find(|f| f.lookup_glyph_index(c) != 0)
                .unwrap_or(&fonts[0]);
            let (m, bitmap) = font.rasterize(c, FONT_PX);
            // fontdue bitmap is top-down; ymin is the offset of the bitmap
            // bottom below the baseline. Using the primary face's ascent for
            // every font keeps baselines aligned across faces.
            let top = (ascent - (m.height as f32 + m.ymin as f32)).round() as i32;
            Glyph {
                w: m.width,
                h: m.height,
                left: m.xmin,
                top,
                bitmap,
            }
        })
    }
}

/// Ask fontconfig for a primary monospace face plus a set of fallback faces
/// that cover symbols/emoji the monospace font is missing. De-dups by path
/// and always tries DejaVu as a last resort.
fn load_fonts() -> Vec<Vec<u8>> {
    let patterns = [
        "monospace",
        "Noto Sans Symbols2",
        "Symbola",
        "Noto Sans Symbols",
        "DejaVu Sans",
        "Noto Color Emoji",
    ];
    let mut paths: Vec<String> = Vec::new();
    for pat in patterns {
        if let Ok(out) = std::process::Command::new("fc-match")
            .args(["-f", "%{file}", pat])
            .output()
        {
            let p = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !p.is_empty() && !paths.contains(&p) {
                paths.push(p);
            }
        }
    }
    let mut fonts: Vec<Vec<u8>> = paths.iter().filter_map(|p| std::fs::read(p).ok()).collect();
    if fonts.is_empty() {
        fonts.push(
            std::fs::read("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
                .expect("no monospace font found (install fontconfig or DejaVu Sans Mono)"),
        );
    }
    fonts
}

// Light mode by default.
const FG: u32 = 0x1a_1a_1a;
const BG: u32 = 0xff_ff_ff;
const SEL: u32 = 0xbf_d9_f2;

// --- Win2k chrome ----------------------------------------------------------
// Windows 2000 design *principles* (not its colours): explicit borders and
// bevels, a subtle vertical gradient, compact fixed heights, dense layout,
// left-aligned titles.
const SIDEBAR_W: usize = 212; // fixed-width tree pane
const HEADER_H: usize = 24; // title bar over the terminal & sidebar head
const ROW_H: usize = 20; // one tree row
const INDENT: usize = 14; // px added per tree depth
const EXPANDER_W: usize = 14; // hit width of a group's [+]/[-] box
const CTX_W: usize = 124; // context-menu width
const RPANEL_W: usize = 252; // right inspector pane width
const WBTN_W: usize = 30; // min/max/close button width
const INFO_H: usize = 24; // sidebar "info" toggle band, below WORKSPACE
const EDGE: f64 = 5.0; // borderless-window resize-grip thickness

const STRIP_BG: u32 = 0xc0_c0_c0; // chrome background
const PANEL_HI: u32 = 0xff_ff_ff; // top of a raised gradient
const PANEL_LO: u32 = 0xe4_e4_e4; // bottom of a raised gradient
const HEAD_HI: u32 = 0xd9_d9_d9; // header gradient top
const HEAD_LO: u32 = 0xbe_be_be; // header gradient bottom
const BEVEL_LT: u32 = 0xff_ff_ff; // raised highlight (top/left)
const BEVEL_DK: u32 = 0x80_80_80; // raised shadow (bottom/right)
const INK: u32 = 0x1a_1a_1a; // primary chrome text
const INK_DIM: u32 = 0x5a_5a_5a; // secondary chrome text
const RUN_INK: u32 = 0x10_7c_10; // "running" marker (green square)

/// Map a terminal color to RGB. A compact, good-enough palette.
fn rgb(color: Color, default: u32) -> u32 {
    match color {
        Color::Spec(c) => pack(c.r, c.g, c.b),
        Color::Named(n) => named_rgb(n).unwrap_or(default),
        Color::Indexed(i) => indexed_rgb(i),
    }
}

fn pack(r: u8, g: u8, b: u8) -> u32 {
    (r as u32) << 16 | (g as u32) << 8 | b as u32
}

fn named_rgb(n: NamedColor) -> Option<u32> {
    Some(match n {
        NamedColor::Black => 0x1d1f21,
        NamedColor::Red => 0xcc6666,
        NamedColor::Green => 0xb5bd68,
        NamedColor::Yellow => 0xf0c674,
        NamedColor::Blue => 0x81a2be,
        NamedColor::Magenta => 0xb294bb,
        NamedColor::Cyan => 0x8abeb7,
        NamedColor::White => 0xc5c8c6,
        NamedColor::BrightBlack => 0x666666,
        NamedColor::BrightRed => 0xd54e53,
        NamedColor::BrightGreen => 0xb9ca4a,
        NamedColor::BrightYellow => 0xe7c547,
        NamedColor::BrightBlue => 0x7aa6da,
        NamedColor::BrightMagenta => 0xc397d8,
        NamedColor::BrightCyan => 0x70c0b1,
        NamedColor::BrightWhite => 0xeaeaea,
        NamedColor::Foreground | NamedColor::BrightForeground => FG,
        NamedColor::Background => BG,
        NamedColor::Cursor => FG,
        _ => return None,
    })
}

/// Standard xterm 256-color cube + grayscale ramp.
fn indexed_rgb(i: u8) -> u32 {
    match i {
        0..=15 => named_rgb(ANSI16[i as usize]).unwrap_or(FG),
        16..=231 => {
            let i = i - 16;
            let levels = [0u8, 95, 135, 175, 215, 255];
            let r = levels[(i / 36) as usize];
            let g = levels[((i / 6) % 6) as usize];
            let b = levels[(i % 6) as usize];
            pack(r, g, b)
        }
        _ => {
            let v = 8 + (i - 232) * 10;
            pack(v, v, v)
        }
    }
}

const ANSI16: [NamedColor; 16] = [
    NamedColor::Black,
    NamedColor::Red,
    NamedColor::Green,
    NamedColor::Yellow,
    NamedColor::Blue,
    NamedColor::Magenta,
    NamedColor::Cyan,
    NamedColor::White,
    NamedColor::BrightBlack,
    NamedColor::BrightRed,
    NamedColor::BrightGreen,
    NamedColor::BrightYellow,
    NamedColor::BrightBlue,
    NamedColor::BrightMagenta,
    NamedColor::BrightCyan,
    NamedColor::BrightWhite,
];

/// One terminal session: its own PTY thread + VT state machine + grid size.
/// Reused as-is; `spawn_session` now also returns the shell pid for `/proc`.
struct Tab {
    term: Arc<FairMutex<Term<Listener>>>,
    pty_tx: EventLoopSender,
    size: TermSize,
    title: String,
}

/// Spawn a fresh PTY-backed terminal session in `workdir`, sized to the
/// current terminal area. Runs no command — it only lands a shell in the right
/// directory. Returns the session and the shell's pid (for `observe`).
fn spawn_session(
    proxy: &EventLoopProxy<UserEvent>,
    id: u64,
    title: String,
    workdir: Option<PathBuf>,
    size: TermSize,
    cell_w: usize,
    cell_h: usize,
) -> (Tab, u32) {
    let window_size = WindowSize {
        num_cols: size.cols as u16,
        num_lines: size.lines as u16,
        cell_width: cell_w as u16,
        cell_height: cell_h as u16,
    };
    let listener = Listener {
        proxy: proxy.clone(),
        id,
    };
    let term = Term::new(Config::default(), &size, listener.clone());
    let term = Arc::new(FairMutex::new(term));

    let opts = PtyOptions {
        working_directory: workdir.filter(|p| p.is_dir()),
        ..PtyOptions::default()
    };
    let pty = tty::new(&opts, window_size, 0)
        .or_else(|_| tty::new(&PtyOptions::default(), window_size, 0))
        .expect("spawn pty");
    // Capture the child shell pid before the event loop takes ownership.
    let pid = pty.child().id();

    let pty_loop =
        PtyEventLoop::new(term.clone(), listener, pty, false, false).expect("create pty event loop");
    let pty_tx = pty_loop.channel();
    pty_loop.spawn();
    (
        Tab {
            term,
            pty_tx,
            size,
            title,
        },
        pid,
    )
}

/// A live session bound to a leaf node.
struct Session {
    tab: Tab,
    shell_pid: u32,
}

// ===========================================================================
// shell — window state and the winit application
// ===========================================================================

/// State of an open right-click menu: where it was opened and on which node.
struct CtxMenu {
    x: usize,
    y: usize,
    node: NodeId,
}

/// Everything that exists once the window is created.
struct State {
    window: Rc<Window>,
    surface: softbuffer::Surface<Rc<Window>, Rc<Window>>,
    renderer: Renderer,
    tree: Tree,
    /// Leaf node -> its live PTY session. Absent = idle/never-started.
    sessions: HashMap<NodeId, Session>,
    /// PTY event id -> owning leaf node, for routing parser-thread events.
    id_of: HashMap<u64, NodeId>,
    selected: NodeId,
    next_id: u64,
    ctx: Option<CtxMenu>,
    clipboard: Option<arboard::Clipboard>,
    mouse: (f64, f64),
    selecting: bool,
    last_click: Option<(std::time::Instant, (f64, f64))>,
    mods: ModifiersState,
    /// Right inspector pane shown.
    inspector: bool,
    /// Inspector field currently captured by the keyboard (`None` = the PTY
    /// gets keystrokes as usual).
    focus: Option<Field>,
    /// Caret position (char index) within the focused field.
    caret: usize,
}

impl State {
    /// Which leaf's terminal to show: the selection if it is a leaf with a
    /// session, otherwise the first session-bearing leaf under the selection.
    fn shown(&self) -> Option<NodeId> {
        if self.tree.is_leaf(self.selected) && self.sessions.contains_key(&self.selected) {
            return Some(self.selected);
        }
        self.tree
            .leaves(self.selected)
            .into_iter()
            .find(|l| self.sessions.contains_key(l))
    }

    /// Grid size for the terminal area (window minus sidebar, header and —
    /// when open — the right inspector pane).
    fn grid_size(&self, win_w: usize, win_h: usize) -> TermSize {
        let avail = term_right(win_w, self.inspector).saturating_sub(SIDEBAR_W);
        TermSize {
            cols: (avail / self.renderer.cell_w).max(1),
            lines: (win_h.saturating_sub(HEADER_H) / self.renderer.cell_h).max(1),
        }
    }

    /// Convert the mouse position to a grid `Point` within the shown terminal,
    /// accounting for the sidebar/header offset and scrollback.
    fn pixel_to_point(&self, term: &Term<Listener>, size: TermSize) -> (Point, Side) {
        let cw = self.renderer.cell_w as f64;
        let ch = self.renderer.cell_h as f64;
        let mx = (self.mouse.0 - SIDEBAR_W as f64).max(0.0);
        let my = (self.mouse.1 - HEADER_H as f64).max(0.0);
        let col = ((mx / cw) as usize).min(size.cols.saturating_sub(1));
        let row = ((my / ch) as usize).min(size.lines.saturating_sub(1));
        let offset = term.grid().display_offset() as i32;
        let line = Line(row as i32 - offset);
        let side = if (mx / cw).fract() < 0.5 {
            Side::Left
        } else {
            Side::Right
        };
        (Point::new(line, Column(col)), side)
    }

    /// Hit-test a point against the sidebar tree. Returns the row's node id and
    /// whether the hit landed on a group's expander box.
    fn sidebar_hit(&self, x: f64, y: f64, rows: &[Row]) -> Option<(NodeId, bool)> {
        let top = HEADER_H + INFO_H; // header band + info-toggle band
        if x >= SIDEBAR_W as f64 || y < top as f64 {
            return None;
        }
        let i = (y as usize - top) / ROW_H;
        let row = rows.get(i)?;
        let ind = row.depth * INDENT;
        let on_expander =
            row.is_group && (x as usize) >= ind && (x as usize) < ind + EXPANDER_W;
        Some((row.id, on_expander))
    }
}

struct App {
    proxy: EventLoopProxy<UserEvent>,
    state: Option<State>,
}

impl App {
    /// Run a session-lifecycle plan: spawn/run/sigint each action in order.
    fn apply(&mut self, acts: Vec<Action>) {
        for act in acts {
            match act {
                Action::Spawn(node) => self.spawn_for(node),
                Action::Run(node) => {
                    if let Some((_, cmd)) = self
                        .state
                        .as_ref()
                        .and_then(|s| s.tree.leaf_spec(node).map(|(w, c)| (w, c.to_string())))
                    {
                        self.send_to(node, format!("{cmd}\r").into_bytes());
                    }
                }
                Action::Sigint(node) => self.send_to(node, vec![0x03]),
            }
        }
        if let Some(st) = &self.state {
            st.window.request_redraw();
        }
    }

    /// Effect: spawn an idle PTY for leaf `node` in its workdir.
    fn spawn_for(&mut self, node: NodeId) {
        let Some(st) = self.state.as_mut() else { return };
        if st.sessions.contains_key(&node) {
            return;
        }
        let (Some((workdir, _)), name) = (
            st.tree.leaf_spec(node).map(|(w, c)| (w.to_path_buf(), c)),
            st.tree.nodes[node].name.clone(),
        ) else {
            return;
        };
        let win = st.window.inner_size();
        let size = st.grid_size(win.width as usize, win.height as usize);
        let id = st.next_id;
        st.next_id += 1;
        let (tab, pid) = spawn_session(
            &self.proxy,
            id,
            name,
            Some(workdir),
            size,
            st.renderer.cell_w,
            st.renderer.cell_h,
        );
        st.id_of.insert(id, node);
        st.sessions.insert(node, Session { tab, shell_pid: pid });
    }

    fn send_to(&self, node: NodeId, bytes: Vec<u8>) {
        if let Some(s) = self.state.as_ref().and_then(|st| st.sessions.get(&node)) {
            let _ = s.tab.pty_tx.send(Msg::Input(bytes.into()));
        }
    }

    /// Observe every live session's `/proc` state so the pure planners can
    /// decide what's already running.
    fn observations(&self) -> HashMap<NodeId, Obs> {
        self.state
            .as_ref()
            .map(|st| {
                st.sessions
                    .iter()
                    .map(|(&n, s)| (n, observe(s.shell_pid)))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn start(&mut self, target: NodeId) {
        let obs = self.observations();
        let acts = {
            let st = self.state.as_ref().unwrap();
            let has = |n: NodeId| st.sessions.contains_key(&n);
            plan_start(&st.tree, &has, &obs, target)
        };
        self.apply(acts);
    }

    fn stop(&mut self, target: NodeId) {
        let acts = {
            let st = self.state.as_ref().unwrap();
            let has = |n: NodeId| st.sessions.contains_key(&n);
            plan_stop(&st.tree, &has, target)
        };
        self.apply(acts);
    }

    /// Create a scratch session under the group of the current selection (a
    /// selected group is its own target; Scratch/Transient are valid). It
    /// inherits the selected leaf's directory, else `$HOME`.
    fn new_scratch(&mut self) {
        let Some(st) = self.state.as_mut() else { return };
        let group = st.tree.group_for_new(st.selected);
        let home = home_dir();
        let cwd = st
            .tree
            .leaf_spec(st.selected)
            .map(|(w, _)| w.to_path_buf())
            .unwrap_or(home);
        let n = st.tree.nodes[group].children.len() + 1;
        let node = st.tree.push(
            Some(group),
            format!("{} {n}", st.tree.nodes[group].name),
            Kind::Leaf {
                workdir: cwd,
                command: String::new(),
            },
            true,
        );
        st.tree.nodes[group].expanded = true;
        st.selected = node;
        self.spawn_for(node);
        if let Some(st) = &self.state {
            st.window.request_redraw();
        }
    }

    /// Tear down the selected session. Dynamic (scratch) leaves are removed
    /// from the tree; spec leaves stay so they can be re-started.
    fn close_selected(&mut self) {
        let Some(st) = self.state.as_mut() else { return };
        let node = st.selected;
        if let Some(s) = st.sessions.remove(&node) {
            let _ = s.tab.pty_tx.send(Msg::Shutdown);
            st.id_of.retain(|_, &mut v| v != node);
        }
        if st.tree.nodes[node].dynamic {
            if let Some(p) = st.tree.nodes[node].parent {
                st.tree.nodes[p].children.retain(|&c| c != node);
                st.selected = p;
            }
        }
        st.window.request_redraw();
    }

    /// Reflow every session's grid to the current terminal area. The window
    /// is shared, so a resize *or* an inspector toggle reflows them all (not
    /// just the visible one) to keep background sessions sane.
    fn relayout(&mut self) {
        let Some(st) = self.state.as_mut() else { return };
        let win = st.window.inner_size();
        let size = st.grid_size(win.width as usize, win.height as usize);
        let ws = WindowSize {
            num_cols: size.cols as u16,
            num_lines: size.lines as u16,
            cell_width: st.renderer.cell_w as u16,
            cell_height: st.renderer.cell_h as u16,
        };
        for s in st.sessions.values_mut() {
            s.tab.size = size;
            s.tab.term.lock().resize(size);
            let _ = s.tab.pty_tx.send(Msg::Resize(ws));
        }
        st.window.request_redraw();
    }

    /// Toggle the right inspector pane (and reflow the terminal into the
    /// freed/used space). Hiding it drops keyboard focus back to the PTY.
    fn toggle_inspector(&mut self) {
        if let Some(st) = self.state.as_mut() {
            st.inspector = !st.inspector;
            if !st.inspector {
                st.focus = None;
            }
        }
        self.relayout();
    }

    /// Focus an inspector field, placing the caret under the click. `Command`
    /// on a group is not editable, so focus is refused there.
    fn focus_field(&mut self, f: Field, click_x: f64) {
        let Some(st) = self.state.as_mut() else { return };
        if f == Field::Command && st.tree.field_text(st.selected, Field::Command).is_none() {
            st.focus = None;
            return;
        }
        let text = st.tree.field_text(st.selected, f).unwrap_or_default();
        let fr = field_rects(
            st.window.inner_size().width as usize,
            st.renderer.cell_h,
        )[if f == Field::Title { 0 } else { 1 }];
        let rel = (click_x - (fr.0 + 5) as f64).max(0.0);
        let idx = (rel / st.renderer.cell_w as f64).round() as usize;
        st.focus = Some(f);
        st.caret = idx.min(text.chars().count());
        st.window.request_redraw();
    }

    /// Apply one keystroke to the focused inspector field, writing the result
    /// straight back into the tree (the tree is the single source of truth).
    /// Returns `true` if the key was consumed (kept off the PTY).
    fn edit_focused(&mut self, key: &Key, text_in: Option<&str>) -> bool {
        let Some(st) = self.state.as_mut() else { return false };
        let Some(f) = st.focus else { return false };
        let sel = st.selected;
        let e = match key {
            Key::Named(NamedKey::Escape) | Key::Named(NamedKey::Enter) => {
                st.focus = None;
                st.window.request_redraw();
                return true;
            }
            Key::Named(NamedKey::Tab) => {
                // Hop between the two fields (skip Command on a group).
                let next = if f == Field::Title { Field::Command } else { Field::Title };
                st.focus = (st.tree.field_text(sel, next).is_some()
                    || next == Field::Title)
                    .then_some(next);
                st.caret = st
                    .focus
                    .and_then(|nf| st.tree.field_text(sel, nf))
                    .map(|t| t.chars().count())
                    .unwrap_or(0);
                st.window.request_redraw();
                return true;
            }
            Key::Named(NamedKey::Backspace) => Edit::Back,
            Key::Named(NamedKey::Delete) => Edit::Del,
            Key::Named(NamedKey::ArrowLeft) => Edit::Left,
            Key::Named(NamedKey::ArrowRight) => Edit::Right,
            Key::Named(NamedKey::Home) => Edit::Home,
            Key::Named(NamedKey::End) => Edit::End,
            Key::Named(NamedKey::Space) => Edit::Ins(' '),
            _ => match text_in.and_then(|t| t.chars().next()) {
                Some(ch) if !ch.is_control() => Edit::Ins(ch),
                _ => return true, // swallow other keys while editing
            },
        };
        let cur = st.tree.field_text(sel, f).unwrap_or_default();
        let (next, caret) = apply_edit(&cur, st.caret, e);
        match f {
            Field::Title => st.tree.nodes[sel].name = next,
            Field::Command => {
                if let Some(c) = st.tree.command_mut(sel) {
                    *c = next;
                }
            }
        }
        st.caret = caret;
        st.window.request_redraw();
        true
    }

    fn redraw(&mut self) {
        let Some(st) = self.state.as_mut() else { return };
        let win = st.window.inner_size();
        let (Some(w), Some(h)) = (NonZeroU32::new(win.width), NonZeroU32::new(win.height)) else {
            return;
        };
        // `shown()` borrows all of `State`; resolve it before the framebuffer
        // takes a (disjoint, but whole-`self`-incompatible) mutable borrow.
        let shown = st.shown();
        st.surface.resize(w, h).unwrap();
        let mut buf = st.surface.buffer_mut().unwrap();
        let (pw, ph) = (win.width as usize, win.height as usize);
        buf.fill(BG);

        let cw = st.renderer.cell_w;
        let ch = st.renderer.cell_h;
        // Right edge of the terminal: the inspector pane (if open) eats into it.
        let tr = term_right(pw, st.inspector);

        // --- terminal area --------------------------------------------------
        if let Some(node) = shown {
            let term = st.sessions[&node].tab.term.lock();
            let content = term.renderable_content();
            let cursor = content.cursor.point;
            let selection = content.selection;
            for cell in content.display_iter {
                let line = cell.point.line.0;
                if line < 0 {
                    continue;
                }
                let col = cell.point.column.0;
                let x0 = SIDEBAR_W + col * cw;
                let y0 = HEADER_H + line as usize * ch;
                let is_cursor =
                    cell.point.line == cursor.line && cell.point.column == cursor.column;
                let selected = selection.is_some_and(|s| s.contains(cell.point));
                let mut fg = rgb(cell.fg, FG);
                let mut bg = rgb(cell.bg, BG);
                if is_cursor {
                    std::mem::swap(&mut fg, &mut bg);
                } else if selected {
                    bg = SEL;
                }
                if bg != BG {
                    fill_rect(&mut buf, pw, ph, x0, y0, cw, ch, bg);
                }
                let c = cell.c;
                if c == ' ' || c == '\0' {
                    continue;
                }
                let g = st.renderer.glyph(c);
                for gy in 0..g.h {
                    let py = y0 as i32 + g.top + gy as i32;
                    if py < 0 || py as usize >= ph {
                        continue;
                    }
                    for gx in 0..g.w {
                        let px = x0 as i32 + g.left + gx as i32;
                        if px < SIDEBAR_W as i32 || px as usize >= tr {
                            continue;
                        }
                        let a = g.bitmap[gy * g.w + gx] as u32;
                        if a == 0 {
                            continue;
                        }
                        let idx = py as usize * pw + px as usize;
                        buf[idx] = blend(fg, buf[idx], a);
                    }
                }
            }
            drop(term);
        } else {
            draw_text(
                &mut buf,
                pw,
                ph,
                &mut st.renderer,
                SIDEBAR_W + 10,
                HEADER_H + 10,
                tr.saturating_sub(SIDEBAR_W + 20),
                "No session here. Right-click a node \u{2192} Start, or Ctrl+Shift+T for a shell.",
                INK_DIM,
            );
        }

        // --- header bar over the terminal ----------------------------------
        let running = shown.is_some_and(|n| {
            st.sessions
                .get(&n)
                .map(|s| observe(s.shell_pid).foreground)
                .unwrap_or(false)
        });
        let title = match shown {
            Some(n) => format!(
                "{}      [{}]",
                st.tree.path(n),
                if running { "running" } else { "idle" }
            ),
            None => "termem".to_string(),
        };
        vgradient(&mut buf, pw, ph, SIDEBAR_W, 0, pw - SIDEBAR_W, HEADER_H, HEAD_HI, HEAD_LO);
        fill_rect(&mut buf, pw, ph, SIDEBAR_W, HEADER_H - 1, pw - SIDEBAR_W, 1, BEVEL_DK);
        fill_rect(&mut buf, pw, ph, SIDEBAR_W, 0, pw - SIDEBAR_W, 1, BEVEL_LT);
        let ty = HEADER_H.saturating_sub(st.renderer.cell_h) / 2;
        let maxed = st.window.is_maximized();
        let [bmin, bmax, bclose] = win_btns(pw);
        draw_text(
            &mut buf,
            pw,
            ph,
            &mut st.renderer,
            SIDEBAR_W + 8,
            ty,
            bmin.0.saturating_sub(SIDEBAR_W + 16),
            &title,
            INK,
        );

        // Window controls, right-aligned in the header row.
        for (rect, label) in [
            (bmin, "\u{2013}"),                                // –  minimize
            (bmax, if maxed { "\u{2750}" } else { "\u{25a1}" }), // ▢ / ❐
            (bclose, "\u{2715}"),                              // ✕  close
        ] {
            draw_button(&mut buf, pw, ph, &mut st.renderer, rect, label, false);
        }

        // --- sidebar tree ---------------------------------------------------
        let rows = st.tree.rows();
        let sel = st.selected;
        draw_sidebar(
            &mut buf,
            pw,
            ph,
            &mut st.renderer,
            &rows,
            sel,
            &st.sessions,
            st.inspector,
        );

        // --- right inspector ------------------------------------------------
        if st.inspector {
            draw_inspector(
                &mut buf,
                pw,
                ph,
                &mut st.renderer,
                &st.tree,
                sel,
                st.focus,
                st.caret,
            );
        }

        // --- context menu ---------------------------------------------------
        if let Some(m) = &st.ctx {
            draw_ctx_menu(&mut buf, pw, ph, &mut st.renderer, m.x, m.y);
        }

        buf.present().unwrap();
    }

    fn send(&self, bytes: Vec<u8>) {
        if let Some(node) = self.state.as_ref().and_then(|s| s.shown()) {
            self.send_to(node, bytes);
        }
    }

    fn copy_to_clipboard(&mut self) {
        let Some(st) = self.state.as_mut() else { return };
        let Some(node) = st.shown() else { return };
        let text = st.sessions[&node].tab.term.lock().selection_to_string();
        if let (Some(text), Some(cb)) = (text, st.clipboard.as_mut()) {
            if !text.is_empty() {
                let _ = cb.set_text(text);
            }
        }
    }

    fn paste(&mut self) {
        let text = self
            .state
            .as_mut()
            .and_then(|st| st.clipboard.as_mut())
            .and_then(|cb| cb.get_text().ok());
        if let Some(text) = text {
            self.send(text.replace('\n', "\r").into_bytes());
        }
    }
}

// --- drawing helpers (unchanged) -------------------------------------------

fn fill_rect(buf: &mut [u32], pw: usize, ph: usize, x: usize, y: usize, w: usize, h: usize, color: u32) {
    for yy in y..(y + h).min(ph) {
        for xx in x..(x + w).min(pw) {
            buf[yy * pw + xx] = color;
        }
    }
}

/// Vertical gradient fill: row `y` lerps from `top` to `bot`.
fn vgradient(buf: &mut [u32], pw: usize, ph: usize, x: usize, y: usize, w: usize, h: usize, top: u32, bot: u32) {
    for row in 0..h {
        let yy = y + row;
        if yy >= ph {
            break;
        }
        let t = if h > 1 { (row * 255 / (h - 1)) as u32 } else { 0 };
        let color = blend(bot, top, t);
        for col in 0..w {
            let xx = x + col;
            if xx >= pw {
                break;
            }
            buf[yy * pw + xx] = color;
        }
    }
}

/// 1px rectangle outline.
fn stroke_rect(buf: &mut [u32], pw: usize, ph: usize, x: usize, y: usize, w: usize, h: usize, color: u32) {
    if w == 0 || h == 0 {
        return;
    }
    fill_rect(buf, pw, ph, x, y, w, 1, color);
    fill_rect(buf, pw, ph, x, y + h - 1, w, 1, color);
    fill_rect(buf, pw, ph, x, y, 1, h, color);
    fill_rect(buf, pw, ph, x + w - 1, y, 1, h, color);
}

/// Draw a left-aligned monospace string, clipped to `max_w` pixels.
fn draw_text(
    buf: &mut [u32],
    pw: usize,
    ph: usize,
    r: &mut Renderer,
    x: usize,
    y: usize,
    max_w: usize,
    text: &str,
    color: u32,
) {
    let cw = r.cell_w;
    let mut pen = x;
    for c in text.chars() {
        if pen + cw > x + max_w {
            break;
        }
        if c != ' ' {
            let g = r.glyph(c);
            let (gw, gh, gl, gt) = (g.w, g.h, g.left, g.top);
            for gy in 0..gh {
                let py = y as i32 + gt + gy as i32;
                if py < 0 || py as usize >= ph {
                    continue;
                }
                for gx in 0..gw {
                    let px = pen as i32 + gl + gx as i32;
                    if px < 0 || px as usize >= pw {
                        continue;
                    }
                    let a = r.cache[&c].bitmap[gy * gw + gx] as u32;
                    if a == 0 {
                        continue;
                    }
                    let idx = py as usize * pw + px as usize;
                    buf[idx] = blend(color, buf[idx], a);
                }
            }
        }
        pen += cw;
    }
}

/// The left tree pane: a recessed panel with a Win2k head, one fixed-height
/// bevelled row per visible node, boxed `[+]/[-]` expanders for groups, a
/// run-state marker for leaves, and a raised selection highlight.
fn draw_sidebar(
    buf: &mut [u32],
    pw: usize,
    ph: usize,
    r: &mut Renderer,
    rows: &[Row],
    selected: NodeId,
    sessions: &HashMap<NodeId, Session>,
    inspector: bool,
) {
    fill_rect(buf, pw, ph, 0, 0, SIDEBAR_W, ph, STRIP_BG);
    // Header band, matching the terminal header height.
    vgradient(buf, pw, ph, 0, 0, SIDEBAR_W, HEADER_H, HEAD_HI, HEAD_LO);
    fill_rect(buf, pw, ph, 0, 0, SIDEBAR_W, 1, BEVEL_LT);
    fill_rect(buf, pw, ph, 0, HEADER_H - 1, SIDEBAR_W, 1, BEVEL_DK);
    let ty = HEADER_H.saturating_sub(r.cell_h) / 2;
    draw_text(buf, pw, ph, r, 8, ty, SIDEBAR_W - 16, "WORKSPACE", INK);
    // Hard divider between the pane and the terminal.
    fill_rect(buf, pw, ph, SIDEBAR_W - 1, 0, 1, ph, BEVEL_DK);
    // Latched info icon directly below the WORKSPACE head.
    draw_info_icon(buf, pw, ph, info_btn(), inspector);

    for (i, row) in rows.iter().enumerate() {
        let y = HEADER_H + INFO_H + i * ROW_H;
        if y + ROW_H > ph {
            break;
        }
        let ind = row.depth * INDENT;
        let is_sel = row.id == selected;
        if is_sel {
            vgradient(buf, pw, ph, 1, y, SIDEBAR_W - 2, ROW_H, PANEL_HI, PANEL_LO);
            stroke_rect(buf, pw, ph, 1, y, SIDEBAR_W - 2, ROW_H, BEVEL_DK);
            fill_rect(buf, pw, ph, 1, y, SIDEBAR_W - 2, 1, BEVEL_LT);
        }
        let gy = y + ty;
        if row.is_group {
            // Boxed expander, classic tree control.
            let bx = 1 + ind;
            stroke_rect(buf, pw, ph, bx, y + 4, EXPANDER_W - 4, ROW_H - 8, BEVEL_DK);
            let mark = if row.expanded { "-" } else { "+" };
            let glyph_ok = row.has_children;
            draw_text(
                buf,
                pw,
                ph,
                r,
                bx + 2,
                gy,
                EXPANDER_W,
                if glyph_ok { mark } else { " " },
                INK,
            );
            draw_text(
                buf,
                pw,
                ph,
                r,
                1 + ind + EXPANDER_W + 2,
                gy,
                SIDEBAR_W.saturating_sub(ind + EXPANDER_W + 8),
                &row.name,
                INK,
            );
        } else {
            let live = sessions.contains_key(&row.id);
            // Run-state marker: filled = running a command, hollow = idle.
            let running = sessions
                .get(&row.id)
                .map(|s| observe(s.shell_pid).foreground)
                .unwrap_or(false);
            let (mark, mc) = if running {
                ("\u{25a0}", RUN_INK) // ■
            } else if live {
                ("\u{25a1}", INK_DIM) // □
            } else {
                ("\u{00b7}", INK_DIM) // ·
            };
            draw_text(buf, pw, ph, r, 1 + ind + 4, gy, EXPANDER_W, mark, mc);
            draw_text(
                buf,
                pw,
                ph,
                r,
                1 + ind + EXPANDER_W + 4,
                gy,
                SIDEBAR_W.saturating_sub(ind + EXPANDER_W + 10),
                &row.name,
                if is_sel { INK } else { INK_DIM },
            );
        }
    }
}

/// A small bevelled Start/Stop popup at the cursor.
fn draw_ctx_menu(buf: &mut [u32], pw: usize, ph: usize, r: &mut Renderer, x: usize, y: usize) {
    let h = ROW_H * 2 + 2;
    vgradient(buf, pw, ph, x, y, CTX_W, h, PANEL_HI, PANEL_LO);
    stroke_rect(buf, pw, ph, x, y, CTX_W, h, BEVEL_DK);
    fill_rect(buf, pw, ph, x, y, CTX_W, 1, BEVEL_LT);
    fill_rect(buf, pw, ph, x, y, 1, h, BEVEL_LT);
    let ty = ROW_H.saturating_sub(r.cell_h) / 2;
    draw_text(buf, pw, ph, r, x + 10, y + 1 + ty, CTX_W, "Start", INK);
    fill_rect(buf, pw, ph, x + 4, y + ROW_H, CTX_W - 8, 1, BEVEL_DK);
    draw_text(buf, pw, ph, r, x + 10, y + ROW_H + 1 + ty, CTX_W, "Stop", INK);
}

/// Which context-menu item (if any) the point falls on. `0` = Start, `1` =
/// Stop, `None` = outside the menu (dismiss).
fn ctx_item_at(m: &CtxMenu, x: f64, y: f64) -> Option<usize> {
    let (mx, my) = (m.x as f64, m.y as f64);
    let h = (ROW_H * 2 + 2) as f64;
    if x < mx || x >= mx + CTX_W as f64 || y < my || y >= my + h {
        return None;
    }
    Some(if (y - my) < ROW_H as f64 { 0 } else { 1 })
}

/// A Win2k push-button: raised by default, sunken+inset when `pressed`
/// (used for the latched "Properties" toggle and the window controls).
fn draw_button(buf: &mut [u32], pw: usize, ph: usize, r: &mut Renderer, rect: Rect, label: &str, pressed: bool) {
    let (x, y, w, h) = rect;
    if pressed {
        fill_rect(buf, pw, ph, x, y, w, h, PANEL_LO);
        fill_rect(buf, pw, ph, x, y, w, 1, BEVEL_DK);
        fill_rect(buf, pw, ph, x, y, 1, h, BEVEL_DK);
        fill_rect(buf, pw, ph, x, y + h - 1, w, 1, BEVEL_LT);
        fill_rect(buf, pw, ph, x + w - 1, y, 1, h, BEVEL_LT);
    } else {
        vgradient(buf, pw, ph, x, y, w, h, PANEL_HI, PANEL_LO);
        stroke_rect(buf, pw, ph, x, y, w, h, BEVEL_DK);
        fill_rect(buf, pw, ph, x, y, w, 1, BEVEL_LT);
        fill_rect(buf, pw, ph, x, y, 1, h, BEVEL_LT);
    }
    let tw = label.chars().count() * r.cell_w;
    let off = pressed as usize;
    let tx = x + w.saturating_sub(tw) / 2 + off;
    let ty = y + h.saturating_sub(r.cell_h) / 2 + off;
    draw_text(buf, pw, ph, r, tx, ty, w, label, INK);
}

/// A small square info icon (1px border, a drawn "i"). Not a button: no
/// bevel/gradient. `active` inverts it (filled box, light glyph) to show the
/// inspector pane is open.
fn draw_info_icon(buf: &mut [u32], pw: usize, ph: usize, rect: Rect, active: bool) {
    let (x, y, w, h) = rect;
    let (face, glyph) = if active { (INK, BG) } else { (BG, INK) };
    fill_rect(buf, pw, ph, x, y, w, h, face);
    stroke_rect(buf, pw, ph, x, y, w, h, INK);
    // The "i": a dot near the top and a stem below, both centered.
    let d = (w / 6).max(2);
    let cx = x + (w - d) / 2;
    fill_rect(buf, pw, ph, cx, y + h / 5, d, d, glyph);
    fill_rect(buf, pw, ph, cx, y + h * 2 / 5, d, h * 2 / 5, glyph);
}

/// A sunken Win2k text box. `focused` draws the caret; `enabled` is false for
/// the command field on a group (no command to edit).
fn draw_field(
    buf: &mut [u32],
    pw: usize,
    ph: usize,
    r: &mut Renderer,
    rect: Rect,
    text: &str,
    focused: bool,
    caret: usize,
    enabled: bool,
) {
    let (x, y, w, h) = rect;
    fill_rect(buf, pw, ph, x, y, w, h, if enabled { BG } else { STRIP_BG });
    // Inset bevel: shadow on top/left, highlight on bottom/right.
    fill_rect(buf, pw, ph, x, y, w, 1, BEVEL_DK);
    fill_rect(buf, pw, ph, x, y, 1, h, BEVEL_DK);
    fill_rect(buf, pw, ph, x, y + h - 1, w, 1, BEVEL_LT);
    fill_rect(buf, pw, ph, x + w - 1, y, 1, h, BEVEL_LT);
    let tx = x + 5;
    let ty = y + h.saturating_sub(r.cell_h) / 2;
    draw_text(
        buf,
        pw,
        ph,
        r,
        tx,
        ty,
        w.saturating_sub(10),
        text,
        if enabled { INK } else { INK_DIM },
    );
    if focused {
        let cx = tx + caret.min(text.chars().count()) * r.cell_w;
        fill_rect(buf, pw, ph, cx, y + 3, 1, h.saturating_sub(6), INK);
    }
}

/// The right inspector pane: a recessed panel echoing the sidebar, with the
/// selected node's path, an editable Title and Default-command field, and the
/// (read-only) working directory.
fn draw_inspector(
    buf: &mut [u32],
    pw: usize,
    ph: usize,
    r: &mut Renderer,
    tree: &Tree,
    sel: NodeId,
    focus: Option<Field>,
    caret: usize,
) {
    let px = panel_x(pw);
    fill_rect(buf, pw, ph, px, 0, RPANEL_W, ph, STRIP_BG);
    fill_rect(buf, pw, ph, px, 0, 1, ph, BEVEL_DK); // hard divider
    vgradient(buf, pw, ph, px, 0, RPANEL_W, HEADER_H, HEAD_HI, HEAD_LO);
    fill_rect(buf, pw, ph, px, 0, RPANEL_W, 1, BEVEL_LT);
    fill_rect(buf, pw, ph, px, HEADER_H - 1, RPANEL_W, 1, BEVEL_DK);
    let hty = HEADER_H.saturating_sub(r.cell_h) / 2;
    draw_text(buf, pw, ph, r, px + 10, hty, RPANEL_W - 20, "PROPERTIES", INK);
    draw_text(
        buf,
        pw,
        ph,
        r,
        px + 12,
        HEADER_H + 8,
        RPANEL_W - 24,
        &tree.path(sel),
        INK_DIM,
    );

    let cell_h = r.cell_h;
    let [tb, cb] = field_rects(pw, cell_h);
    let lab_y = |by: usize| by.saturating_sub(cell_h + 4);
    draw_text(buf, pw, ph, r, tb.0, lab_y(tb.1), RPANEL_W, "Title", INK);
    draw_field(
        buf,
        pw,
        ph,
        r,
        tb,
        &tree.field_text(sel, Field::Title).unwrap_or_default(),
        focus == Some(Field::Title),
        caret,
        true,
    );

    draw_text(buf, pw, ph, r, cb.0, lab_y(cb.1), RPANEL_W, "Default command", INK);
    match tree.field_text(sel, Field::Command) {
        Some(cmd) => draw_field(
            buf,
            pw,
            ph,
            r,
            cb,
            &cmd,
            focus == Some(Field::Command),
            caret,
            true,
        ),
        None => draw_field(buf, pw, ph, r, cb, "(group \u{2014} no command)", false, 0, false),
    }

    if let Some((wd, _)) = tree.leaf_spec(sel) {
        let dy = cb.1 + cb.3 + 16;
        draw_text(buf, pw, ph, r, cb.0, dy, RPANEL_W, "Directory", INK);
        draw_text(
            buf,
            pw,
            ph,
            r,
            cb.0,
            dy + r.cell_h + 4,
            RPANEL_W - 24,
            &wd.to_string_lossy(),
            INK_DIM,
        );
    }
}

// --- chrome geometry -------------------------------------------------------
// One source of truth for every clickable chrome rect, shared by draw and
// hit-test (the same discipline as `Tree::rows`): a pixel can't be drawn one
// place and clicked another.

type Rect = (usize, usize, usize, usize); // x, y, w, h

fn hit(r: Rect, x: f64, y: f64) -> bool {
    x >= r.0 as f64 && x < (r.0 + r.2) as f64 && y >= r.1 as f64 && y < (r.1 + r.3) as f64
}

/// Left edge of the right inspector pane.
fn panel_x(pw: usize) -> usize {
    pw.saturating_sub(RPANEL_W)
}

/// Right edge of the live terminal area (shrinks when the inspector is open).
fn term_right(pw: usize, inspector: bool) -> usize {
    if inspector { panel_x(pw) } else { pw }
}

/// `[minimize, maximize, close]`, square, flush to the window's top-right.
fn win_btns(pw: usize) -> [Rect; 3] {
    let c = pw.saturating_sub(WBTN_W);
    [
        (c - 2 * WBTN_W, 0, WBTN_W, HEADER_H),
        (c - WBTN_W, 0, WBTN_W, HEADER_H),
        (c, 0, WBTN_W, HEADER_H),
    ]
}

/// The latched info icon — a small square box, left-aligned in its own band
/// directly below the WORKSPACE head. Filled = the inspector pane is shown.
fn info_btn() -> Rect {
    let s = INFO_H - 8; // square side
    (6, HEADER_H + (INFO_H - s) / 2, s, s)
}

/// `[title box, command box]` inside the inspector pane.
fn field_rects(pw: usize, cell_h: usize) -> [Rect; 2] {
    let px = panel_x(pw);
    let pad = 12;
    let fx = px + pad;
    let fw = RPANEL_W - pad * 2;
    let bh = cell_h + 8;
    // Each box sits exactly `cell_h + 4` below its label (see `lab_y`), and
    // the first label clears the path line under the PROPERTIES head.
    let title_lab = HEADER_H + 8 + cell_h + 10;
    let title_y = title_lab + cell_h + 4;
    let cmd_lab = title_y + bh + 16;
    let cmd_y = cmd_lab + cell_h + 4;
    [(fx, title_y, fw, bh), (fx, cmd_y, fw, bh)]
}

/// Which resize grip (if any) the point is in, for a borderless window.
fn resize_dir(pw: usize, ph: usize, x: f64, y: f64) -> Option<ResizeDirection> {
    let l = x < EDGE;
    let r = x >= pw as f64 - EDGE;
    let t = y < EDGE;
    let b = y >= ph as f64 - EDGE;
    Some(match (t, b, l, r) {
        (true, _, true, _) => ResizeDirection::NorthWest,
        (true, _, _, true) => ResizeDirection::NorthEast,
        (_, true, true, _) => ResizeDirection::SouthWest,
        (_, true, _, true) => ResizeDirection::SouthEast,
        (true, ..) => ResizeDirection::North,
        (_, true, ..) => ResizeDirection::South,
        (_, _, true, _) => ResizeDirection::West,
        (_, _, _, true) => ResizeDirection::East,
        _ => return None,
    })
}

// --- gamma-correct alpha blending (unchanged) ------------------------------

/// sRGB(0..=255) -> linear-light(0..=1) lookup table. Blending glyph coverage
/// in linear light (not raw sRGB) is what makes anti-aliased text crisp and
/// correctly weighted instead of muddy.
fn srgb_lut() -> &'static [f32; 256] {
    static LUT: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = [0.0f32; 256];
        for (i, v) in t.iter_mut().enumerate() {
            let c = i as f32 / 255.0;
            *v = if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            };
        }
        t
    })
}

/// linear-light(0..=1) -> sRGB(0..=255), rounded.
fn lin_to_srgb(v: f32) -> u32 {
    let v = v.clamp(0.0, 1.0);
    let s = if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0 + 0.5) as u32
}

/// Alpha-blend `fg` over `bg` with coverage `a` (0..=255), gamma-correct.
fn blend(fg: u32, bg: u32, a: u32) -> u32 {
    let lut = srgb_lut();
    let t = a as f32 / 255.0;
    let mix = |shift: u32| -> u32 {
        let s = lut[((fg >> shift) & 0xff) as usize];
        let d = lut[((bg >> shift) & 0xff) as usize];
        lin_to_srgb(d + (s - d) * t)
    };
    mix(16) << 16 | mix(8) << 8 | mix(0)
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/"))
}

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        // No OS title bar: we draw our own in the header row to reclaim that
        // strip of screen.
        let window = Rc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("termem")
                        .with_decorations(false),
                )
                .expect("create window"),
        );
        let renderer = Renderer::new();
        let ctx = softbuffer::Context::new(window.clone()).unwrap();
        let surface = softbuffer::Surface::new(&ctx, window.clone()).unwrap();

        let home = home_dir();
        let ws_text = std::fs::read_to_string("workspace").unwrap_or_default();
        let tree = parse_workspace(&ws_text, &home);

        let inner = window.inner_size();
        let mut st = State {
            window,
            surface,
            renderer,
            selected: tree
                .first_leaf(tree.root)
                .unwrap_or(tree.root),
            tree,
            sessions: HashMap::new(),
            id_of: HashMap::new(),
            next_id: 0,
            ctx: None,
            clipboard: arboard::Clipboard::new().ok(),
            mouse: (0.0, 0.0),
            selecting: false,
            last_click: None,
            mods: ModifiersState::empty(),
            inspector: false,
            focus: None,
            caret: 0,
        };
        let size = st.grid_size(inner.width as usize, inner.height as usize);

        // On open: an idle PTY per spec leaf — cwd set, no command run.
        let leaves = st.tree.leaves(st.tree.root);
        for node in leaves {
            let (Some((workdir, _)), name) = (
                st.tree.leaf_spec(node).map(|(w, c)| (w.to_path_buf(), c)),
                st.tree.nodes[node].name.clone(),
            ) else {
                continue;
            };
            let id = st.next_id;
            st.next_id += 1;
            let (tab, pid) = spawn_session(
                &self.proxy,
                id,
                name,
                Some(workdir),
                size,
                st.renderer.cell_w,
                st.renderer.cell_h,
            );
            st.id_of.insert(id, node);
            st.sessions.insert(node, Session { tab, shell_pid: pid });
        }
        self.state = Some(st);
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::Wakeup => {
                if let Some(st) = &self.state {
                    st.window.request_redraw();
                }
            }
            UserEvent::Exit(id) => {
                if let Some(st) = self.state.as_mut() {
                    if let Some(&node) = st.id_of.get(&id) {
                        if let Some(s) = st.sessions.remove(&node) {
                            let _ = s.tab.pty_tx.send(Msg::Shutdown);
                        }
                        st.id_of.remove(&id);
                        // A scratch tab disappears when its shell exits; a
                        // spec leaf stays so Start can bring it back.
                        if st.tree.nodes[node].dynamic {
                            if let Some(p) = st.tree.nodes[node].parent {
                                st.tree.nodes[p].children.retain(|&c| c != node);
                                if st.selected == node {
                                    st.selected = p;
                                }
                            }
                        }
                        st.window.request_redraw();
                    }
                }
            }
            UserEvent::Title(id, title) => {
                if let Some(st) = self.state.as_mut() {
                    if let Some(&node) = st.id_of.get(&id) {
                        if let Some(s) = st.sessions.get_mut(&node) {
                            s.tab.title = title;
                        }
                        st.window.request_redraw();
                    }
                }
            }
            UserEvent::ResetTitle(id) => {
                if let Some(st) = self.state.as_mut() {
                    if let Some(&node) = st.id_of.get(&id) {
                        let name = st.tree.nodes[node].name.clone();
                        if let Some(s) = st.sessions.get_mut(&node) {
                            s.tab.title = name;
                        }
                        st.window.request_redraw();
                    }
                }
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => self.redraw(),
            WindowEvent::Resized(_) => self.relayout(),
            WindowEvent::ModifiersChanged(m) => {
                if let Some(st) = self.state.as_mut() {
                    st.mods = m.state();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(st) = self.state.as_mut() {
                    st.mouse = (position.x, position.y);
                    if st.selecting {
                        if let Some(node) = st.shown() {
                            let size = st.sessions[&node].tab.size;
                            let mut term = st.sessions[&node].tab.term.lock();
                            let (point, side) = st.pixel_to_point(&term, size);
                            if let Some(sel) = term.selection.as_mut() {
                                sel.update(point, side);
                            }
                            drop(term);
                            st.window.request_redraw();
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let Some(stref) = self.state.as_ref() else { return };
                let (mx, my) = stref.mouse;
                match (button, state) {
                    (MouseButton::Left, ElementState::Pressed) => {
                        let sz = self.state.as_ref().unwrap().window.inner_size();
                        let (pw, ph) = (sz.width as usize, sz.height as usize);

                        // 1. A click anywhere resolves an open context menu.
                        if let Some(m) = &self.state.as_ref().unwrap().ctx {
                            let pick = ctx_item_at(m, mx, my);
                            let node = m.node;
                            self.state.as_mut().unwrap().ctx = None;
                            match pick {
                                Some(0) => self.start(node),
                                Some(1) => self.stop(node),
                                _ => {}
                            }
                            if let Some(st) = &self.state {
                                st.window.request_redraw();
                            }
                            return;
                        }
                        // 2. Window controls: minimize / maximize / close.
                        let [bmin, bmax, bclose] = win_btns(pw);
                        if hit(bclose, mx, my) {
                            event_loop.exit();
                            return;
                        }
                        if hit(bmin, mx, my) {
                            self.state.as_ref().unwrap().window.set_minimized(true);
                            return;
                        }
                        if hit(bmax, mx, my) {
                            let w = self.state.as_ref().unwrap().window.clone();
                            w.set_maximized(!w.is_maximized());
                            return;
                        }
                        // 3. The sidebar "info" toggle (below WORKSPACE).
                        if hit(info_btn(), mx, my) {
                            self.toggle_inspector();
                            return;
                        }
                        // 4. Inspector pane: focus a field, else defocus.
                        if self.state.as_ref().unwrap().inspector
                            && mx >= panel_x(pw) as f64
                        {
                            let ch = self.state.as_ref().unwrap().renderer.cell_h;
                            let [tb, cb] = field_rects(pw, ch);
                            if hit(tb, mx, my) {
                                self.focus_field(Field::Title, mx);
                            } else if hit(cb, mx, my) {
                                self.focus_field(Field::Command, mx);
                            } else if let Some(st) = self.state.as_mut() {
                                st.focus = None;
                                st.window.request_redraw();
                            }
                            return;
                        }
                        // 5. Borderless-window resize grips.
                        if let Some(dir) = resize_dir(pw, ph, mx, my) {
                            let _ = self
                                .state
                                .as_ref()
                                .unwrap()
                                .window
                                .drag_resize_window(dir);
                            return;
                        }
                        // 6. The header row (anywhere else) drags the window.
                        if my < HEADER_H as f64 {
                            let _ = self.state.as_ref().unwrap().window.drag_window();
                            return;
                        }
                        // 7. Sidebar: expander toggles fold, row selects.
                        let rows = self.state.as_ref().unwrap().tree.rows();
                        if let Some((node, on_exp)) =
                            self.state.as_ref().unwrap().sidebar_hit(mx, my, &rows)
                        {
                            let st = self.state.as_mut().unwrap();
                            if on_exp {
                                let e = &mut st.tree.nodes[node].expanded;
                                *e = !*e;
                            } else {
                                st.selected = node;
                                st.focus = None;
                            }
                            st.window.request_redraw();
                            return;
                        }
                        // 8. Terminal area: start a text selection.
                        let tr = term_right(pw, self.state.as_ref().unwrap().inspector);
                        if mx >= SIDEBAR_W as f64
                            && (mx as usize) < tr
                            && my >= HEADER_H as f64
                        {
                            if let Some(node) = self.state.as_ref().unwrap().shown() {
                                let now = std::time::Instant::now();
                                let st = self.state.as_mut().unwrap();
                                st.focus = None;
                                let double = st.last_click.is_some_and(|(t, p)| {
                                    now.duration_since(t).as_millis() < 400
                                        && (p.0 - mx).abs() < 4.0
                                        && (p.1 - my).abs() < 4.0
                                });
                                st.last_click = Some((now, (mx, my)));
                                let ty = if double {
                                    SelectionType::Semantic
                                } else {
                                    SelectionType::Simple
                                };
                                let size = st.sessions[&node].tab.size;
                                let mut term = st.sessions[&node].tab.term.lock();
                                let (point, side) = st.pixel_to_point(&term, size);
                                term.selection = Some(Selection::new(ty, point, side));
                                drop(term);
                                st.selecting = true;
                                st.window.request_redraw();
                            }
                        }
                    }
                    (MouseButton::Left, ElementState::Released) => {
                        if let Some(st) = self.state.as_mut() {
                            st.selecting = false;
                        }
                        self.copy_to_clipboard();
                    }
                    (MouseButton::Right, ElementState::Pressed) => {
                        let rows = self.state.as_ref().unwrap().tree.rows();
                        if let Some((node, _)) =
                            self.state.as_ref().unwrap().sidebar_hit(mx, my, &rows)
                        {
                            let st = self.state.as_mut().unwrap();
                            st.selected = node;
                            st.focus = None;
                            st.ctx = Some(CtxMenu {
                                x: (mx as usize).min(SIDEBAR_W),
                                y: my as usize,
                                node,
                            });
                            st.window.request_redraw();
                        } else if let Some(st) = self.state.as_mut() {
                            st.ctx = None;
                            st.window.request_redraw();
                        }
                    }
                    (MouseButton::Middle, ElementState::Pressed) => self.paste(),
                    _ => {}
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state != ElementState::Pressed {
                    return;
                }
                // While an inspector field is focused, the keyboard edits it
                // instead of feeding the PTY.
                if self
                    .state
                    .as_ref()
                    .is_some_and(|s| s.inspector && s.focus.is_some())
                {
                    let txt = event.text.as_ref().map(|s| s.to_string());
                    if self.edit_focused(&event.logical_key, txt.as_deref()) {
                        return;
                    }
                }
                let kmods = self.state.as_ref().map(|s| s.mods).unwrap_or_default();
                if kmods.control_key() && kmods.shift_key() {
                    match &event.logical_key {
                        Key::Character(c) if c.eq_ignore_ascii_case("c") => {
                            self.copy_to_clipboard();
                            return;
                        }
                        Key::Character(c) if c.eq_ignore_ascii_case("v") => {
                            self.paste();
                            return;
                        }
                        Key::Character(c) if c.eq_ignore_ascii_case("t") => {
                            self.new_scratch();
                            return;
                        }
                        Key::Character(c) if c.eq_ignore_ascii_case("w") => {
                            self.close_selected();
                            return;
                        }
                        Key::Character(c) if c.eq_ignore_ascii_case("s") => {
                            if let Some(n) = self.state.as_ref().map(|s| s.selected) {
                                self.start(n);
                            }
                            return;
                        }
                        Key::Character(c) if c.eq_ignore_ascii_case("x") => {
                            if let Some(n) = self.state.as_ref().map(|s| s.selected) {
                                self.stop(n);
                            }
                            return;
                        }
                        _ => {}
                    }
                }
                let mods = kmods;
                let (ctrl, alt, shift) = (mods.control_key(), mods.alt_key(), mods.shift_key());
                // xterm modifier parameter: 1 + shift + 2*alt + 4*ctrl.
                let m = 1 + shift as u8 + 2 * alt as u8 + 4 * ctrl as u8;
                let csi = |tail: &str| -> Vec<u8> {
                    if m > 1 {
                        format!("\x1b[1;{m}{tail}").into_bytes()
                    } else {
                        format!("\x1b[{tail}").into_bytes()
                    }
                };
                let tilde = |n: u8| -> Vec<u8> {
                    if m > 1 {
                        format!("\x1b[{n};{m}~").into_bytes()
                    } else {
                        format!("\x1b[{n}~").into_bytes()
                    }
                };
                let bytes: Vec<u8> = match &event.logical_key {
                    Key::Named(NamedKey::Enter) => vec![b'\r'],
                    Key::Named(NamedKey::Backspace) => {
                        if ctrl { b"\x17".to_vec() } else { vec![0x7f] }
                    }
                    Key::Named(NamedKey::Tab) => {
                        if shift { b"\x1b[Z".to_vec() } else { vec![b'\t'] }
                    }
                    Key::Named(NamedKey::Escape) => vec![0x1b],
                    Key::Named(NamedKey::ArrowUp) => csi("A"),
                    Key::Named(NamedKey::ArrowDown) => csi("B"),
                    Key::Named(NamedKey::ArrowRight) => csi("C"),
                    Key::Named(NamedKey::ArrowLeft) => csi("D"),
                    Key::Named(NamedKey::Home) => csi("H"),
                    Key::Named(NamedKey::End) => csi("F"),
                    Key::Named(NamedKey::Delete) => tilde(3),
                    Key::Named(NamedKey::PageUp) => tilde(5),
                    Key::Named(NamedKey::PageDown) => tilde(6),
                    Key::Named(NamedKey::Space) => {
                        if ctrl { vec![0] } else { vec![b' '] }
                    }
                    Key::Character(c) if ctrl => match c.chars().next() {
                        Some(ch) if ch.is_ascii() => {
                            let b = (ch as u8).to_ascii_uppercase();
                            let ctl = match b {
                                b'@'..=b'_' => b & 0x1f,
                                b' ' => 0,
                                b'?' => 0x7f,
                                _ => return,
                            };
                            if alt { vec![0x1b, ctl] } else { vec![ctl] }
                        }
                        _ => return,
                    },
                    _ => match event.text {
                        Some(ref t) if alt => {
                            let mut v = vec![0x1b];
                            v.extend_from_slice(t.as_bytes());
                            v
                        }
                        Some(ref t) => t.as_bytes().to_vec(),
                        None => return,
                    },
                };
                self.send(bytes);
            }
            _ => {}
        }
    }
}

fn main() {
    let event_loop = EventLoop::<UserEvent>::with_user_event()
        .build()
        .expect("build event loop");
    event_loop.set_control_flow(ControlFlow::Wait);
    let proxy = event_loop.create_proxy();
    let mut app = App { proxy, state: None };
    event_loop.run_app(&mut app).expect("run");
}

// ===========================================================================
// tests — the functional core, exercised without a display
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn tree() -> Tree {
        parse_workspace(include_str!("../workspace"), Path::new("/home/u"))
    }

    fn group(t: &Tree, name: &str) -> NodeId {
        t.nodes
            .iter()
            .position(|n| n.name == name && matches!(n.kind, Kind::Group))
            .unwrap()
    }

    #[test]
    fn parses_groups_leaves_and_synthetic_areas() {
        let t = tree();
        // Root child groups, in file order, then the two synthetic ones.
        let groups: Vec<&str> = t.nodes[t.root]
            .children
            .iter()
            .map(|&c| t.nodes[c].name.as_str())
            .collect();
        assert_eq!(
            groups,
            [
                "Apps",
                "Music",
                "Diabetes",
                "Morphology",
                "HTGAA",
                "Study",
                "Scratch",
                "Transient",
            ]
        );
        // Scratch / Transient are empty groups (homes for ad-hoc tabs).
        assert!(t.leaves(group(&t, "Scratch")).is_empty());
        assert!(t.leaves(group(&t, "Transient")).is_empty());
    }

    #[test]
    fn leaf_spec_expands_tilde_and_keeps_command() {
        let t = tree();
        let music = group(&t, "Music");
        let leaves = t.leaves(music);
        assert_eq!(leaves.len(), 2);
        let (wd, cmd) = t.leaf_spec(leaves[0]).unwrap();
        assert_eq!(t.nodes[leaves[0]].name, "Hermes chat");
        assert_eq!(wd, Path::new("/home/u/Music")); // ~ expanded
        assert_eq!(cmd, "hermes");
        // Quoted multi-word command survives intact.
        assert_eq!(t.leaf_spec(leaves[1]).unwrap().1, "nano wanted");
    }

    #[test]
    fn group_for_new_resolves_context() {
        let t = tree();
        let music = group(&t, "Music");
        let leaf = t.leaves(music)[0];
        assert_eq!(t.group_for_new(leaf), music); // leaf -> its group
        assert_eq!(t.group_for_new(music), music); // group -> itself
    }

    #[test]
    fn already_running_predicate() {
        let idle = Obs::default();
        let busy = Obs {
            foreground: true,
            cmd: Some("/usr/bin/hermes --serve".into()),
        };
        assert!(!already_running(&idle, "hermes")); // no foreground job
        assert!(!already_running(&busy, "")); // a bare shell is never "running"
        assert!(already_running(&busy, "hermes")); // program matches
        assert!(already_running(&Obs { foreground: true, cmd: Some("bun run main.ts".into()) }, "bun run main.ts"));
        assert!(!already_running(&busy, "./run.sh")); // different program
    }

    #[test]
    fn plan_start_is_idempotent() {
        let t = tree();
        let music = group(&t, "Music");
        let [a, b] = t.leaves(music)[..] else { panic!() };

        // Nothing spawned yet: each leaf gets Spawn then Run.
        let none = |_: NodeId| false;
        let acts = plan_start(&t, &none, &HashMap::new(), music);
        assert_eq!(acts.len(), 4);
        assert!(matches!(acts[0], Action::Spawn(x) if x == a));
        assert!(matches!(acts[1], Action::Run(x) if x == a));

        // Sessions exist; `a` is already running its command -> skip it
        // entirely, only `b` is (re)run.
        let all = |_: NodeId| true;
        let mut obs = HashMap::new();
        obs.insert(a, Obs { foreground: true, cmd: Some("hermes".into()) });
        let acts = plan_start(&t, &all, &obs, music);
        assert_eq!(acts.len(), 1);
        assert!(matches!(acts[0], Action::Run(x) if x == b));
    }

    #[test]
    fn plan_stop_targets_only_live_leaves() {
        let t = tree();
        let music = group(&t, "Music");
        let [a, _b] = t.leaves(music)[..] else { panic!() };
        let only_a = move |n: NodeId| n == a;
        let acts = plan_stop(&t, &only_a, music);
        assert_eq!(acts.len(), 1);
        assert!(matches!(acts[0], Action::Sigint(x) if x == a));
    }

    #[test]
    fn text_field_editing_is_pure_and_utf8_safe() {
        // Insert at caret, advancing it.
        let (s, c) = apply_edit("ac", 1, Edit::Ins('b'));
        assert_eq!((s.as_str(), c), ("abc", 2));
        // Backspace removes the char before the caret.
        let (s, c) = apply_edit("abc", 2, Edit::Back);
        assert_eq!((s.as_str(), c), ("ac", 1));
        // Delete removes the char at the caret; Backspace at 0 is a no-op.
        assert_eq!(apply_edit("abc", 0, Edit::Del).0, "bc");
        assert_eq!(apply_edit("abc", 0, Edit::Back), ("abc".to_string(), 0));
        // Home/End/Left/Right move only the caret.
        assert_eq!(apply_edit("abc", 1, Edit::End).1, 3);
        assert_eq!(apply_edit("abc", 3, Edit::Right).1, 3); // clamped
        // Caret is a *char* index, so multibyte content stays valid.
        let (s, c) = apply_edit("é€", 2, Edit::Ins('x'));
        assert_eq!((s.as_str(), c), ("é€x", 3));
        let (s, _) = apply_edit("é€x", 1, Edit::Back);
        assert_eq!(s, "€x");
    }

    #[test]
    fn field_text_reflects_kind() {
        let t = tree();
        let music = group(&t, "Music");
        let leaf = t.leaves(music)[0];
        assert_eq!(t.field_text(leaf, Field::Title).as_deref(), Some("Hermes chat"));
        assert_eq!(t.field_text(leaf, Field::Command).as_deref(), Some("hermes"));
        // A group has a title but no command (that field renders disabled).
        assert_eq!(t.field_text(music, Field::Title).as_deref(), Some("Music"));
        assert_eq!(t.field_text(music, Field::Command), None);
    }
}
