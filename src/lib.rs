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
use alacritty_terminal::grid::{Dimensions, Scroll};
use alacritty_terminal::index::{Column, Line, Point, Side};
use alacritty_terminal::selection::{Selection, SelectionType};
use alacritty_terminal::sync::FairMutex;
use alacritty_terminal::term::cell::Flags;
use alacritty_terminal::term::{Config, Term, TermMode};
use alacritty_terminal::tty::{self, Options as PtyOptions};
use alacritty_terminal::vte::ansi::{Color, NamedColor};

use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::{Key, ModifiersState, NamedKey};
use winit::window::{CursorIcon, ResizeDirection, Window, WindowId};

pub mod testkit;

const FONT_PX: f32 = 16.0;

/// Empty-terminal hint, with the platform's own "new shell" shortcut.
#[cfg(target_os = "macos")]
const NO_SESSION_HINT: &str =
    "No session here. Right-click a node \u{2192} Start, or \u{2318}T for a shell.";
#[cfg(not(target_os = "macos"))]
const NO_SESSION_HINT: &str =
    "No session here. Right-click a node \u{2192} Start, or Ctrl+Shift+T for a shell.";

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

    /// Set a leaf's working directory (the inspector / use-cwd button). No-op
    /// on a group.
    fn set_workdir(&mut self, id: NodeId, dir: PathBuf) {
        if let Kind::Leaf { workdir, .. } = &mut self.nodes[id].kind {
            *workdir = dir;
        }
    }

    /// Current text of an inspector field for `id`. `Command`/`Directory` are
    /// `None` on a group (it has neither — those fields render disabled).
    fn field_text(&self, id: NodeId, f: Field) -> Option<String> {
        match f {
            Field::Title => Some(self.nodes[id].name.clone()),
            Field::Command => self.leaf_spec(id).map(|(_, c)| c.to_string()),
            Field::Dir => self
                .leaf_spec(id)
                .map(|(w, _)| w.display().to_string()),
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

    /// The sibling immediately before `id` among its parent's children, if any
    /// (i.e. `None` when `id` is the first child). Used to pick what to select
    /// after closing a session.
    fn prev_sibling(&self, id: NodeId) -> Option<NodeId> {
        let p = self.nodes[id].parent?;
        let kids = &self.nodes[p].children;
        let i = kids.iter().position(|&c| c == id)?;
        i.checked_sub(1).map(|j| kids[j])
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

/// Inverse of `expand_tilde`: re-collapse a `$HOME`-prefixed path back to `~`
/// so the saved file keeps the user's tilde style.
fn collapse_tilde(p: &Path, home: &Path) -> String {
    match p.strip_prefix(home) {
        Ok(rest) if rest.as_os_str().is_empty() => "~".into(),
        Ok(rest) => format!("~/{}", rest.display()),
        Err(_) => p.display().to_string(),
    }
}

/// Parse the `workspace` file into a tree.
///
/// Format: indentation depth = tree depth (one `\t` per level). Fields on a
/// line are tab-separated; surrounding whitespace and `"`quotes`"` are
/// stripped. A line with >1 tab-separated field is a `Leaf` (`name`,
/// `workdir`, optional `command`); a single bare token is a `Group`. The
/// first non-empty line (`workspaces`) is the root sentinel. `Scratch` and
/// `Transient` are ensured present (appended only if the file lacks them) as
/// homes for ad-hoc sessions. Inverse of [`serialize_workspace`].
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
        if depth == 0 {
            // The `workspaces` root line; already created.
            continue;
        }
        let rest = raw.trim_start_matches('\t');
        // Split on tabs *before* trimming so an empty trailing command field
        // (`"name"\tdir\t`) still marks the line as a leaf, not a group.
        let parts: Vec<&str> = rest.split('\t').collect();
        let clean = |s: &str| s.trim().trim_matches('"').trim().to_string();
        let parent = *stack.get(depth - 1).unwrap_or(&root);
        let id = if parts.len() == 1 {
            let name = clean(parts[0]);
            if name.is_empty() {
                continue;
            }
            tree.push(Some(parent), name, Kind::Group, false)
        } else {
            let workdir = expand_tilde(&clean(parts[1]), home);
            let command = parts.get(2).map(|s| clean(s)).unwrap_or_default();
            tree.push(
                Some(parent),
                clean(parts[0]),
                Kind::Leaf { workdir, command },
                false,
            )
        };
        stack.truncate(depth);
        stack.push(id);
    }

    // Ensure the ad-hoc homes exist, but don't duplicate them when a saved
    // file already carries them (with their persisted scratch sessions).
    for name in ["Scratch", "Transient"] {
        let present = tree.nodes[root]
            .children
            .iter()
            .any(|&c| tree.nodes[c].name == name && matches!(tree.nodes[c].kind, Kind::Group));
        if !present {
            tree.push(Some(root), name.into(), Kind::Group, false);
        }
    }
    tree
}

/// Serialize the tree back to the `workspace` file format. Inverse of
/// [`parse_workspace`] (round-trips structure, names, workdirs, commands).
/// Pure; `home` re-collapses absolute workdirs to `~`.
fn serialize_workspace(tree: &Tree, home: &Path) -> String {
    fn go(t: &Tree, id: NodeId, depth: usize, home: &Path, out: &mut String) {
        for &c in &t.nodes[id].children {
            let n = &t.nodes[c];
            let indent = "\t".repeat(depth);
            match &n.kind {
                Kind::Group => {
                    out.push_str(&format!("{indent}{}\n", n.name));
                    go(t, c, depth + 1, home, out);
                }
                Kind::Leaf { workdir, command } => {
                    let wd = collapse_tilde(workdir, home);
                    if command.is_empty() {
                        out.push_str(&format!("{indent}\"{}\"\t{}\n", n.name, wd));
                    } else {
                        // Quote a command only when it contains whitespace,
                        // mirroring the hand-written file's style.
                        let cmd = if command.split_whitespace().nth(1).is_some() {
                            format!("\"{command}\"")
                        } else {
                            command.clone()
                        };
                        out.push_str(&format!("{indent}\"{}\"\t{}\t{}\n", n.name, wd, cmd));
                    }
                }
                Kind::Root => {}
            }
        }
    }
    let mut out = String::from("workspaces\n");
    go(tree, tree.root, 1, home, &mut out);
    out
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
    Dir,
}

impl Field {
    /// Inspector field order (also its row index in `field_rects`).
    const ALL: [Field; 3] = [Field::Title, Field::Command, Field::Dir];
    fn index(self) -> usize {
        Field::ALL.iter().position(|&x| x == self).unwrap()
    }
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

/// A window-level command triggered by a keyboard shortcut, independent of the
/// physical keys that produce it. The keymap is OS-specific (see
/// [`match_shortcut`]); everything downstream of it is not — this is the
/// abstraction that lets the same actions wear ⌘ on macOS and Ctrl+Shift
/// elsewhere.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Shortcut {
    Copy,
    Paste,
    NewTab,
    CloseTab,
    SelectPrev,
    SelectNext,
    Collapse,
    Expand,
    Start,
    Stop,
}

/// Map a modified keystroke to a [`Shortcut`], or `None` to pass it through to
/// the terminal. Bindings are the intuitive ones for each platform:
///
/// * **macOS** uses ⌘ (Command), like every native mac app: ⌘C copy, ⌘V paste,
///   ⌘T new tab, ⌘W close, ⌘↑/⌘↓ select previous/next tree node, ⌘←/⌘→
///   collapse/expand the selected group, ⌘R run/start, ⌘. stop. ⌘ never
///   collides with the shell's own Ctrl- control codes, so all of these are
///   safe to claim.
/// * **Elsewhere** there is no ⌘, so the terminal convention Ctrl+Shift is used
///   (plain Ctrl+C etc. must stay free for the shell): Ctrl+Shift+C/V/T/W for
///   copy/paste/new/close, Ctrl+Shift+↑/↓ select previous/next node,
///   Ctrl+Shift+←/→ collapse/expand the selected group, Ctrl+Shift+S start,
///   Ctrl+Shift+X stop.
fn match_shortcut(mods: ModifiersState, key: &Key) -> Option<Shortcut> {
    let ch = |name: &str| matches!(key, Key::Character(c) if c.eq_ignore_ascii_case(name));
    #[cfg(target_os = "macos")]
    {
        if !mods.super_key() {
            return None;
        }
        if ch(".") {
            return Some(Shortcut::Stop); // ⌘. — the mac "cancel" gesture
        }
        Some(match key {
            _ if ch("c") => Shortcut::Copy,
            _ if ch("v") => Shortcut::Paste,
            _ if ch("t") => Shortcut::NewTab,
            _ if ch("w") => Shortcut::CloseTab,
            Key::Named(NamedKey::ArrowUp) => Shortcut::SelectPrev,
            Key::Named(NamedKey::ArrowDown) => Shortcut::SelectNext,
            Key::Named(NamedKey::ArrowLeft) => Shortcut::Collapse,
            Key::Named(NamedKey::ArrowRight) => Shortcut::Expand,
            _ if ch("r") => Shortcut::Start,
            _ => return None,
        })
    }
    #[cfg(not(target_os = "macos"))]
    {
        if !(mods.control_key() && mods.shift_key()) {
            return None;
        }
        Some(match key {
            _ if ch("c") => Shortcut::Copy,
            _ if ch("v") => Shortcut::Paste,
            _ if ch("t") => Shortcut::NewTab,
            _ if ch("w") => Shortcut::CloseTab,
            Key::Named(NamedKey::ArrowUp) => Shortcut::SelectPrev,
            Key::Named(NamedKey::ArrowDown) => Shortcut::SelectNext,
            Key::Named(NamedKey::ArrowLeft) => Shortcut::Collapse,
            Key::Named(NamedKey::ArrowRight) => Shortcut::Expand,
            _ if ch("s") => Shortcut::Start,
            _ if ch("x") => Shortcut::Stop,
            _ => return None,
        })
    }
}

/// Observe what a PTY's shell is doing (is a command in the foreground, and
/// what). Best-effort and OS-abstracted (see [`sys`]); any failure degrades to
/// "nothing observed" (`Obs::default`), which the planner treats as "not
/// running". `pid == 0` is the headless test harness's sentinel — never a real
/// shell — so it short-circuits to idle.
fn observe(shell_pid: u32) -> Obs {
    if shell_pid == 0 {
        return Obs::default();
    }
    sys::observe(shell_pid)
}

/// The shell's *current* working directory (where `cd` at the prompt left it).
/// OS-abstracted (see [`sys`]), best-effort.
fn proc_cwd(shell_pid: u32) -> Option<PathBuf> {
    if shell_pid == 0 {
        return None;
    }
    sys::proc_cwd(shell_pid)
}

/// Per-OS process introspection. Each variant exposes the same two pure-ish
/// functions; the rest of the program never sees the platform. `observe` is
/// called every frame (per visible session), so it must be cheap — Linux reads
/// `/proc`, macOS uses the `libproc` syscalls directly (no subprocess).
mod sys {
    use super::Obs;
    use std::path::PathBuf;

    #[cfg(target_os = "linux")]
    pub fn observe(shell_pid: u32) -> Obs {
        fn children(pid: u32) -> Vec<u32> {
            std::fs::read_to_string(format!("/proc/{pid}/task/{pid}/children"))
                .ok()
                .map(|s| s.split_whitespace().filter_map(|x| x.parse().ok()).collect())
                .unwrap_or_default()
        }
        let mut cur = shell_pid;
        while let Some(&c) = children(cur).first() {
            cur = c;
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

    #[cfg(target_os = "linux")]
    pub fn proc_cwd(shell_pid: u32) -> Option<PathBuf> {
        std::fs::read_link(format!("/proc/{shell_pid}/cwd")).ok()
    }

    #[cfg(target_os = "macos")]
    mod libproc {
        use std::os::raw::{c_int, c_void};

        // macOS has no `/proc`; these libproc(3) syscalls are the supported way
        // to walk the process tree and read a process's executable path.
        unsafe extern "C" {
            pub fn proc_listchildpids(ppid: c_int, buffer: *mut c_void, buffersize: c_int) -> c_int;
            pub fn proc_pidpath(pid: c_int, buffer: *mut c_void, buffersize: u32) -> c_int;
        }

        /// Direct children of `pid`. The buffer is zero-initialized and every
        /// non-zero slot read back, which sidesteps the documented ambiguity in
        /// `proc_listchildpids`'s return value (bytes vs. count).
        pub fn children(pid: u32) -> Vec<u32> {
            let mut buf = vec![0i32; 1024];
            let n = unsafe {
                proc_listchildpids(
                    pid as c_int,
                    buf.as_mut_ptr() as *mut c_void,
                    (buf.len() * std::mem::size_of::<i32>()) as c_int,
                )
            };
            if n <= 0 {
                return Vec::new();
            }
            buf.into_iter().filter(|&p| p > 0).map(|p| p as u32).collect()
        }

        /// Absolute path of `pid`'s executable, if readable.
        pub fn path(pid: u32) -> Option<String> {
            const MAX: usize = 4096; // PROC_PIDPATHINFO_MAXSIZE
            let mut buf = vec![0u8; MAX];
            let n = unsafe {
                proc_pidpath(pid as c_int, buf.as_mut_ptr() as *mut c_void, MAX as u32)
            };
            if n <= 0 {
                return None;
            }
            buf.truncate(n as usize);
            String::from_utf8(buf).ok().filter(|s| !s.is_empty())
        }
    }

    #[cfg(target_os = "macos")]
    pub fn observe(shell_pid: u32) -> Obs {
        // Descend to the deepest foreground descendant of the shell.
        let mut cur = shell_pid;
        while let Some(&c) = libproc::children(cur).first() {
            cur = c;
        }
        if cur == shell_pid {
            return Obs::default();
        }
        Obs {
            foreground: true,
            cmd: libproc::path(cur),
        }
    }

    #[cfg(target_os = "macos")]
    pub fn proc_cwd(shell_pid: u32) -> Option<PathBuf> {
        // No `/proc`; `lsof` reports the cwd file descriptor. This only runs on
        // an explicit "use current working dir" click, so the subprocess cost
        // is irrelevant.
        let out = std::process::Command::new("lsof")
            .args(["-a", "-d", "cwd", "-Fn", "-p", &shell_pid.to_string()])
            .output()
            .ok()?;
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .find_map(|l| l.strip_prefix('n').map(PathBuf::from))
    }

    // Any other OS: introspection isn't wired up, so report "idle" / unknown.
    // The app stays fully usable; only the running-marker and use-cwd dim out.
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    pub fn observe(_shell_pid: u32) -> Obs {
        Obs::default()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    pub fn proc_cwd(_shell_pid: u32) -> Option<PathBuf> {
        None
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
    /// Glyph-coverage fonts (emoji + OS CJK/symbol faces) finished loading on a
    /// worker thread — swap them into the renderer so later frames cover more.
    FallbackFonts(Vec<fontdue::Font>),
}

/// `EventListener` impl handed to `alacritty_terminal`. Forwards the events we
/// care about onto the winit loop. `Clone + Send` (the PTY runs on its own
/// thread); `id` ties events back to the owning session.
#[derive(Clone)]
enum Listener {
    /// Live session: forward PTY events onto the winit loop.
    Winit {
        proxy: EventLoopProxy<UserEvent>,
        id: u64,
    },
    /// Headless (testkit): a `Term` with no event loop behind it. Events are
    /// dropped — the harness drives the parser synchronously and renders on
    /// demand, so nothing needs waking.
    Null,
}

impl EventListener for Listener {
    fn send_event(&self, event: TermEvent) {
        let Listener::Winit { proxy, id } = self else {
            return;
        };
        let _ = match event {
            TermEvent::Wakeup => proxy.send_event(UserEvent::Wakeup),
            TermEvent::Exit | TermEvent::ChildExit(_) => {
                proxy.send_event(UserEvent::Exit(*id))
            }
            TermEvent::Title(t) => proxy.send_event(UserEvent::Title(*id, t)),
            TermEvent::ResetTitle => proxy.send_event(UserEvent::ResetTitle(*id)),
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

/// Which embedded face to draw a cell with, derived from its VT attributes.
/// The discriminant doubles as the index into `Renderer::styles`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FontStyle {
    Regular = 0,
    Bold = 1,
    Italic = 2,
    BoldItalic = 3,
}

/// Map a cell's `Flags` to the face it should be drawn with — "draw bold text
/// in bold font" and "allow italic text", in iTerm's terms.
fn font_style(flags: Flags) -> FontStyle {
    match (flags.contains(Flags::BOLD), flags.contains(Flags::ITALIC)) {
        (true, true) => FontStyle::BoldItalic,
        (true, false) => FontStyle::Bold,
        (false, true) => FontStyle::Italic,
        (false, false) => FontStyle::Regular,
    }
}

struct Renderer {
    /// The four embedded JetBrains Mono faces, indexed by `FontStyle as usize`.
    /// `styles[0]` (Regular) defines the cell metrics and the shared baseline.
    styles: [fontdue::Font; 4],
    /// OS fallback faces (regular only), consulted in order for glyphs the
    /// primary family lacks (emoji, CJK, rare symbols).
    fallbacks: Vec<fontdue::Font>,
    cell_w: usize,
    cell_h: usize,
    /// Rasterized-glyph cache keyed by `(char, style, pixel-size-bits)`. Keying
    /// on the size lets the same renderer serve both the logical 1× pass and
    /// the crisp Retina pass (glyphs rasterized at `FONT_PX × scale`).
    cache: HashMap<(char, u8, u32), Glyph>,
    /// When `Some`, `draw_text` records (rather than rasterizes) each chrome
    /// string in *logical* coordinates instead of drawing it. The GUI uses this
    /// to keep text out of the logical frame so it can be replayed crisply at
    /// device resolution after the shape layer is upscaled — the same Retina
    /// fix the terminal grid already gets. `None` = draw immediately (headless).
    text_log: Option<Vec<TextCmd>>,
}

/// A deferred chrome-text draw, captured in logical coordinates.
struct TextCmd {
    x: usize,
    y: usize,
    max_w: usize,
    text: String,
    color: u32,
}

impl Renderer {
    fn new() -> Self {
        let load = |b: &[u8]| {
            fontdue::Font::from_bytes(b.to_vec(), fontdue::FontSettings::default())
                .expect("embedded font failed to parse")
        };
        let styles = [
            load(UBUNTU_MONO_REGULAR),
            load(UBUNTU_MONO_BOLD),
            load(UBUNTU_MONO_ITALIC),
            load(UBUNTU_MONO_BOLD_ITALIC),
        ];
        // The fallback chain (embedded emoji + OS faces for CJK/rare symbols) is
        // *not* loaded here: parsing those fonts — and the `fc-match` subprocesses
        // that locate the OS ones — cost ~100ms+ and would block the window from
        // appearing at all. They're only ever consulted lazily by `glyph()` for a
        // char the primary family lacks, so `load_fallback_fonts()` runs on a
        // worker thread and the result is swapped in via `UserEvent::FallbackFonts`.
        let fallbacks: Vec<fontdue::Font> = Vec::new();

        let lm = styles[0]
            .horizontal_line_metrics(FONT_PX)
            .expect("font line metrics");
        let cell_h = lm.new_line_size.ceil() as usize;
        // Monospace: every cell is the advance width of a representative glyph.
        let cell_w = styles[0].metrics('M', FONT_PX).advance_width.ceil() as usize;

        Self {
            styles,
            fallbacks,
            cell_w: cell_w.max(1),
            cell_h: cell_h.max(1),
            cache: HashMap::new(),
            text_log: None,
        }
    }

    /// Rasterize (or fetch from cache) glyph `c` in `style` at `px` pixels.
    /// Falls back styled-face → regular primary → OS fallbacks for coverage,
    /// but always positions on the primary's baseline so faces stay aligned.
    fn glyph(&mut self, c: char, style: FontStyle, px: f32) -> &Glyph {
        let key = (c, style as u8, px.to_bits());
        let styles = &self.styles;
        let fallbacks = &self.fallbacks;
        self.cache.entry(key).or_insert_with(|| {
            let styled = &styles[style as usize];
            let font = if styled.lookup_glyph_index(c) != 0 {
                styled
            } else if styles[0].lookup_glyph_index(c) != 0 {
                &styles[0]
            } else {
                fallbacks
                    .iter()
                    .find(|f| f.lookup_glyph_index(c) != 0)
                    .unwrap_or(&styles[0])
            };
            // One baseline (the primary's ascent) for every face/fallback.
            let ascent = styles[0]
                .horizontal_line_metrics(px)
                .map(|m| m.ascent)
                .unwrap_or(px * 0.8);
            let (m, mut bitmap) = font.rasterize(c, px);
            // "Thin strokes on Retina Displays": at hi-dpi pixel sizes, lighten
            // partial coverage so stems read thinner and cleaner, the way macOS
            // renders text natively (a no-op at the logical 1× size).
            if px >= FONT_PX * 1.5 {
                for v in bitmap.iter_mut() {
                    *v = ((*v as f32 / 255.0).powf(1.45) * 255.0).round() as u8;
                }
            }
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

// The primary family is **embedded in the binary** — Ubuntu Mono derivative
// Powerline (the font from the iTerm setup), all four faces. Shipping the fonts
// instead of discovering them means the app renders identically on every OS,
// needs no fontconfig (the previous Linux-only dependency that made it panic on
// a clean macOS), and keeps screenshots reproducible. Bundling
// Bold/Italic/BoldItalic is what lets cells render in the correct face per their
// VT attributes; the "Powerline" patch carries the segment/separator glyphs.
const UBUNTU_MONO_REGULAR: &[u8] = include_bytes!("../assets/fonts/UbuntuMono-Regular.ttf");
const UBUNTU_MONO_BOLD: &[u8] = include_bytes!("../assets/fonts/UbuntuMono-Bold.ttf");
const UBUNTU_MONO_ITALIC: &[u8] = include_bytes!("../assets/fonts/UbuntuMono-Italic.ttf");
const UBUNTU_MONO_BOLD_ITALIC: &[u8] = include_bytes!("../assets/fonts/UbuntuMono-BoldItalic.ttf");

/// Embedded monochrome emoji face (Noto Emoji, SIL OFL). fontdue rasterizes
/// outline glyphs only, so colour-bitmap emoji (Apple Color Emoji) can't be
/// drawn — this gives crisp single-colour emoji that work on every platform.
const NOTO_EMOJI: &[u8] = include_bytes!("../assets/fonts/NotoEmoji-Regular.ttf");

/// OS-specific fallback font paths for glyph coverage beyond the primary face.
/// Pure list of candidate paths; non-existent ones are skipped by `load_fonts`.
/// Build the lazy fallback chain: embedded Noto Emoji (monochrome) first so
/// emoji render identically on every OS, then OS faces (`fallback_font_paths`)
/// for anything the primary family lacks (CJK, rare symbols). Runs off the
/// critical path on a worker thread — see `Renderer::new`.
fn load_fallback_fonts() -> Vec<fontdue::Font> {
    let mut fallbacks: Vec<fontdue::Font> = Vec::new();
    if let Ok(f) = fontdue::Font::from_bytes(NOTO_EMOJI.to_vec(), fontdue::FontSettings::default()) {
        fallbacks.push(f);
    }
    fallbacks.extend(
        fallback_font_paths()
            .iter()
            .filter_map(|p| std::fs::read(p).ok())
            .filter_map(|b| fontdue::Font::from_bytes(b, fontdue::FontSettings::default()).ok()),
    );
    fallbacks
}

fn fallback_font_paths() -> Vec<String> {
    #[cfg(target_os = "macos")]
    {
        // Symbol/CJK/emoji faces shipped with every macOS. Apple Color Emoji
        // is a bitmap (sbix) face; fontdue reads its outline layer, which is
        // enough to avoid blank cells for the common pictographs.
        [
            "/System/Library/Fonts/Apple Symbols.ttf",
            "/System/Library/Fonts/Symbol.ttf",
            "/System/Library/Fonts/Apple Color Emoji.ttc",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }
    #[cfg(not(target_os = "macos"))]
    {
        // Elsewhere (Linux/BSD), ask fontconfig for symbol/emoji coverage if
        // it's present; absence is fine, the embedded primary still renders.
        let patterns = [
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
        paths
    }
}

// Dark theme: white text over the teal/green backdrop image, black chrome.
const FG: u32 = 0xea_ea_ea; // terminal default text (white)
// `BG` doubles as the "default background" sentinel — cells with this bg are
// left unfilled so the backdrop image shows through. Its value only shows if a
// cell is *explicitly* filled with the default bg, so it's a dark teal that
// blends with the image.
const BG: u32 = 0x0b_1d_1d;
const SEL: u32 = 0x2f_5d_6e; // selection fill (white text stays readable)

// --- Win2k chrome ----------------------------------------------------------
// Windows 2000 design *principles* (not its colours): explicit borders and
// bevels, a subtle vertical gradient, compact fixed heights, dense layout,
// left-aligned titles.
const SIDEBAR_W: usize = 212; // fixed-width tree pane
const HEADER_H: usize = 24; // title bar over the terminal & sidebar head
const ROW_H: usize = 20; // one tree row
const INDENT: usize = 14; // px added per tree depth
const EXPANDER_W: usize = 14; // width of the expander / leaf-marker column
const TPAD: usize = 7; // left gutter inside the tree, so rows clear the pane edge
const MARK_GAP: usize = 5; // gap between the marker column and the row label
const CTX_W: usize = 124; // context-menu width
const RPANEL_W: usize = 252; // right inspector pane width
const WBTN_W: usize = 30; // min/max/close button width
const INFO_H: usize = 24; // sidebar "info" toggle band, below WORKSPACE
const EDGE: f64 = 5.0; // borderless-window resize-grip thickness

// Black chrome with white text. The Win2k bevel structure is kept but recolored
// to dark grays so panels read as raised/inset without any light surfaces.
const STRIP_BG: u32 = 0x0a_0a_0a; // chrome background (near-black)
const PANEL_HI: u32 = 0x33_33_33; // top of a raised gradient (selected row/button)
const PANEL_LO: u32 = 0x1f_1f_1f; // bottom of a raised gradient
const HEAD_HI: u32 = 0x16_16_16; // header gradient top
const HEAD_LO: u32 = 0x08_08_08; // header gradient bottom
const BEVEL_LT: u32 = 0x3a_3a_3a; // raised highlight (top/left)
const BEVEL_DK: u32 = 0x00_00_00; // raised shadow (bottom/right)
const INK: u32 = 0xf0_f0_f0; // primary chrome text (white)
const INK_DIM: u32 = 0xa0_a0_a0; // secondary chrome text (gray)
const RUN_INK: u32 = 0x3f_d0_4f; // "running" marker (green square)

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

// ANSI palette tuned for the **dark** backdrop: bright, saturated hues that pop
// against the teal/green image, with white text as the default foreground.
fn named_rgb(n: NamedColor) -> Option<u32> {
    Some(match n {
        NamedColor::Black => 0x2b2b2b,
        NamedColor::Red => 0xff6d67,
        NamedColor::Green => 0x8ae234,
        NamedColor::Yellow => 0xfce94f,
        NamedColor::Blue => 0x78b6ff,
        NamedColor::Magenta => 0xe48bff,
        NamedColor::Cyan => 0x54e1d6,
        NamedColor::White => 0xeaeaea,
        NamedColor::BrightBlack => 0x6a6a6a,
        NamedColor::BrightRed => 0xff8b86,
        NamedColor::BrightGreen => 0xadf85f,
        NamedColor::BrightYellow => 0xfff27a,
        NamedColor::BrightBlue => 0x9ccbff,
        NamedColor::BrightMagenta => 0xf0a8ff,
        NamedColor::BrightCyan => 0x7defe4,
        NamedColor::BrightWhite => 0xffffff,
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

// --- backdrop image --------------------------------------------------------
// The terminal area is drawn over the user's profile background image (the same
// teal/green gradient as their iTerm). It's embedded so it travels with the
// binary, decoded once, and pre-darkened so white text and the bright ANSI
// palette stay readable over it (this is iTerm's "background blend").

/// Embedded background image bytes (PNG).
const BACKDROP_PNG: &[u8] = include_bytes!("../assets/profile-default-bg.png");

/// How far to blend the image toward black, 0..=255 (readability dimming).
const BACKDROP_DIM: u32 = 96;

/// A decoded, pre-darkened RGB image: `px[y * w + x]` packed `0x00RRGGBB`.
struct Backdrop {
    w: usize,
    h: usize,
    px: Vec<u32>,
}

/// Decode the embedded image once (lazily) and cache it for the process. Decode
/// failure degrades to a 1×1 dark tile, so the app still runs (just a flat dark
/// terminal background) rather than panicking.
fn backdrop() -> &'static Backdrop {
    static BACKDROP: std::sync::OnceLock<Backdrop> = std::sync::OnceLock::new();
    BACKDROP.get_or_init(|| {
        match image::load_from_memory(BACKDROP_PNG) {
            Ok(img) => {
                let rgb = img.to_rgb8();
                let (w, h) = (rgb.width() as usize, rgb.height() as usize);
                let px = rgb
                    .pixels()
                    .map(|p| darken(pack(p[0], p[1], p[2]), BACKDROP_DIM))
                    .collect();
                Backdrop { w, h, px }
            }
            Err(_) => Backdrop {
                w: 1,
                h: 1,
                px: vec![darken(BG, BACKDROP_DIM)],
            },
        }
    })
}

/// Fast integer blend of `color` toward black by `k`/255 (no gamma — this is a
/// bulk fill, not glyph anti-aliasing). `k = 0` keeps the color, `255` = black.
fn darken(color: u32, k: u32) -> u32 {
    let f = 255 - k;
    let r = (((color >> 16) & 0xff) * f) / 255;
    let g = (((color >> 8) & 0xff) * f) / 255;
    let b = ((color & 0xff) * f) / 255;
    r << 16 | g << 8 | b
}

/// Paint the backdrop image, stretched to cover the rect `(x, y, w, h)`, into
/// `buf`. Used wherever the terminal background is laid down (the logical
/// compose and the crisp Retina overdraw) so the image shows under every cell
/// that doesn't set its own background. Stretching a smooth gradient is
/// visually lossless, so no aspect-correct cropping is needed.
fn fill_backdrop(buf: &mut [u32], bw: usize, bh: usize, x: usize, y: usize, w: usize, h: usize) {
    let img = backdrop();
    if w == 0 || h == 0 || img.w == 0 || img.h == 0 {
        return;
    }
    for ry in 0..h {
        let py = y + ry;
        if py >= bh {
            break;
        }
        let sy = (ry * img.h / h).min(img.h - 1);
        let srow = sy * img.w;
        let drow = py * bw;
        for rx in 0..w {
            let px = x + rx;
            if px >= bw {
                break;
            }
            let sx = (rx * img.w / w).min(img.w - 1);
            buf[drow + px] = img.px[srow + sx];
        }
    }
}

/// One terminal session: its own PTY thread + VT state machine + grid size.
/// Reused as-is; `spawn_session` now also returns the shell pid for `/proc`.
struct Tab {
    term: Arc<FairMutex<Term<Listener>>>,
    /// PTY input channel. `None` for a headless harness session (no real
    /// PTY); every write goes through `App::send_to`, which then no-ops.
    pty_tx: Option<EventLoopSender>,
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
    let listener = Listener::Winit {
        proxy: proxy.clone(),
        id,
    };
    let term = Term::new(Config::default(), &size, listener.clone());
    let term = Arc::new(FairMutex::new(term));

    // Don't rely on inheriting TERM: launched from the .desktop entry there
    // is no controlling terminal, so TERM is unset and the shell's rc files
    // fall back to a colourless prompt (and terminfo programs go monochrome).
    // Pin a widely-present 256-colour terminfo so colours work either way.
    let env = || {
        HashMap::from([
            ("TERM".to_string(), "xterm-256color".to_string()),
            ("COLORTERM".to_string(), "truecolor".to_string()),
        ])
    };
    let opts = PtyOptions {
        working_directory: workdir.filter(|p| p.is_dir()),
        env: env(),
        ..PtyOptions::default()
    };
    let pty = tty::new(&opts, window_size, 0)
        .or_else(|_| {
            tty::new(
                &PtyOptions {
                    env: env(),
                    ..PtyOptions::default()
                },
                window_size,
                0,
            )
        })
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
            pty_tx: Some(pty_tx),
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

/// Everything that exists once the window is created — or, headlessly, once
/// the test harness builds it without a window at all.
struct State {
    /// `None` when driven headlessly by the test harness (no winit window).
    window: Option<Rc<Window>>,
    /// Off-screen framebuffer composed at *logical* size (`logical_size()`),
    /// packed `0x00RRGGBB`. The whole UI is drawn here at 1× and then
    /// nearest-neighbour upscaled onto the physical surface, so the layout is
    /// pixel-identical at every display density (this is the macOS/Retina
    /// fix — previously everything was laid out in raw physical pixels and
    /// came out half-size on a 2× display) and screenshots are
    /// DPI-independent.
    fb: Vec<u32>,
    /// Physical pixel size of the window (or the harness's virtual target).
    phys: (usize, usize),
    /// Device pixel ratio (winit `scale_factor`). 1.0 headless / non-HiDPI.
    scale: f64,
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
    /// Sub-line wheel remainder, so a high-resolution touchpad's many small
    /// deltas accumulate into whole-line scrolls instead of being dropped.
    scroll_acc: f64,
    /// Cursor icon currently set on the window, so `set_cursor` is only called
    /// when the shape actually changes (e.g. entering/leaving a resize grip)
    /// rather than on every mouse-move event.
    cursor: CursorIcon,
}

impl State {
    /// Logical (device-independent) size the UI is laid out and drawn at:
    /// physical pixels divided by the display scale. All chrome geometry and
    /// hit-testing work in these coordinates.
    fn logical_size(&self) -> (usize, usize) {
        let s = self.scale.max(1.0);
        (
            ((self.phys.0 as f64 / s).round() as usize).max(1),
            ((self.phys.1 as f64 / s).round() as usize).max(1),
        )
    }

    /// Pull current physical size + scale off the window (no-op headless).
    fn sync_metrics(&mut self) {
        if let Some(w) = &self.window {
            let s = w.inner_size();
            self.phys = (s.width as usize, s.height as usize);
            self.scale = w.scale_factor().max(1.0);
        }
    }

    /// Ask the windowing system to redraw; a no-op when headless (the harness
    /// renders explicitly via `paint`).
    fn request_redraw(&self) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    /// Compose the whole UI into `self.fb` at logical (device-independent)
    /// size. No window or surface involved — the GUI path upscales `fb` onto
    /// the real surface afterwards; the test harness reads `fb` straight off.
    fn paint(&mut self) {
        let (pw, ph) = self.logical_size();
        // `shown()` borrows all of `State`; resolve it before we take `fb`.
        let shown = self.shown();
        // Reuse the framebuffer's allocation across frames; taking it out
        // detaches it from `self` so the draw code can freely borrow other
        // fields (renderer glyph cache, sessions, tree) at the same time.
        let mut buf = std::mem::take(&mut self.fb);
        buf.clear();
        buf.resize(pw * ph, BG);

        let cw = self.renderer.cell_w;
        let ch = self.renderer.cell_h;
        // Right edge of the terminal: the inspector pane (if open) eats into it.
        let tr = term_right(pw, self.inspector);

        // --- terminal area --------------------------------------------------
        // Lay the backdrop image behind the grid; cells without their own
        // background let it show through (the dark theme).
        fill_backdrop(
            &mut buf,
            pw,
            ph,
            SIDEBAR_W,
            HEADER_H,
            tr.saturating_sub(SIDEBAR_W),
            ph.saturating_sub(HEADER_H),
        );
        if let Some(node) = shown {
            // Render the grid at logical (1×) size into `buf`. The GUI redraw
            // re-renders it crisply at device resolution on Retina; the test
            // harness reads this buffer directly.
            let lines = self.sessions[&node].tab.size.lines as i32;
            let term = self.sessions[&node].tab.term.lock();
            draw_terminal_cells(
                &mut buf,
                pw,
                ph,
                &mut self.renderer,
                &term,
                lines,
                SIDEBAR_W,
                HEADER_H,
                tr,
                cw,
                ch,
                FONT_PX,
            );
            drop(term);
        } else {
            draw_text(
                &mut buf,
                pw,
                ph,
                &mut self.renderer,
                SIDEBAR_W + 10,
                HEADER_H + 10,
                tr.saturating_sub(SIDEBAR_W + 20),
                NO_SESSION_HINT,
                INK_DIM,
            );
        }

        // --- header bar over the terminal ----------------------------------
        // (No title text — the path/status label is intentionally hidden.)
        vgradient(&mut buf, pw, ph, SIDEBAR_W, 0, pw - SIDEBAR_W, HEADER_H, HEAD_HI, HEAD_LO);
        fill_rect(&mut buf, pw, ph, SIDEBAR_W, HEADER_H - 1, pw - SIDEBAR_W, 1, BEVEL_DK);
        fill_rect(&mut buf, pw, ph, SIDEBAR_W, 0, pw - SIDEBAR_W, 1, BEVEL_LT);
        let maxed = self.window.as_ref().is_some_and(|w| w.is_maximized());
        let [bmin, bmax, bclose] = win_btns(pw);

        // Window controls, right-aligned in the header row.
        for (rect, label) in [
            (bmin, "\u{2013}"),                                // –  minimize
            (bmax, if maxed { "x" } else { "\u{25a1}" }), // □ / x (restore — placeholder)
            (bclose, "\u{2715}"),                              // ✕  close
        ] {
            draw_button(&mut buf, pw, ph, &mut self.renderer, rect, label, false, true);
        }

        // --- sidebar tree ---------------------------------------------------
        let rows = self.tree.rows();
        let sel = self.selected;
        draw_sidebar(
            &mut buf,
            pw,
            ph,
            &mut self.renderer,
            &rows,
            sel,
            &self.sessions,
            self.inspector,
        );

        // --- right inspector ------------------------------------------------
        if self.inspector {
            let can_use_cwd = self.tree.is_leaf(sel) && self.sessions.contains_key(&sel);
            draw_inspector(
                &mut buf,
                pw,
                ph,
                &mut self.renderer,
                &self.tree,
                sel,
                self.focus,
                self.caret,
                can_use_cwd,
            );
        }

        // --- context menu ---------------------------------------------------
        if let Some(m) = &self.ctx {
            draw_ctx_menu(&mut buf, pw, ph, &mut self.renderer, m.x, m.y);
        }

        self.fb = buf;
    }

    /// Re-render just the terminal grid at full device resolution, directly onto
    /// the physical surface `buf`, *after* the logical frame has been upscaled
    /// into it. This is the Retina text fix: the chrome (pixel-art bevels) is
    /// fine nearest-neighbour-upscaled, but text doubled that way is chunky, so
    /// here the glyphs are rasterized at `FONT_PX × scale` and drawn crisply
    /// over the upscaled copy. A no-op at 1× (the logical frame is already
    /// native) and headless (no surface, so this is never called).
    fn overdraw_terminal_physical(&mut self, buf: &mut [u32], pw: usize, ph: usize) {
        let (lw, lh) = self.logical_size();
        if (lw, lh) == (pw, ph) {
            return; // 1×: `paint` already produced native-resolution text
        }
        let Some(node) = self.shown() else { return };
        let sx = pw as f64 / lw as f64;
        let sy = ph as f64 / lh as f64;
        let origin_x = (SIDEBAR_W as f64 * sx).round() as usize;
        let origin_y = (HEADER_H as f64 * sy).round() as usize;
        let clip_right = (term_right(lw, self.inspector) as f64 * sx).round() as usize;
        let cell_w = ((self.renderer.cell_w as f64 * sx).round() as usize).max(1);
        let cell_h = ((self.renderer.cell_h as f64 * sy).round() as usize).max(1);
        let font_px = FONT_PX * sy as f32;
        let lines = self.sessions[&node].tab.size.lines as i32;
        // Re-lay the backdrop image in the terminal viewport so the crisp pass
        // leaves no antialiasing ghosts behind and the image stays under the
        // text; per-cell backgrounds are repainted by `draw_terminal_cells`.
        fill_backdrop(
            buf,
            pw,
            ph,
            origin_x,
            origin_y,
            clip_right.saturating_sub(origin_x),
            ph.saturating_sub(origin_y),
        );
        let term = self.sessions[&node].tab.term.lock();
        draw_terminal_cells(
            buf,
            pw,
            ph,
            &mut self.renderer,
            &term,
            lines,
            origin_x,
            origin_y,
            clip_right,
            cell_w,
            cell_h,
            font_px,
        );
    }

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
        let top = HEADER_H + INFO_H + 1; // header band + info-toggle band + divider
        if x >= SIDEBAR_W as f64 || y < top as f64 {
            return None;
        }
        let i = (y as usize - top) / ROW_H;
        let row = rows.get(i)?;
        let mark_x = TPAD + row.depth * INDENT;
        let on_expander =
            row.is_group && (x as usize) >= mark_x && (x as usize) < mark_x + EXPANDER_W;
        Some((row.id, on_expander))
    }
}

struct App {
    proxy: EventLoopProxy<UserEvent>,
    state: Option<State>,
    /// The window's pixel surface. Lives here (not on `State`) so the
    /// headless harness can own a `State` with no surface at all.
    surface: Option<softbuffer::Surface<Rc<Window>, Rc<Window>>>,
    /// Where the workspace tree is loaded from and persisted to (CLI arg, or
    /// the default `~/.termspace/workspace01`).
    ws_path: PathBuf,
    /// The window is created hidden and revealed once, after the first frame
    /// is presented, so no blank surface flashes on open.
    revealed: bool,
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
            st.request_redraw();
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
        let (lw, lh) = st.logical_size();
        let size = st.grid_size(lw, lh);
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
        if let Some(tx) = self
            .state
            .as_ref()
            .and_then(|st| st.sessions.get(&node))
            .and_then(|s| s.tab.pty_tx.as_ref())
        {
            let _ = tx.send(Msg::Input(bytes.into()));
        }
    }

    /// Wheel scrolling for the shown session. `delta` is in text lines,
    /// positive when the wheel moves away from the user (scroll back into
    /// history). Fractional input (touchpads) is accumulated so nothing is
    /// lost. On the primary screen this walks the scrollback buffer; on the
    /// alternate screen (full-screen TUIs — `less`, `man`, `vim`, which keep
    /// no scrollback) it instead sends arrow keys, the usual "alternate
    /// scroll".
    fn scroll(&mut self, delta: f64) {
        let Some(st) = self.state.as_mut() else { return };
        let Some(node) = st.shown() else { return };
        st.scroll_acc += delta;
        let lines = st.scroll_acc.trunc() as i32;
        if lines == 0 {
            return;
        }
        st.scroll_acc -= lines as f64;

        let session = &st.sessions[&node];
        let mut term = session.tab.term.lock();
        if term.mode().contains(TermMode::ALT_SCREEN) {
            drop(term);
            // Up arrow on scroll-back, Down on scroll-forward.
            let seq: &[u8] = if lines > 0 { b"\x1b[A" } else { b"\x1b[B" };
            let mut bytes = Vec::with_capacity(seq.len() * lines.unsigned_abs() as usize);
            for _ in 0..lines.unsigned_abs() {
                bytes.extend_from_slice(seq);
            }
            if let Some(tx) = &session.tab.pty_tx {
                let _ = tx.send(Msg::Input(bytes.into()));
            }
        } else {
            term.scroll_display(Scroll::Delta(lines));
            drop(term);
            st.request_redraw();
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
        self.save_workspace();
        if let Some(st) = &self.state {
            st.request_redraw();
        }
    }

    /// Tear down the selected session. Dynamic (scratch) leaves are removed
    /// from the tree; spec leaves stay so they can be re-started.
    fn close_selected(&mut self) {
        let Some(st) = self.state.as_mut() else { return };
        let node = st.selected;
        if let Some(s) = st.sessions.remove(&node) {
            if let Some(tx) = &s.tab.pty_tx {
                let _ = tx.send(Msg::Shutdown);
            }
            st.id_of.retain(|_, &mut v| v != node);
        }
        let removed = st.tree.nodes[node].dynamic;
        if removed {
            if let Some(p) = st.tree.nodes[node].parent {
                // Land on the preceding session; fall back to the parent only
                // when there is no sibling before this one.
                let target = st.tree.prev_sibling(node).unwrap_or(p);
                st.tree.nodes[p].children.retain(|&c| c != node);
                st.selected = target;
            }
        }
        st.request_redraw();
        if removed {
            self.save_workspace();
        }
    }

    /// Move the sidebar selection by `delta` visible rows — the keyboard
    /// equivalent of clicking the row above/below (⌘↑/⌘↓, Ctrl+Shift+↑/↓
    /// elsewhere). It folds over the same `rows()` the sidebar draws, so up/down
    /// always tracks the picture on screen, and wraps around the ends so the
    /// tabs cycle. A selection that isn't currently visible (its group is
    /// collapsed) snaps back to the first row.
    fn select_relative(&mut self, delta: i32) {
        let Some(st) = self.state.as_mut() else { return };
        let rows = st.tree.rows();
        if rows.is_empty() {
            return;
        }
        let next = match rows.iter().position(|r| r.id == st.selected) {
            Some(i) => (i as i32 + delta).rem_euclid(rows.len() as i32) as usize,
            None => 0,
        };
        st.selected = rows[next].id;
        st.focus = None;
        st.request_redraw();
    }

    /// Fold (⌘←/Ctrl+Shift+←) or unfold (⌘→/Ctrl+Shift+→) the selected group.
    /// A no-op when a leaf is selected — leaves have nothing to expand — so the
    /// keystroke is simply swallowed rather than leaking to the PTY.
    fn set_group_expanded(&mut self, open: bool) {
        let Some(st) = self.state.as_mut() else { return };
        if !st.tree.is_group(st.selected) {
            return;
        }
        st.tree.nodes[st.selected].expanded = open;
        st.request_redraw();
    }

    /// Reflow every session's grid to the current terminal area. The window
    /// is shared, so a resize *or* an inspector toggle reflows them all (not
    /// just the visible one) to keep background sessions sane.
    fn relayout(&mut self) {
        let Some(st) = self.state.as_mut() else { return };
        st.sync_metrics();
        let (lw, lh) = st.logical_size();
        let size = st.grid_size(lw, lh);
        let ws = WindowSize {
            num_cols: size.cols as u16,
            num_lines: size.lines as u16,
            cell_width: st.renderer.cell_w as u16,
            cell_height: st.renderer.cell_h as u16,
        };
        for s in st.sessions.values_mut() {
            s.tab.size = size;
            s.tab.term.lock().resize(size);
            if let Some(tx) = &s.tab.pty_tx {
                let _ = tx.send(Msg::Resize(ws));
            }
        }
        st.request_redraw();
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

    /// Persist the tree to the `workspace` file so new sessions, renamed
    /// titles and edited commands survive a restart. Best-effort: a write
    /// failure is non-fatal.
    fn save_workspace(&self) {
        if let Some(st) = &self.state {
            let text = serialize_workspace(&st.tree, &home_dir());
            let _ = std::fs::write(&self.ws_path, text);
        }
    }

    /// Pin the selected leaf's default working directory to its session's
    /// *current* shell cwd (where you've `cd`'d to), then persist.
    fn use_current_cwd(&mut self) {
        let Some(st) = self.state.as_mut() else { return };
        let sel = st.selected;
        let Some(pid) = st.sessions.get(&sel).map(|s| s.shell_pid) else {
            return;
        };
        if let Some(cwd) = proc_cwd(pid) {
            st.tree.set_workdir(sel, cwd);
            st.request_redraw();
            self.save_workspace();
        }
    }

    /// Focus an inspector field, placing the caret under the click. Fields
    /// with no text for the selection (Command/Dir on a group) refuse focus.
    fn focus_field(&mut self, f: Field, click_x: f64) {
        let Some(st) = self.state.as_mut() else { return };
        let Some(text) = st.tree.field_text(st.selected, f) else {
            st.focus = None;
            return;
        };
        let fr = field_rects(st.logical_size().0, st.renderer.cell_h)[f.index()];
        let rel = (click_x - (fr.0 + 5) as f64).max(0.0);
        let idx = (rel / st.renderer.cell_w as f64).round() as usize;
        st.focus = Some(f);
        st.caret = idx.min(text.chars().count());
        st.request_redraw();
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
                st.request_redraw();
                return true;
            }
            Key::Named(NamedKey::Tab) => {
                // Hop to the next field that exists for this node (groups
                // only have Title), wrapping around.
                let order = Field::ALL;
                let cur = f.index();
                let next = (1..=order.len())
                    .map(|k| order[(cur + k) % order.len()])
                    .find(|&nf| st.tree.field_text(sel, nf).is_some())
                    .unwrap_or(Field::Title);
                st.caret = st
                    .tree
                    .field_text(sel, next)
                    .map(|t| t.chars().count())
                    .unwrap_or(0);
                st.focus = Some(next);
                st.request_redraw();
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
            // Absolute input is left as-is (stable caret); a typed leading
            // `~` is expanded on commit so `~/foo` still resolves.
            Field::Dir => st.tree.set_workdir(sel, expand_tilde(&next, &home_dir())),
        }
        st.caret = caret;
        st.request_redraw();
        self.save_workspace();
        true
    }

    /// Compose the frame and blit it onto the physical window surface. On a
    /// Retina display the chrome's *shapes* (pixel-art bevels/gradients) are
    /// nearest-neighbour upscaled — that keeps them crisp — but all *text*
    /// (chrome and terminal) is rendered fresh at true device resolution on
    /// top, so nothing looks doubled or soft. The harness skips this and reads
    /// `State::fb` directly. Non-HiDPI is a straight copy.
    fn redraw(&mut self) {
        let scaled = if let Some(st) = self.state.as_mut() {
            st.sync_metrics();
            let (lw, lh) = st.logical_size();
            let scaled = (lw, lh) != st.phys;
            // Capture chrome text instead of drawing it into the logical frame,
            // so it doesn't get upscaled-and-chunky — we replay it crisply below.
            st.renderer.text_log = if scaled { Some(Vec::new()) } else { None };
            st.paint();
            scaled
        } else {
            return;
        };
        let App { state, surface, revealed, .. } = self;
        let (Some(st), Some(surface)) = (state.as_mut(), surface.as_mut()) else {
            return;
        };
        // Map the window the first time a frame is about to reach the screen —
        // *before* `present()`, not after. An X11 window has no backing store,
        // so presenting while it's still hidden is discarded, and mapping an
        // un-painted window flashes whatever is behind it. Mapping first, then
        // presenting in the same synchronous block, makes the server process
        // Map→PutImage in order so the compositor's first composite already
        // holds our pixels — no flash, no see-through frame.
        let reveal = |st: &State, revealed: &mut bool| {
            if !*revealed {
                if let Some(w) = &st.window {
                    w.set_visible(true);
                }
                *revealed = true;
            }
        };
        let (pw, ph) = st.phys;
        let (Some(w), Some(h)) = (NonZeroU32::new(pw as u32), NonZeroU32::new(ph as u32)) else {
            return;
        };
        let (lw, lh) = st.logical_size();
        surface.resize(w, h).unwrap();
        let mut buf = surface.buffer_mut().unwrap();
        if !scaled {
            // Non-HiDPI: the logical frame already is the physical frame.
            buf.copy_from_slice(&st.fb);
            reveal(st, revealed);
            buf.present().unwrap();
            return;
        }
        // Nearest-neighbour upscale of the shape layer (text was captured, not
        // drawn, so this doesn't blur any glyphs).
        for py in 0..ph {
            let ly = (py * lh / ph).min(lh - 1);
            let (srow, drow) = (ly * lw, py * pw);
            for px in 0..pw {
                let lx = (px * lw / pw).min(lw - 1);
                buf[drow + px] = st.fb[srow + lx];
            }
        }
        // Now render text at true device resolution over the upscaled shapes:
        // the terminal grid first, then the captured chrome strings.
        st.overdraw_terminal_physical(&mut buf[..], pw, ph);
        let (sx, sy) = (pw as f64 / lw as f64, ph as f64 / lh as f64);
        if let Some(cmds) = st.renderer.text_log.take() {
            render_text_cmds(&mut buf[..], pw, ph, &mut st.renderer, cmds, sx, sy);
        }
        reveal(st, revealed);
        buf.present().unwrap();
    }

    fn send(&self, bytes: Vec<u8>) {
        let Some(st) = self.state.as_ref() else { return };
        let Some(node) = st.shown() else { return };
        // Typing snaps back to the prompt if we were scrolled up, like xterm.
        let mut term = st.sessions[&node].tab.term.lock();
        if term.grid().display_offset() != 0 {
            term.scroll_display(Scroll::Bottom);
            drop(term);
            st.request_redraw();
        } else {
            drop(term);
        }
        self.send_to(node, bytes);
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

/// Draw a left-aligned monospace string, clipped to `max_w` pixels. Chrome
/// text is always the Regular face at the logical size. When the renderer's
/// `text_log` is armed (the GUI's Retina path), the string is *recorded* in
/// logical coordinates and replayed crisply later by [`render_text_cmds`]
/// instead of being rasterized here.
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
    if let Some(log) = &mut r.text_log {
        log.push(TextCmd {
            x,
            y,
            max_w,
            text: text.to_string(),
            color,
        });
        return;
    }
    let cw = r.cell_w;
    let mut pen = x;
    for c in text.chars() {
        if pen + cw > x + max_w {
            break;
        }
        if c != ' ' {
            let g = r.glyph(c, FontStyle::Regular, FONT_PX);
            for gy in 0..g.h {
                let py = y as i32 + g.top + gy as i32;
                if py < 0 || py as usize >= ph {
                    continue;
                }
                for gx in 0..g.w {
                    let px = pen as i32 + g.left + gx as i32;
                    if px < 0 || px as usize >= pw {
                        continue;
                    }
                    let a = g.bitmap[gy * g.w + gx] as u32;
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

/// Replay captured chrome-text commands onto `buf` at device resolution: each
/// logical command is re-laid at `(sx, sy)` with glyphs rasterized at
/// `FONT_PX × sy`, so chrome text comes out as crisp as the terminal grid on a
/// Retina display (rather than being nearest-neighbour-doubled and chunky).
fn render_text_cmds(buf: &mut [u32], bw: usize, bh: usize, r: &mut Renderer, cmds: Vec<TextCmd>, sx: f64, sy: f64) {
    let font_px = FONT_PX * sy as f32;
    let cell_w = ((r.cell_w as f64 * sx).round() as usize).max(1);
    for cmd in cmds {
        let x0 = (cmd.x as f64 * sx).round() as usize;
        let y0 = (cmd.y as f64 * sy).round() as usize;
        let max_w = (cmd.max_w as f64 * sx).round() as usize;
        let mut pen = x0;
        for c in cmd.text.chars() {
            if pen + cell_w > x0 + max_w {
                break;
            }
            if c != ' ' {
                let g = r.glyph(c, FontStyle::Regular, font_px);
                for gy in 0..g.h {
                    let py = y0 as i32 + g.top + gy as i32;
                    if py < 0 || py as usize >= bh {
                        continue;
                    }
                    for gx in 0..g.w {
                        let px = pen as i32 + g.left + gx as i32;
                        if px < 0 || px as usize >= bw {
                            continue;
                        }
                        let a = g.bitmap[gy * g.w + gx] as u32;
                        if a == 0 {
                            continue;
                        }
                        let idx = py as usize * bw + px as usize;
                        buf[idx] = blend(cmd.color, buf[idx], a);
                    }
                }
            }
            pen += cell_w;
        }
    }
}

/// Render the visible grid of one terminal into `buf` at an arbitrary scale.
/// All geometry is in *target* pixels, so the same routine serves the logical
/// 1× compose (`paint`) and the crisp device-resolution Retina pass (`redraw`):
/// pass cell/origin sizes already multiplied by the device scale and `font_px =
/// FONT_PX × scale`. Honours per-cell colour, the cursor block, selection,
/// bold/italic faces, dim, underline/strikeout and Unicode-9 wide cells.
#[allow(clippy::too_many_arguments)]
fn draw_terminal_cells(
    buf: &mut [u32],
    bw: usize,
    bh: usize,
    r: &mut Renderer,
    term: &Term<Listener>,
    lines: i32,
    origin_x: usize,
    origin_y: usize,
    clip_right: usize,
    cell_w: usize,
    cell_h: usize,
    font_px: f32,
) {
    // `display_iter` numbers scrollback with negative grid lines; the visible
    // viewport is shifted by the scroll offset, so a grid line maps to on-screen
    // row `line + display_offset`.
    let offset = term.grid().display_offset() as i32;
    let content = term.renderable_content();
    let cursor = content.cursor.point;
    let selection = content.selection;
    for cell in content.display_iter {
        let row = cell.point.line.0 + offset;
        if row < 0 || row >= lines {
            continue;
        }
        // The trailing half of a wide (double-width) char carries no glyph of
        // its own — the lead cell's glyph spans into it.
        if cell.flags.contains(Flags::WIDE_CHAR_SPACER) {
            continue;
        }
        let col = cell.point.column.0;
        let x0 = origin_x + col * cell_w;
        let y0 = origin_y + row as usize * cell_h;
        let wide = cell.flags.contains(Flags::WIDE_CHAR);
        let span = if wide { cell_w * 2 } else { cell_w };

        let is_cursor = cell.point.line == cursor.line && cell.point.column == cursor.column;
        let selected = selection.is_some_and(|s| s.contains(cell.point));
        let mut fg = rgb(cell.fg, FG);
        let mut bg = rgb(cell.bg, BG);
        if cell.flags.contains(Flags::DIM) {
            fg = blend(fg, bg, 150); // dim = pull the ink toward its background
        }
        if is_cursor {
            std::mem::swap(&mut fg, &mut bg);
        } else if selected {
            bg = SEL;
        }
        if bg != BG {
            fill_rect(buf, bw, bh, x0, y0, span.min(clip_right.saturating_sub(x0)), cell_h, bg);
        }

        let c = cell.c;
        let drawable = c != ' ' && c != '\0' && !cell.flags.contains(Flags::HIDDEN);
        if drawable {
            let g = r.glyph(c, font_style(cell.flags), font_px);
            for gy in 0..g.h {
                let py = y0 as i32 + g.top + gy as i32;
                if py < 0 || py as usize >= bh {
                    continue;
                }
                for gx in 0..g.w {
                    let px = x0 as i32 + g.left + gx as i32;
                    if px < origin_x as i32 || px as usize >= clip_right {
                        continue;
                    }
                    let a = g.bitmap[gy * g.w + gx] as u32;
                    if a == 0 {
                        continue;
                    }
                    let idx = py as usize * bw + px as usize;
                    buf[idx] = blend(fg, buf[idx], a);
                }
            }
        }

        // Decorations sit on the baseline-ish; thickness scales with the cell.
        let thick = (cell_h / 14).max(1);
        if cell.flags.contains(Flags::UNDERLINE) {
            let uy = y0 + cell_h.saturating_sub(thick + 1);
            hline(buf, bw, bh, x0, uy, span, clip_right, thick, fg);
        }
        if cell.flags.contains(Flags::STRIKEOUT) {
            let sy = y0 + cell_h / 2;
            hline(buf, bw, bh, x0, sy, span, clip_right, thick, fg);
        }
    }
}

/// A clipped horizontal rule `thick` px tall (terminal underline/strikeout).
fn hline(
    buf: &mut [u32],
    bw: usize,
    bh: usize,
    x: usize,
    y: usize,
    w: usize,
    clip_right: usize,
    thick: usize,
    color: u32,
) {
    let right = (x + w).min(clip_right).min(bw);
    for yy in y..(y + thick).min(bh) {
        for xx in x..right {
            buf[yy * bw + xx] = color;
        }
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
    // (No "WORKSPACE" header label — intentionally hidden.)
    // Text sits centred within a *row*, not the taller header band, so glyphs
    // never ride low against the row below.
    let ty = ROW_H.saturating_sub(r.cell_h) / 2;
    // Hard divider between the pane and the terminal.
    fill_rect(buf, pw, ph, SIDEBAR_W - 1, 0, 1, ph, BEVEL_DK);
    // Latched info icon directly below the WORKSPACE head, on its own toolbar
    // band: a subtle gradient + bottom divider sets the "toolbar" off from the
    // list so the icon doesn't float in empty space.
    vgradient(buf, pw, ph, 0, HEADER_H, SIDEBAR_W, INFO_H, HEAD_HI, HEAD_LO);
    fill_rect(buf, pw, ph, 0, HEADER_H + INFO_H - 1, SIDEBAR_W, 1, BEVEL_DK);
    fill_rect(buf, pw, ph, 0, HEADER_H + INFO_H, SIDEBAR_W, 1, BEVEL_LT);
    draw_info_icon(buf, pw, ph, info_btn(), inspector);

    for (i, row) in rows.iter().enumerate() {
        let y = HEADER_H + INFO_H + 1 + i * ROW_H;
        if y + ROW_H > ph {
            break;
        }
        let ind = row.depth * INDENT;
        // One column grid shared by groups and leaves: a fixed left gutter, the
        // marker column, a gap, then the label — so every label at a given
        // depth lines up regardless of node kind.
        let mark_x = TPAD + ind;
        let text_x = mark_x + EXPANDER_W + MARK_GAP;
        let text_w = SIDEBAR_W.saturating_sub(text_x + TPAD);
        let is_sel = row.id == selected;
        if is_sel {
            // A flat raised bar (top highlight + bottom shadow) with a coloured
            // accent down the left edge — reads as a selected list row, not a
            // detached button.
            vgradient(buf, pw, ph, 1, y, SIDEBAR_W - 2, ROW_H, PANEL_HI, PANEL_LO);
            fill_rect(buf, pw, ph, 1, y, SIDEBAR_W - 2, 1, BEVEL_LT);
            fill_rect(buf, pw, ph, 1, y + ROW_H - 1, SIDEBAR_W - 2, 1, BEVEL_DK);
            fill_rect(buf, pw, ph, 1, y, 3, ROW_H, SEL);
        }
        let gy = y + ty;
        if row.is_group {
            // Boxed expander, classic tree control — a tidy square centred in
            // the marker column.
            let bw = EXPANDER_W - 4;
            let bx = mark_x + (EXPANDER_W - bw) / 2;
            let by = y + (ROW_H - bw) / 2;
            stroke_rect(buf, pw, ph, bx, by, bw, bw, BEVEL_DK);
            let mark = if row.expanded { "-" } else { "+" };
            let glyph_ok = row.has_children;
            draw_text(
                buf,
                pw,
                ph,
                r,
                bx + 1,
                gy,
                EXPANDER_W,
                if glyph_ok { mark } else { " " },
                INK,
            );
            draw_text(buf, pw, ph, r, text_x, gy, text_w, &row.name, INK);
        } else {
            let live = sessions.contains_key(&row.id);
            let running = sessions
                .get(&row.id)
                .map(|s| observe(s.shell_pid).foreground)
                .unwrap_or(false);
            // Run-state marker drawn (not a font glyph) so it stays a crisp,
            // fixed size and centres in the same column as the expander box:
            // filled square = running, hollow square = live/idle, small dot =
            // not live.
            let cx = mark_x + EXPANDER_W / 2;
            let cy = y + ROW_H / 2;
            if running {
                fill_rect(buf, pw, ph, cx - 3, cy - 3, 7, 7, RUN_INK);
            } else if live {
                stroke_rect(buf, pw, ph, cx - 4, cy - 4, 8, 8, INK_DIM);
            } else {
                fill_rect(buf, pw, ph, cx - 1, cy - 1, 3, 3, INK_DIM);
            }
            draw_text(
                buf,
                pw,
                ph,
                r,
                text_x,
                gy,
                text_w,
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

/// A Win2k push-button: raised by default, sunken+inset when `pressed`, dim
/// when `!enabled`. Used for the window controls and the use-cwd action.
fn draw_button(
    buf: &mut [u32],
    pw: usize,
    ph: usize,
    r: &mut Renderer,
    rect: Rect,
    label: &str,
    pressed: bool,
    enabled: bool,
) {
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
    draw_text(buf, pw, ph, r, tx, ty, w, label, if enabled { INK } else { INK_DIM });
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
/// selected node's path and editable Title / Default-command / Working-dir
/// fields, plus a "use current working dir" button.
fn draw_inspector(
    buf: &mut [u32],
    pw: usize,
    ph: usize,
    r: &mut Renderer,
    tree: &Tree,
    sel: NodeId,
    focus: Option<Field>,
    caret: usize,
    can_use_cwd: bool,
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
    let rects = field_rects(pw, cell_h);
    let lab_y = |by: usize| by.saturating_sub(cell_h + 4);
    for f in Field::ALL {
        let rect = rects[f.index()];
        let label = match f {
            Field::Title => "Title",
            Field::Command => "Default command",
            Field::Dir => "Working directory",
        };
        draw_text(buf, pw, ph, r, rect.0, lab_y(rect.1), RPANEL_W, label, INK);
        match tree.field_text(sel, f) {
            Some(t) => draw_field(buf, pw, ph, r, rect, &t, focus == Some(f), caret, true),
            None => {
                let dis = match f {
                    Field::Command => "(group \u{2014} no command)",
                    Field::Dir => "(group \u{2014} no directory)",
                    Field::Title => "",
                };
                draw_field(buf, pw, ph, r, rect, dis, false, 0, false);
            }
        }
    }
    draw_button(
        buf,
        pw,
        ph,
        r,
        usecwd_btn(pw, cell_h),
        "Use current working dir",
        false,
        can_use_cwd,
    );
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
    (TPAD, HEADER_H + (INFO_H - s) / 2, s, s)
}

/// `[title, command, directory]` boxes inside the inspector pane, in
/// `Field::ALL` order (so `Field::index` is the row index here).
fn field_rects(pw: usize, cell_h: usize) -> [Rect; 3] {
    let px = panel_x(pw);
    let pad = 12;
    let fx = px + pad;
    let fw = RPANEL_W - pad * 2;
    let bh = cell_h + 8;
    // Each box sits exactly `cell_h + 4` below its label (see `lab_y`), and
    // the first label clears the path line under the PROPERTIES head.
    let lab0 = HEADER_H + 8 + cell_h + 10;
    let y0 = lab0 + cell_h + 4;
    let step = bh + 16 + cell_h + 4; // box -> next label -> next box
    [
        (fx, y0, fw, bh),
        (fx, y0 + step, fw, bh),
        (fx, y0 + 2 * step, fw, bh),
    ]
}

/// The "Use current working dir" button, just below the directory box.
fn usecwd_btn(pw: usize, cell_h: usize) -> Rect {
    let (x, y, w, h) = field_rects(pw, cell_h)[Field::Dir.index()];
    (x, y + h + 8, w, h)
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

/// The OS cursor that signals each resize direction, so the pointer turns into
/// the matching double-headed arrow when it enters a window edge or corner.
fn resize_cursor(dir: ResizeDirection) -> CursorIcon {
    match dir {
        ResizeDirection::North => CursorIcon::NResize,
        ResizeDirection::South => CursorIcon::SResize,
        ResizeDirection::East => CursorIcon::EResize,
        ResizeDirection::West => CursorIcon::WResize,
        ResizeDirection::NorthEast => CursorIcon::NeResize,
        ResizeDirection::NorthWest => CursorIcon::NwResize,
        ResizeDirection::SouthEast => CursorIcon::SeResize,
        ResizeDirection::SouthWest => CursorIcon::SwResize,
    }
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

/// Which workspace file to use: the first CLI argument if given (a leading
/// `~` is expanded), else the default `~/.termspace/workspace01`. A
/// missing/blank argument falls back to the default. If the file doesn't
/// exist the app just opens a blank workspace and creates it on first save.
fn resolve_workspace_path() -> PathBuf {
    match std::env::args().nth(1) {
        Some(a) if !a.trim().is_empty() => expand_tilde(a.trim(), &home_dir()),
        _ => home_dir().join(".termspace").join("workspace01"),
    }
}

/// The tree a brand-new (missing/empty) workspace opens with: a `Project`
/// group holding one session in the current working directory, plus the
/// usual Scratch/Transient areas (appended by `parse_workspace`). Nothing is
/// written to disk — this only lives in memory until the first save.
fn default_workspace_text() -> String {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let name = cwd
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("session");
    format!(
        "workspaces\n\tProject\n\t\t\"{name}\"\t{}\n",
        cwd.display()
    )
}

/// Tell X11/the compositor the window is fully opaque and clear its backing to
/// black, before it's mapped. Without this, a freshly-mapped compositor-redirected
/// window has an undefined (often see-through) pixmap for the frame or two before
/// our first `present()` lands — which shows as a flash of the desktop behind it.
/// Best-effort: any failure (Wayland, no X server, refused request) just falls
/// through to the previous behaviour.
#[cfg(target_os = "linux")]
fn mark_window_opaque(window: &Window) {
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use x11rb::connection::Connection;
    use x11rb::protocol::xproto::{
        AtomEnum, ChangeWindowAttributesAux, ConnectionExt as _, PropMode,
    };
    // `change_property32` lives on a separate helper trait.
    use x11rb::wrapper::ConnectionExt as _;

    let xid = match window.window_handle().map(|h| h.as_raw()) {
        Ok(RawWindowHandle::Xlib(h)) => h.window as u32,
        Ok(RawWindowHandle::Xcb(h)) => h.window.get(),
        // Wayland (or anything else): no see-through-on-map problem to solve.
        _ => return,
    };
    let Ok((conn, _screen)) = x11rb::connect(None) else {
        return;
    };
    // New/exposed regions clear to black instead of leaving stale/transparent
    // pixels for the compositor to show.
    let _ = conn.change_window_attributes(
        xid,
        &ChangeWindowAttributesAux::new().background_pixel(0),
    );
    // `_NET_WM_OPAQUE_REGION = whole window` → the compositor won't alpha-blend
    // the surface against the desktop, even before we've drawn into it.
    if let Ok(reply) = conn
        .intern_atom(false, b"_NET_WM_OPAQUE_REGION")
        .map_err(drop)
        .and_then(|c| c.reply().map_err(drop))
    {
        // Oversized on purpose — the compositor clips the region to the actual
        // window, so this stays correct even if the WM resizes us after map
        // (the pre-map `inner_size()` can't be trusted for that).
        let region = [0u32, 0, 1 << 15, 1 << 15];
        let _ = conn.change_property32(
            PropMode::REPLACE,
            xid,
            reply.atom,
            AtomEnum::CARDINAL,
            &region,
        );
    }
    // Round-trip so the server has applied all of the above before winit maps
    // the window on its own (separate) connection.
    let _ = conn.flush();
    let _ = conn.get_input_focus().map(|c| c.reply());
}

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        // No OS title bar: we draw our own in the header row to reclaim that
        // strip of screen.
        // `mut` is only used by the Linux WM-class block below.
        #[allow(unused_mut)]
        // Start hidden so the compositor never shows the empty/garbage surface
        // before our first frame lands; `redraw` reveals it after the first
        // `present()`.
        let mut attrs = Window::default_attributes()
            .with_title("termem")
            .with_decorations(false)
            .with_visible(false);
        // Pin a stable WM class / Wayland app_id so the desktop entry's
        // `StartupWMClass=termem` binds the launcher icon to this window
        // (see scripts/install-icon.sh).
        #[cfg(target_os = "linux")]
        {
            use winit::platform::wayland::WindowAttributesExtWayland;
            use winit::platform::x11::WindowAttributesExtX11;
            attrs = WindowAttributesExtX11::with_name(attrs, "termem", "termem");
            attrs = WindowAttributesExtWayland::with_name(attrs, "termem", "termem");
        }
        let window = Rc::new(
            event_loop.create_window(attrs).expect("create window"),
        );
        // Mark the X11 window opaque + black-backed *before* it maps, so the
        // compositor's first composite of it is a flat dark frame rather than a
        // see-through hole onto the desktop while our pixels are still in flight.
        #[cfg(target_os = "linux")]
        mark_window_opaque(&window);
        let renderer = Renderer::new();
        // Load the glyph-coverage fallback fonts off the critical path; they
        // arrive back as `UserEvent::FallbackFonts` once parsed.
        {
            let proxy = self.proxy.clone();
            std::thread::spawn(move || {
                let _ = proxy.send_event(UserEvent::FallbackFonts(load_fallback_fonts()));
            });
        }
        let ctx = softbuffer::Context::new(window.clone()).unwrap();
        self.surface = Some(softbuffer::Surface::new(&ctx, window.clone()).unwrap());

        let home = home_dir();
        let mut ws_text = std::fs::read_to_string(&self.ws_path).unwrap_or_default();
        if ws_text.trim().is_empty() {
            ws_text = default_workspace_text();
        }
        let tree = parse_workspace(&ws_text, &home);

        let inner = window.inner_size();
        let st = State {
            phys: (inner.width as usize, inner.height as usize),
            scale: window.scale_factor().max(1.0),
            window: Some(window),
            fb: Vec::new(),
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
            scroll_acc: 0.0,
            cursor: CursorIcon::Default,
        };
        self.state = Some(st);

        // Paint and reveal the window *before* spawning any shells: the chrome
        // (sidebar tree, header, backdrop) is fully determined by the parsed
        // workspace, so there's nothing to wait for. Forking a PTY per leaf
        // costs ~100ms and produces no visible content anyway — a terminal only
        // shows what its shell writes, which streams in asynchronously — so it
        // has no business being on the path to first paint.
        //
        // Twice on purpose: the first call maps the window (`set_visible`) then
        // presents; the second guarantees a present lands *after* the map even
        // though winit (map) and softbuffer (present) drive separate X11
        // connections whose request order isn't otherwise synchronised. Cheap
        // (one extra chrome paint at startup) insurance against a see-through
        // first frame.
        self.redraw();
        self.redraw();

        let st = self.state.as_mut().unwrap();
        let (lw, lh) = st.logical_size();
        let size = st.grid_size(lw, lh);

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
        st.request_redraw();
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::Wakeup => {
                if let Some(st) = &self.state {
                    st.request_redraw();
                }
            }
            UserEvent::Exit(id) => {
                let mut removed_dynamic = false;
                if let Some(st) = self.state.as_mut() {
                    if let Some(&node) = st.id_of.get(&id) {
                        if let Some(s) = st.sessions.remove(&node) {
                            if let Some(tx) = &s.tab.pty_tx {
                                let _ = tx.send(Msg::Shutdown);
                            }
                        }
                        st.id_of.remove(&id);
                        // A scratch tab disappears when its shell exits; a
                        // spec leaf stays so Start can bring it back.
                        if st.tree.nodes[node].dynamic {
                            removed_dynamic = true;
                            if let Some(p) = st.tree.nodes[node].parent {
                                let target = st.tree.prev_sibling(node).unwrap_or(p);
                                st.tree.nodes[p].children.retain(|&c| c != node);
                                if st.selected == node {
                                    st.selected = target;
                                }
                            }
                        }
                        st.request_redraw();
                    }
                }
                if removed_dynamic {
                    self.save_workspace();
                }
            }
            UserEvent::Title(id, title) => {
                if let Some(st) = self.state.as_mut() {
                    if let Some(&node) = st.id_of.get(&id) {
                        if let Some(s) = st.sessions.get_mut(&node) {
                            s.tab.title = title;
                        }
                        st.request_redraw();
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
                        st.request_redraw();
                    }
                }
            }
            UserEvent::FallbackFonts(fonts) => {
                if let Some(st) = self.state.as_mut() {
                    st.renderer.fallbacks = fonts;
                    // Glyphs rendered before the fallbacks arrived were cached
                    // using the primary face (tofu for chars it lacks); drop the
                    // cache so they re-rasterize with full coverage.
                    st.renderer.cache.clear();
                    st.request_redraw();
                }
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => self.redraw(),
            // Repaint *synchronously* on every size change. During a resize the
            // OS runs its own modal loop, so a queued `request_redraw` isn't
            // serviced until the drag ends — leaving a frozen, stretched frame.
            // Drawing here makes the content track the window edge live instead.
            WindowEvent::Resized(_) => {
                self.relayout();
                self.redraw();
            }
            // Moving between displays of different density (or an OS zoom
            // change) must reflow: `relayout` re-reads size *and* scale off
            // the window, keeping the logical layout constant.
            WindowEvent::ScaleFactorChanged { .. } => self.relayout(),
            WindowEvent::ModifiersChanged(m) => {
                if let Some(st) = self.state.as_mut() {
                    st.mods = m.state();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(st) = self.state.as_mut() {
                    // winit reports physical pixels; the UI lives in logical
                    // space, so divide by the scale before hit-testing.
                    let s = st.scale.max(1.0);
                    st.mouse = (position.x / s, position.y / s);
                    // Show a resize cursor over the window's edge/corner grips
                    // so the resize affordance is discoverable; fall back to the
                    // default arrow everywhere else. Only push to the OS when the
                    // shape changes to avoid a `set_cursor` per mouse-move.
                    let (pw, ph) = st.logical_size();
                    let want = resize_dir(pw, ph, st.mouse.0, st.mouse.1)
                        .map(resize_cursor)
                        .unwrap_or(CursorIcon::Default);
                    if want != st.cursor {
                        st.cursor = want;
                        if let Some(w) = &st.window {
                            w.set_cursor(want);
                        }
                    }
                    if st.selecting {
                        if let Some(node) = st.shown() {
                            let size = st.sessions[&node].tab.size;
                            let mut term = st.sessions[&node].tab.term.lock();
                            let (point, side) = st.pixel_to_point(&term, size);
                            if let Some(sel) = term.selection.as_mut() {
                                sel.update(point, side);
                            }
                            drop(term);
                            st.request_redraw();
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let Some(stref) = self.state.as_ref() else { return };
                let (mx, my) = stref.mouse;
                match (button, state) {
                    (MouseButton::Left, ElementState::Pressed) => {
                        let (pw, ph) = self.state.as_ref().unwrap().logical_size();

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
                                st.request_redraw();
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
                            if let Some(w) = &self.state.as_ref().unwrap().window {
                                w.set_minimized(true);
                            }
                            return;
                        }
                        if hit(bmax, mx, my) {
                            if let Some(w) = self.state.as_ref().unwrap().window.clone() {
                                w.set_maximized(!w.is_maximized());
                            }
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
                            let rects = field_rects(pw, ch);
                            if let Some(f) =
                                Field::ALL.into_iter().find(|f| hit(rects[f.index()], mx, my))
                            {
                                self.focus_field(f, mx);
                            } else if hit(usecwd_btn(pw, ch), mx, my) {
                                self.use_current_cwd();
                            } else if let Some(st) = self.state.as_mut() {
                                st.focus = None;
                                st.request_redraw();
                            }
                            return;
                        }
                        // 5. Borderless-window resize grips.
                        if let Some(dir) = resize_dir(pw, ph, mx, my) {
                            if let Some(w) = &self.state.as_ref().unwrap().window {
                                let _ = w.drag_resize_window(dir);
                            }
                            return;
                        }
                        // 6. The header row (anywhere else): double-click
                        // toggles maximize/restore (like a native title bar);
                        // a single click drags the window.
                        if my < HEADER_H as f64 {
                            let now = std::time::Instant::now();
                            let st = self.state.as_ref().unwrap();
                            let double = st.last_click.is_some_and(|(t, p)| {
                                now.duration_since(t).as_millis() < 400
                                    && (p.0 - mx).abs() < 4.0
                                    && (p.1 - my).abs() < 4.0
                            });
                            if double {
                                if let Some(w) = st.window.clone() {
                                    w.set_maximized(!w.is_maximized());
                                }
                                self.state.as_mut().unwrap().last_click = None;
                                return;
                            }
                            self.state.as_mut().unwrap().last_click = Some((now, (mx, my)));
                            if let Some(w) = &self.state.as_ref().unwrap().window {
                                let _ = w.drag_window();
                            }
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
                            st.request_redraw();
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
                                st.request_redraw();
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
                            st.request_redraw();
                        } else if let Some(st) = self.state.as_mut() {
                            st.ctx = None;
                            st.request_redraw();
                        }
                    }
                    (MouseButton::Middle, ElementState::Pressed) => self.paste(),
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                // A wheel notch ≈ 3 lines; a high-resolution touchpad reports
                // pixels, converted to lines by the cell height. Positive =
                // away from the user = back into scrollback.
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64 * 3.0,
                    MouseScrollDelta::PixelDelta(p) => {
                        let ch = self.state.as_ref().map(|s| s.renderer.cell_h).unwrap_or(1);
                        p.y / ch.max(1) as f64
                    }
                };
                self.scroll(lines);
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
                if let Some(sc) = match_shortcut(kmods, &event.logical_key) {
                    let sel = self.state.as_ref().map(|s| s.selected);
                    match sc {
                        Shortcut::Copy => self.copy_to_clipboard(),
                        Shortcut::Paste => self.paste(),
                        Shortcut::NewTab => self.new_scratch(),
                        Shortcut::CloseTab => self.close_selected(),
                        Shortcut::SelectPrev => self.select_relative(-1),
                        Shortcut::SelectNext => self.select_relative(1),
                        Shortcut::Collapse => self.set_group_expanded(false),
                        Shortcut::Expand => self.set_group_expanded(true),
                        Shortcut::Start => {
                            if let Some(n) = sel {
                                self.start(n);
                            }
                        }
                        Shortcut::Stop => {
                            if let Some(n) = sel {
                                self.stop(n);
                            }
                        }
                    }
                    return;
                }
                // On macOS, swallow any other ⌘-combo so it can't leak into the
                // PTY as stray text. (Ctrl-combos fall through to the shell.)
                #[cfg(target_os = "macos")]
                if kmods.super_key() {
                    return;
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

/// Real entry point: build the winit event loop and run the GUI app. The
/// binary (`src/main.rs`) is a thin shim over this so the rest of the crate
/// stays a library the integration harness (`testkit`) can drive headlessly.
pub fn run() {
    // Warm the backdrop decode off-thread so its one-time ~15ms cost overlaps
    // event-loop + font setup instead of landing on the first paint. `backdrop()`
    // caches into a `OnceLock`, so the first `paint()` just reads the result.
    std::thread::spawn(|| {
        let _ = backdrop();
    });
    let event_loop = EventLoop::<UserEvent>::with_user_event()
        .build()
        .expect("build event loop");
    event_loop.set_control_flow(ControlFlow::Wait);
    let proxy = event_loop.create_proxy();
    let ws_path = resolve_workspace_path();
    let mut app = App {
        proxy,
        state: None,
        surface: None,
        ws_path,
        revealed: false,
    };
    event_loop.run_app(&mut app).expect("run");
}

// ===========================================================================
// tests — the functional core, exercised without a display
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // A fixed fixture in the original hand-written style. Tests must not
    // depend on the live `workspace` file — the app rewrites it (persisted
    // sessions, edits), so coupling tests to it makes them flaky.
    const FIXTURE: &str = "workspaces
\tApps
\t\t\"qwen\"\t/home/liam/.apps\t./run-llama.sh
\tMusic
\t\t\"Hermes chat\"\t~/Music\thermes
\t\t\"Wanted music\"\t~/Music\t\"nano wanted\"
\tDiabetes
\t\t\"Meal planner\"\t~/Documents/projects/diabetes/mealplanner\t \"bun run main.ts\"
\t\t\"Sugar tracker\"\t/home/liam/Documents/projects/diabetes/sugar\t./run.sh
\tMorphology
\t\t\"morpheus\"\t~/Documents/projects/morpheus\tbash
\tHTGAA
\t\t\"website\"\t/home/liam/Documents/projects/webpages\tbash
\tStudy
\t\t\"Papers\"\t~/Documents/papers\tbash
\t\t\"Podcasts\"\t/home/liam/Dropbox/podcast-learn\tbash
";

    fn tree() -> Tree {
        parse_workspace(FIXTURE, Path::new("/home/u"))
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
    fn closing_selects_preceding_sibling_then_parent() {
        let t = tree();
        let music = group(&t, "Music");
        let [a, b] = t.leaves(music)[..] else { panic!() };
        // Closing the 2nd session lands on the 1st (the preceding one)...
        assert_eq!(t.prev_sibling(b), Some(a));
        // ...and closing the 1st has no preceding sibling, so the caller
        // falls back to the parent group.
        assert_eq!(t.prev_sibling(a), None);
        assert_eq!(t.prev_sibling(a).unwrap_or(music), music);
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
        assert_eq!(
            t.field_text(leaf, Field::Dir).as_deref(),
            Some("/home/u/Music")
        );
        // A group has a title but no command/dir (those render disabled).
        assert_eq!(t.field_text(music, Field::Title).as_deref(), Some("Music"));
        assert_eq!(t.field_text(music, Field::Command), None);
        assert_eq!(t.field_text(music, Field::Dir), None);

        // Editing the directory field round-trips back into the tree and the
        // saved file (a typed leading `~` expands on commit).
        let mut t = t;
        t.set_workdir(leaf, expand_tilde("~/elsewhere", Path::new("/home/u")));
        assert_eq!(
            t.field_text(leaf, Field::Dir).as_deref(),
            Some("/home/u/elsewhere")
        );
        let reloaded =
            parse_workspace(&serialize_workspace(&t, Path::new("/home/u")), Path::new("/home/u"));
        let rleaf = reloaded.leaves(group(&reloaded, "Music"))[0];
        assert_eq!(reloaded.leaf_spec(rleaf).unwrap().0, Path::new("/home/u/elsewhere"));
    }

    /// Flatten the tree to a comparable shape: (depth, name, kind, workdir,
    /// command) per node in DFS order.
    fn shape(t: &Tree, home: &Path) -> Vec<(usize, String, &'static str, String, String)> {
        fn go(
            t: &Tree,
            id: NodeId,
            d: usize,
            home: &Path,
            out: &mut Vec<(usize, String, &'static str, String, String)>,
        ) {
            for &c in &t.nodes[id].children {
                let n = &t.nodes[c];
                let row = match &n.kind {
                    Kind::Group => (d, n.name.clone(), "g", String::new(), String::new()),
                    Kind::Leaf { workdir, command } => (
                        d,
                        n.name.clone(),
                        "l",
                        collapse_tilde(workdir, home),
                        command.clone(),
                    ),
                    Kind::Root => continue,
                };
                out.push(row);
                go(t, c, d + 1, home, out);
            }
        }
        let mut v = Vec::new();
        go(t, t.root, 0, home, &mut v);
        v
    }

    #[test]
    fn workspace_round_trips_through_disk_format() {
        let home = Path::new("/home/u");
        let t1 = parse_workspace(FIXTURE, home);
        let text = serialize_workspace(&t1, home);
        let t2 = parse_workspace(&text, home);
        // Re-parsing the serialized form yields the same tree...
        assert_eq!(shape(&t1, home), shape(&t2, home));
        // ...and the synthetic homes are present exactly once (not doubled
        // by the round trip).
        let count = |t: &Tree, name: &str| {
            t.nodes[t.root]
                .children
                .iter()
                .filter(|&&c| t.nodes[c].name == name)
                .count()
        };
        assert_eq!(count(&t2, "Scratch"), 1);
        assert_eq!(count(&t2, "Transient"), 1);
    }

    #[test]
    fn saved_scratch_session_reloads_as_a_leaf() {
        // A new scratch tab has an empty command; it must come back as a
        // leaf (with its cwd), not get misread as a group.
        let home = Path::new("/home/u");
        let mut t = parse_workspace("workspaces\n\tScratch\n", home);
        let scratch = group(&t, "Scratch");
        t.push(
            Some(scratch),
            "Scratch 1".into(),
            Kind::Leaf {
                workdir: PathBuf::from("/home/u/work"),
                command: String::new(),
            },
            true,
        );
        let reloaded = parse_workspace(&serialize_workspace(&t, home), home);
        let s = group(&reloaded, "Scratch");
        let leaves = reloaded.leaves(s);
        assert_eq!(leaves.len(), 1);
        let (wd, cmd) = reloaded.leaf_spec(leaves[0]).unwrap();
        assert_eq!(reloaded.nodes[leaves[0]].name, "Scratch 1");
        assert_eq!(wd, Path::new("/home/u/work"));
        assert_eq!(cmd, "");
    }

    #[test]
    fn new_workspace_has_project_session_in_cwd_plus_scratch() {
        // A brand-new (missing/empty) workspace opens with a `Project` group
        // holding one session whose workdir is the current directory, plus
        // the Scratch/Transient areas.
        let t = parse_workspace(&default_workspace_text(), Path::new("/home/u"));
        let is_group = |name| {
            t.nodes
                .iter()
                .any(|n| n.name == name && matches!(n.kind, Kind::Group))
        };
        assert!(is_group("Project"));
        assert!(is_group("Scratch") && is_group("Transient"));

        let project = group(&t, "Project");
        let leaves = t.leaves(project);
        assert_eq!(leaves.len(), 1, "one starter session");
        let (wd, cmd) = t.leaf_spec(leaves[0]).unwrap();
        assert_eq!(wd, std::env::current_dir().unwrap());
        assert_eq!(cmd, "", "just a shell, no default command");
    }

    #[test]
    fn embedded_font_loads_and_rasterizes() {
        // The primary face is baked into the binary, so this must succeed on a
        // bare machine with no fontconfig and no system fonts — the macOS fix.
        let mut r = Renderer::new();
        assert!(r.cell_w >= 1 && r.cell_h >= 1, "cell metrics must be positive");
        let g = r.glyph('M', FontStyle::Regular, FONT_PX);
        assert!(g.w > 0 && g.h > 0, "a basic glyph must rasterize");
        // Bold and italic faces are distinct from regular (real attribute
        // rendering, not faux-styling): a wide glyph rasterizes in each.
        for s in [FontStyle::Bold, FontStyle::Italic, FontStyle::BoldItalic] {
            assert!(r.glyph('W', s, FONT_PX).w > 0, "styled face must rasterize");
        }
    }

    #[test]
    fn shortcuts_map_per_platform() {
        let c = Key::Character("c".into());
        let dot = Key::Character(".".into());
        let cmd = ModifiersState::SUPER;
        let ctrl_shift = ModifiersState::CONTROL | ModifiersState::SHIFT;

        // A bare keystroke is never a shortcut on any platform (it goes to the
        // PTY).
        assert_eq!(match_shortcut(ModifiersState::empty(), &c), None);

        if cfg!(target_os = "macos") {
            // ⌘ is the mac modifier; Ctrl+Shift is not.
            assert_eq!(match_shortcut(cmd, &c), Some(Shortcut::Copy));
            assert_eq!(match_shortcut(cmd, &dot), Some(Shortcut::Stop));
            assert_eq!(match_shortcut(ctrl_shift, &c), None);
        } else {
            // Ctrl+Shift is the modifier elsewhere; ⌘ doesn't exist.
            assert_eq!(match_shortcut(ctrl_shift, &c), Some(Shortcut::Copy));
            assert_eq!(match_shortcut(cmd, &c), None);
        }
    }
}
