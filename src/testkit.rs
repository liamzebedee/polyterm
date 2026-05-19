//! Headless integration-test harness.
//!
//! Builds the *real* application state (`State`) with no winit window, drives
//! it through actions (select a tab, toggle the inspector, feed terminal
//! output, change the display scale), renders the genuine UI into an
//! off-screen framebuffer via `State::paint`, and writes PNG screenshots into
//! a fresh per-run temp directory with sequential filenames. Each save prints
//!
//! ```text
//! Screenshot taken: /tmp/termem-shots-<run>/001-<label>.png
//! ```
//!
//! so a reviewer (human or agent) can open the files and see exactly what the
//! UI looks like after each step. Because everything is composed at *logical*
//! size, screenshots are DPI-independent and identical to what macOS would
//! show at the same `scale` (see `State::paint`).

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use alacritty_terminal::sync::FairMutex;
use alacritty_terminal::term::{Config, Term};
use alacritty_terminal::vte::ansi;
use winit::keyboard::ModifiersState;

use crate::{Kind, Listener, NodeId, Renderer, Session, State, Tab, home_dir, parse_workspace};

/// A headless app instance plus a screenshot sink.
pub struct Harness {
    state: State,
    dir: PathBuf,
    seq: usize,
}

impl Harness {
    /// Build a harness from a `workspace`-file spec string, at a default
    /// 1100×720 logical window and scale 1.0.
    pub fn new(workspace: &str) -> Self {
        Self::with_window(workspace, 1100, 720, 1.0)
    }

    /// Build at an explicit *physical* window size and device scale. The UI
    /// lays out at `(w/scale, h/scale)`; pass `scale = 2.0` to reproduce a
    /// macOS Retina display.
    pub fn with_window(workspace: &str, w: usize, h: usize, scale: f64) -> Self {
        let tree = parse_workspace(workspace, &home_dir());
        let selected = tree.first_leaf(tree.root).unwrap_or(tree.root);
        let state = State {
            window: None,
            fb: Vec::new(),
            phys: (w, h),
            scale,
            renderer: Renderer::new(),
            tree,
            sessions: std::collections::HashMap::new(),
            id_of: std::collections::HashMap::new(),
            selected,
            next_id: 0,
            ctx: None,
            clipboard: None,
            mouse: (0.0, 0.0),
            selecting: false,
            last_click: None,
            mods: ModifiersState::empty(),
            inspector: false,
            focus: None,
            caret: 0,
            scroll_acc: 0.0,
        };
        let dir = std::env::temp_dir().join(format!(
            "termem-shots-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).expect("create screenshot dir");
        Harness {
            state,
            dir,
            seq: 0,
        }
    }

    /// The per-run screenshot directory (printed once for convenience).
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    // --- actions -----------------------------------------------------------

    /// Select a node by name, or by a `Group/Sub/Leaf` path. Ancestors are
    /// expanded so the row is visible, just like a real click would leave it.
    pub fn select(&mut self, path: &str) -> &mut Self {
        let id = self
            .find(path)
            .unwrap_or_else(|| panic!("no node {path:?}; have: {:?}", self.node_names()));
        // Expand the node (if a group) and every ancestor.
        self.state.tree.nodes[id].expanded = true;
        let mut cur = self.state.tree.nodes[id].parent;
        while let Some(p) = cur {
            self.state.tree.nodes[p].expanded = true;
            cur = self.state.tree.nodes[p].parent;
        }
        self.state.selected = id;
        self
    }

    /// Show or hide the right inspector ("info") pane.
    pub fn inspector(&mut self, on: bool) -> &mut Self {
        self.state.inspector = on;
        self
    }

    /// Attach a headless terminal to the named leaf and feed it `text`
    /// (UTF-8, ANSI escapes honoured) so screenshots show real rendered
    /// output. Deterministic: no PTY, no shell, no threads.
    pub fn feed(&mut self, path: &str, text: &str) -> &mut Self {
        let id = self
            .find(path)
            .unwrap_or_else(|| panic!("no node {path:?}"));
        assert!(self.state.tree.is_leaf(id), "{path:?} is not a leaf");
        let (lw, lh) = self.state.logical_size();
        let size = self.state.grid_size(lw, lh);
        let title = self.state.tree.nodes[id].name.clone();
        let term = Arc::new(FairMutex::new(Term::new(
            Config::default(),
            &size,
            Listener::Null,
        )));
        {
            let mut parser = ansi::Processor::<ansi::StdSyncHandler>::new();
            let mut guard = term.lock();
            parser.advance(&mut *guard, text.as_bytes());
        }
        self.state.sessions.insert(
            id,
            Session {
                tab: Tab {
                    term,
                    pty_tx: None,
                    size,
                    title,
                },
                shell_pid: 0,
            },
        );
        self
    }

    /// Change the device scale (e.g. 2.0 to emulate macOS Retina) and reflow.
    pub fn set_scale(&mut self, scale: f64) -> &mut Self {
        self.state.scale = scale.max(1.0);
        self
    }

    /// Resize the (physical) window.
    pub fn resize(&mut self, w: usize, h: usize) -> &mut Self {
        self.state.phys = (w, h);
        self
    }

    /// Render the current state and write the next sequential PNG. Prints
    /// `Screenshot taken: <path>` and returns the path.
    pub fn screenshot(&mut self, label: &str) -> PathBuf {
        // Reflow injected terminals to the current terminal area so toggling
        // the inspector / resizing before a shot reads correctly.
        let (lw, lh) = self.state.logical_size();
        let size = self.state.grid_size(lw, lh);
        for s in self.state.sessions.values_mut() {
            s.tab.size = size;
            s.tab.term.lock().resize(size);
        }

        self.state.paint();
        let (w, h) = self.state.logical_size();

        self.seq += 1;
        let safe: String = label
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '-' })
            .collect();
        let path = self.dir.join(format!("{:03}-{safe}.png", self.seq));
        write_png(&path, w, h, &self.state.fb).expect("write png");
        println!("Screenshot taken: {}", path.display());
        path
    }

    // --- helpers -----------------------------------------------------------

    fn node_names(&self) -> Vec<&str> {
        self.state
            .tree
            .nodes
            .iter()
            .map(|n| n.name.as_str())
            .collect()
    }

    /// Resolve a node by `Group/Leaf` path, or by bare name (first match in
    /// definition order).
    fn find(&self, path: &str) -> Option<NodeId> {
        let t = &self.state.tree;
        if let Some((_, _)) = path.split_once('/') {
            let mut cur = t.root;
            for seg in path.split('/') {
                cur = *t.nodes[cur]
                    .children
                    .iter()
                    .find(|&&c| t.nodes[c].name == seg)?;
            }
            return Some(cur);
        }
        (0..t.nodes.len())
            .find(|&i| t.nodes[i].name == path && !matches!(t.nodes[i].kind, Kind::Root))
    }
}

// ===========================================================================
// Minimal dependency-free PNG writer (8-bit truecolour, zlib *stored* blocks).
// ===========================================================================

fn write_png(path: &Path, w: usize, h: usize, fb: &[u32]) -> std::io::Result<()> {
    assert_eq!(fb.len(), w * h, "framebuffer size mismatch");
    let mut png = Vec::with_capacity(w * h * 3 + 1024);
    png.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);

    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&(w as u32).to_be_bytes());
    ihdr.extend_from_slice(&(h as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8bpc, colour type 2 (RGB)
    chunk(&mut png, b"IHDR", &ihdr);

    // Raw image: each row prefixed with filter byte 0 (no filtering).
    let mut raw = Vec::with_capacity(h * (w * 3 + 1));
    for y in 0..h {
        raw.push(0);
        for x in 0..w {
            let p = fb[y * w + x];
            raw.push((p >> 16) as u8);
            raw.push((p >> 8) as u8);
            raw.push(p as u8);
        }
    }
    chunk(&mut png, b"IDAT", &zlib_store(&raw));
    chunk(&mut png, b"IEND", &[]);
    std::fs::write(path, png)
}

fn chunk(out: &mut Vec<u8>, tag: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(tag);
    out.extend_from_slice(data);
    let mut crc = Crc::new();
    crc.update(tag);
    crc.update(data);
    out.extend_from_slice(&crc.finish().to_be_bytes());
}

/// Wrap `data` as a zlib stream using uncompressed DEFLATE blocks — no
/// compression code, no dependency, still a valid PNG.
fn zlib_store(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01]; // zlib header, no preset dict
    let mut i = 0;
    while {
        let n = (data.len() - i).min(0xffff);
        let last = i + n == data.len();
        out.push(if last { 1 } else { 0 }); // BFINAL, BTYPE=00 (stored)
        out.extend_from_slice(&(n as u16).to_le_bytes());
        out.extend_from_slice(&(!(n as u16)).to_le_bytes());
        out.extend_from_slice(&data[i..i + n]);
        i += n;
        i < data.len()
    } {}
    if data.is_empty() {
        out.extend_from_slice(&[1, 0, 0, 0xff, 0xff]);
    }
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

struct Crc(u32);

impl Crc {
    fn new() -> Self {
        Crc(0xffff_ffff)
    }
    fn update(&mut self, data: &[u8]) {
        for &byte in data {
            let mut c = (self.0 ^ byte as u32) & 0xff;
            for _ in 0..8 {
                c = if c & 1 != 0 {
                    0xedb8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
            }
            self.0 = c ^ (self.0 >> 8);
        }
    }
    fn finish(self) -> u32 {
        self.0 ^ 0xffff_ffff
    }
}
