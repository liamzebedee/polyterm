//! Integration "looks like" test. Run it and open the printed PNGs:
//!
//!     cargo test --test screenshots -- --nocapture
//!
//! Each `screenshot(..)` prints `Screenshot taken: <path>`.

use termem_demo::testkit::Harness;

/// A workspace spec in the on-disk format (tab-indented; name / dir / command
/// tab-separated). `parse_workspace` also appends Scratch + Transient.
const WORKSPACE: &str = "workspaces
\tApps
\t\t\"qwen\"\t~/.apps\t./run-llama.sh
\tDiabetes
\t\t\"Meal planner\"\t~/diabetes/mealplanner\t\"bun run main.ts\"
\t\t\"Sugar tracker\"\t~/diabetes/sugar\t./run.sh
\tStudy
\t\t\"Papers\"\t~/papers\tbash
";

const SAMPLE: &str = concat!(
    "\x1b[1;32mliam@rand\x1b[0m:\x1b[1;34m~/diabetes/sugar\x1b[0m$ ls --color\r\n",
    "\x1b[1;34mdata\x1b[0m  run.sh  \x1b[1;32manalyze\x1b[0m  README.md\r\n",
    "\x1b[1;32mliam@rand\x1b[0m:\x1b[1;34m~/diabetes/sugar\x1b[0m$ cargo test\r\n",
    "\x1b[32m   Compiling\x1b[0m sugar v0.1.0\r\n",
    "\x1b[32m    Finished\x1b[0m test profile in 1.21s\r\n",
    "\x1b[31merror\x1b[0m: something went \x1b[1;31mwrong\x1b[0m on line 42\r\n",
    "\x1b[1;32mliam@rand\x1b[0m:\x1b[1;34m~/diabetes/sugar\x1b[0m$ \u{2588}\r\n",
);

#[test]
fn ui_walkthrough() {
    let mut h = Harness::new(WORKSPACE);
    eprintln!("screenshot dir: {}", h.dir().display());

    // 1. A freshly-opened leaf with no live session.
    h.select("Study/Papers");
    h.screenshot("empty-leaf");

    // 2. Real (deterministic) terminal output, with ANSI colour.
    h.select("Diabetes/Sugar tracker")
        .feed("Sugar tracker", SAMPLE);
    h.screenshot("terminal-output");

    // 3. Inspector ("info") pane open on that leaf.
    h.inspector(true);
    h.screenshot("inspector-open");

    // 4. A group selected (fans out / shows first session under it).
    h.inspector(false).select("Diabetes");
    h.screenshot("group-selected");
}

/// The macOS fix: the UI is laid out in logical pixels, so a 2× Retina
/// display must produce a pixel-identical frame to a 1× display of the same
/// logical size — only the final upscale differs.
#[test]
fn retina_parity() {
    let mut a = Harness::with_window(WORKSPACE, 1100, 720, 1.0);
    a.select("Study/Papers").feed("Papers", SAMPLE);
    let lo = a.screenshot("scale1x");

    let mut b = Harness::with_window(WORKSPACE, 2200, 1440, 2.0);
    b.select("Study/Papers").feed("Papers", SAMPLE);
    let hi = b.screenshot("scale2x");

    let (da, db) = (std::fs::read(&lo).unwrap(), std::fs::read(&hi).unwrap());
    assert_eq!(
        da, db,
        "logical frame must be identical at 1x and 2x (macOS parity)"
    );
}
