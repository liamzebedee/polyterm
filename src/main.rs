//! Thin binary shim. All logic lives in the library crate (`termem_demo`)
//! so the integration harness in `testkit` can drive it headlessly.

fn main() {
    termem_demo::run();
}
