use std::panic::UnwindSafe;

/// Assert that a certain functor panics
pub fn assert_panics<F: FnOnce() -> R + UnwindSafe + 'static, R>(f: F) {
    // Code bloat optimization
    fn polymorphic_impl(f: Box<dyn FnOnce() + UnwindSafe>) {
        assert!(std::panic::catch_unwind(f).is_err())
    }
    polymorphic_impl(Box::new(|| {
        f();
    }))
}
