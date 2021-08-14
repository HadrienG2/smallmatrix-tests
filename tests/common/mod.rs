use more_asserts::*;
use smallmatrix_tests::{Matrix, Scalar};
use std::panic::UnwindSafe;

/// Assert that a certain functor panics
#[allow(unused)]
pub fn assert_panics<F: FnOnce() -> R + UnwindSafe + 'static, R>(f: F) {
    // Code bloat optimization
    fn polymorphic_impl(f: Box<dyn FnOnce() + UnwindSafe>) {
        assert!(std::panic::catch_unwind(f).is_err())
    }
    polymorphic_impl(Box::new(|| {
        f();
    }))
}

// Function signature asserts that the matrix type is the same
#[allow(unused)]
pub fn assert_bits_eq<const ROWS: usize, const COLS: usize>(
    expected: Matrix<ROWS, COLS>,
    result: Matrix<ROWS, COLS>,
) {
    for (expected, result) in expected
        .into_col_major_elems()
        .zip(result.into_col_major_elems())
    {
        assert_eq!(expected.to_bits(), result.to_bits());
    }
}

#[allow(unused)]
pub fn assert_close_scalar(expected: Scalar, result: Scalar, magnitude: Scalar) {
    // Ignore NaN and inf results, they are sensitive to the order of operations
    if !expected.is_finite() || !magnitude.is_finite() {
        return;
    } else if expected.abs() <= Scalar::EPSILON * magnitude.abs() {
        assert_le!((result - expected).abs(), Scalar::EPSILON * magnitude.abs());
    } else {
        assert_le!((result - expected).abs(), Scalar::EPSILON * expected.abs());
    }
}

// Function signature asserts that the matrix type is the same
#[allow(unused)]
pub fn assert_close_matrix<const ROWS: usize, const COLS: usize>(
    expected: Matrix<ROWS, COLS>,
    result: Matrix<ROWS, COLS>,
    magnitude: Scalar,
) {
    for (expected, result) in expected
        .into_col_major_elems()
        .zip(result.into_col_major_elems())
    {
        assert_close_scalar(expected, result, magnitude);
    }
}
