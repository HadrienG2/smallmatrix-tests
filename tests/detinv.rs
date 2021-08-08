//! Tests of inverse and determinant-like computations
//!
//! Due to their N! combinatorics, these computations cannot be tested using a
//! debug build at higher matrix dimensions, a release build is needed for that.
//! So they get their own codegen, and thus their own compilation unit.

mod common;

use self::common::{assert_close_matrix, assert_close_scalar, assert_panics};
use paste::paste;
use quickcheck::TestResult;
use quickcheck_macros::quickcheck;
use simd_tests::{ColVector, Matrix, Scalar, SquareMatrix};

fn test_det_identity<const DIM: usize>() {
    assert_close_scalar(1.0, SquareMatrix::<DIM>::identity().det(), 1.0);
}

fn test_det_multilinear<const DIM: usize>(
    mat: SquareMatrix<DIM>,
    vec: ColVector<DIM>,
    coef: Scalar,
    idx: usize,
) {
    let idx = idx % DIM;
    let first_cols = mat.into_iter().take(idx);
    let mid_col = mat.col(idx);
    let last_cols = mat.into_iter().skip(idx + 1);
    let mat2 = Matrix::from_iter(
        first_cols
            .clone()
            .chain(core::iter::once(vec))
            .chain(last_cols.clone()),
    );
    let mat3 = Matrix::from_iter(
        first_cols
            .chain(core::iter::once(mid_col + coef * vec))
            .chain(last_cols),
    );
    assert_close_scalar(
        mat.det() + coef * mat2.det(),
        mat3.det(),
        mat.norm().powi(DIM as _) + coef * mat2.norm().powi(DIM as _),
    )
}

fn test_det_alternating<const DIM: usize>(
    mat: SquareMatrix<DIM>,
    idx1: usize,
    idx2: usize,
) -> TestResult {
    if DIM < 2 {
        return TestResult::passed();
    }

    let idx1 = idx1 % DIM;
    let idx2 = idx2 % DIM;
    if idx1 == idx2 {
        return TestResult::discard();
    }

    let mat2 = SquareMatrix::from_iter(
        mat.into_iter()
            .take(idx1)
            .chain(core::iter::once(mat.col(idx2)))
            .chain(mat.into_iter().skip(idx1 + 1)),
    );
    assert_close_scalar(0.0, mat2.det(), mat2.norm().powi(DIM as _));
    TestResult::passed()
}

fn test_inverse<const DIM: usize>(mat: SquareMatrix<DIM>) -> TestResult {
    let det = mat.det();
    if !det.is_finite() {
        TestResult::discard()
    } else if det == 0.0 {
        assert_panics(move || mat.inverse());
        TestResult::passed()
    } else if det.abs() <= Scalar::EPSILON {
        TestResult::discard()
    } else {
        let inverse = mat.inverse();
        assert_close_matrix(
            SquareMatrix::<DIM>::identity(),
            mat * inverse,
            mat.norm() * inverse.norm(),
        );
        TestResult::passed()
    }
}

macro_rules! generate_tests {
    () => {
        generate_tests!(1, 2, 3);
    };
    ($($dim:literal),*) => {
        $(
            paste! {
                #[quickcheck]
                fn [< det $dim _identity >]() {
                    test_det_identity::<$dim>();
                }

                #[quickcheck]
                fn [< det $dim _multilinear >](
                    mat: SquareMatrix<$dim>,
                    vec: ColVector<$dim>,
                    coef: Scalar,
                    idx: usize,
                ) {
                    test_det_multilinear::<$dim>(mat, vec, coef, idx)
                }

                #[quickcheck]
                fn [< det $dim _alternating >](
                    mat: SquareMatrix<$dim>,
                    idx1: usize,
                    idx2: usize,
                ) -> TestResult {
                    test_det_alternating::<$dim>(mat, idx1, idx2)
                }

                #[quickcheck]
                fn [< inverse $dim >](mat: SquareMatrix<$dim>) -> TestResult {
                    test_inverse::<$dim>(mat)
                }
            }
        )*
    }
}
generate_tests!();
