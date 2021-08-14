#![feature(const_generics, const_evaluatable_checked)]

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
use smallmatrix_tests::{ColVector, ConstCheck, Matrix, Scalar, SquareMatrix, True};

fn test_minor<const DIM: usize>(mat: SquareMatrix<DIM>, row: usize, col: usize)
where
    [(); DIM - 1]: ,
    ConstCheck<{ DIM >= 2 }>: True,
{
    if row >= DIM || col >= DIM {
        assert_panics(move || mat.minor(row, col))
    } else {
        let sub_mat = SquareMatrix::<{ DIM - 1 }>::from_iter(
            mat.into_iter()
                .take(col)
                .chain(mat.into_iter().skip(col + 1))
                .map(|col| {
                    ColVector::<{ DIM - 1 }>::from_col_major_elems(
                        col.into_col_major_elems()
                            .take(row)
                            .chain(col.into_col_major_elems().skip(row + 1)),
                    )
                }),
        );
        assert_close_scalar(sub_mat.det(), mat.minor(row, col), mat.norm());
    }
}

fn test_cofactor<const DIM: usize>(mat: SquareMatrix<DIM>, row: usize, col: usize)
where
    [(); DIM - 1]: ,
    ConstCheck<{ DIM >= 2 }>: True,
{
    if row >= DIM || col >= DIM {
        assert_panics(move || mat.cofactor(row, col))
    } else {
        assert_close_scalar(
            (-1.0 as Scalar).powi((row + col) as _) * mat.minor(row, col),
            mat.cofactor(row, col),
            mat.norm(),
        );
    }
}

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
) -> TestResult
where
    ConstCheck<{ DIM >= 2 }>: True,
{
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

fn test_inverse<const DIM: usize>(mut mat: SquareMatrix<DIM>) -> TestResult {
    // Matrix inversion is very sensitive to conditioning, so we only test it on
    // normalized matrices. This also has the advantage of propagating any inner
    // NaNs all over the matrix, so it ends up in the determinant.
    mat /= mat.norm();
    let det = mat.det();
    if !det.is_finite() {
        TestResult::discard()
    } else if det == 0.0 {
        assert_panics(move || mat.inverse());
        TestResult::passed()
    } else if det.abs() < Scalar::EPSILON {
        TestResult::discard()
    } else {
        assert_close_matrix(SquareMatrix::<DIM>::identity(), mat * mat.inverse(), 1.0);
        TestResult::passed()
    }
}

// Some tests should be able to run for all matrix dimensions...
macro_rules! generate_tests {
    () => {
        generate_tests!(1, 2, 3, 4, 5);
        #[cfg(not(debug_assertions))]
        generate_tests!(6, 7);
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
                fn [< inverse $dim >](mat: SquareMatrix<$dim>) -> TestResult {
                    test_inverse::<$dim>(mat)
                }
            }
        )*
    }
}
generate_tests!();

// ...while others only make sense for matrix dimensions above 2
macro_rules! generate_more_tests {
    () => {
        generate_more_tests!(2, 3, 4, 5);
        #[cfg(not(debug_assertions))]
        generate_more_tests!(6, 7);
    };
    ($($dim:literal),*) => {
        $(
            paste! {
                #[quickcheck]
                fn [< det $dim _alternating >](
                    mat: SquareMatrix<$dim>,
                    idx1: usize,
                    idx2: usize,
                ) -> TestResult {
                    test_det_alternating::<$dim>(mat, idx1, idx2)
                }

                #[quickcheck]
                fn [< minor $dim >](
                    mat: SquareMatrix<$dim>,
                    row: usize,
                    col: usize,
                ) {
                    test_minor::<$dim>(mat, row, col)
                }

                #[quickcheck]
                fn [< cofactor $dim >](
                    mat: SquareMatrix<$dim>,
                    row: usize,
                    col: usize,
                ) {
                    test_cofactor::<$dim>(mat, row, col)
                }
            }
        )*
    }
}
generate_more_tests!();
