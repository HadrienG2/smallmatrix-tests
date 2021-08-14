//! Rust operators tests
//!
//! Split out from other tests to increase build parallelism

mod common;

use self::common::*;
use num_traits::Zero;
use paste::paste;
use quickcheck_macros::quickcheck;
use smallmatrix_tests::{Matrix, Scalar};

fn test_partial_eq<const ROWS: usize, const COLS: usize>(
    mat1: Matrix<ROWS, COLS>,
    mat2: Matrix<ROWS, COLS>,
) {
    let expected = mat1
        .into_col_major_elems()
        .zip(mat2.into_col_major_elems())
        .all(|(x, y)| x == y);
    assert_eq!(expected, mat1 == mat2);
}

fn test_neg<const ROWS: usize, const COLS: usize>(mat: Matrix<ROWS, COLS>) {
    let expected =
        Matrix::<ROWS, COLS>::from_col_major_elems(mat.into_col_major_elems().map(|x| -x));
    assert_bits_eq(expected, -mat);
}

fn test_mul_scalar<const ROWS: usize, const COLS: usize>(mut mat: Matrix<ROWS, COLS>, x: Scalar) {
    let expected =
        Matrix::<ROWS, COLS>::from_col_major_elems(mat.into_col_major_elems().map(|elem| elem * x));
    assert_bits_eq(expected, mat * x);
    assert_bits_eq(expected, x * mat);
    mat *= x;
    assert_bits_eq(expected, mat);
}

fn test_div_scalar<const ROWS: usize, const COLS: usize>(mut lhs: Matrix<ROWS, COLS>, rhs: Scalar) {
    let expected =
        Matrix::<ROWS, COLS>::from_col_major_elems(lhs.into_col_major_elems().map(|lhs| lhs / rhs));
    assert_bits_eq(expected, lhs / rhs);
    lhs /= rhs;
    assert_bits_eq(expected, lhs);
}

fn test_add<const ROWS: usize, const COLS: usize>(
    mut lhs: Matrix<ROWS, COLS>,
    rhs: Matrix<ROWS, COLS>,
) {
    let expected = Matrix::<ROWS, COLS>::from_col_major_elems(
        lhs.into_col_major_elems()
            .zip(rhs.into_col_major_elems())
            .map(|(x, y)| x + y),
    );
    assert_bits_eq(expected, lhs + rhs);
    lhs += rhs;
    assert_bits_eq(expected, lhs);
}

fn test_sum<const ROWS: usize, const COLS: usize>(mats: Vec<Matrix<ROWS, COLS>>) {
    let expected = mats
        .iter()
        .copied()
        .fold(Matrix::<ROWS, COLS>::zero(), |acc, mat| acc + mat);
    assert_close_matrix(expected, mats.into_iter().sum(), expected.norm());
}

fn test_sub<const ROWS: usize, const COLS: usize>(
    mut lhs: Matrix<ROWS, COLS>,
    rhs: Matrix<ROWS, COLS>,
) {
    let expected = Matrix::<ROWS, COLS>::from_col_major_elems(
        lhs.into_col_major_elems()
            .zip(rhs.into_col_major_elems())
            .map(|(x, y)| x - y),
    );
    assert_bits_eq(expected, lhs - rhs);
    lhs -= rhs;
    assert_bits_eq(expected, lhs);
}

macro_rules! generate_tests {
    () => {
        generate_tests!(1, 2, 3, 4, 5);
    };
    ($($dim:literal),*) => {
        $(
            generate_tests!(
                ($dim, 1),
                ($dim, 2),
                ($dim, 3),
                ($dim, 4),
                ($dim, 5)
            );
        )*
    };
    ($(($dim1:literal, $dim2:literal)),*) => {
        $(
            paste! {
                #[quickcheck]
                fn [< partial_eq $dim1 x $dim2 >](mat1: Matrix<$dim1, $dim2>, mat2: Matrix<$dim1, $dim2>) {
                    test_partial_eq::<$dim1, $dim2>(mat1, mat2);
                }

                #[quickcheck]
                fn [< neg $dim1 x $dim2 >](mat: Matrix<$dim1, $dim2>) {
                    test_neg::<$dim1, $dim2>(mat)
                }

                #[quickcheck]
                fn [< mul_scalar $dim1 x $dim2 >](mat: Matrix<$dim1, $dim2>, x: Scalar) {
                    test_mul_scalar::<$dim1, $dim2>(mat, x);
                }

                #[quickcheck]
                fn [< div_scalar $dim1 x $dim2 >](lhs: Matrix<$dim1, $dim2>, rhs: Scalar) {
                    test_div_scalar::<$dim1, $dim2>(lhs, rhs);
                }

                #[quickcheck]
                fn [< add $dim1 x $dim2 >](lhs: Matrix<$dim1, $dim2>, rhs: Matrix<$dim1, $dim2>) {
                    test_add::<$dim1, $dim2>(lhs, rhs);
                }

                #[quickcheck]
                fn [< sum $dim1 x $dim2 >](mats: Vec<Matrix<$dim1, $dim2>>) {
                    test_sum::<$dim1, $dim2>(mats);
                }

                #[quickcheck]
                fn [< sub $dim1 x $dim2 >](lhs: Matrix<$dim1, $dim2>, rhs: Matrix<$dim1, $dim2>) {
                    test_sub::<$dim1, $dim2>(lhs, rhs);
                }
            }
        )*
    }
}
generate_tests!();
