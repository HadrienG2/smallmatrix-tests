//! Tests of matrix multiplication-like computations
//!
//! Testing matrix multiplication for N different vector space dimensions has
//! O(N^3) compile-time combinatorics. Therefore, it gets its own compilation
//! unit and its own codegen units.

mod common;

use self::common::*;
use num_traits::One;
use paste::paste;
use quickcheck::TestResult;
use quickcheck_macros::quickcheck;
use simd_tests::{Matrix, SquareMatrix};

fn test_matmul<const LEFT_ROWS: usize, const LEFT_COLS: usize, const RIGHT_COLS: usize>(
    lhs: Matrix<LEFT_ROWS, LEFT_COLS>,
    rhs: Matrix<LEFT_COLS, RIGHT_COLS>,
) {
    let expected = Matrix::<LEFT_ROWS, RIGHT_COLS>::from_col_major_elems(
        (0..LEFT_ROWS * RIGHT_COLS)
            .map(|idx| (idx / LEFT_ROWS, idx % LEFT_ROWS))
            .map(|(col, row)| (lhs.row(row) * rhs.col(col)).into()),
    );
    assert_close_matrix(expected, lhs * rhs, expected.norm());
}

fn test_matmul_assign<const LEFT_ROWS: usize, const LEFT_COLS: usize>(
    mut lhs: Matrix<LEFT_ROWS, LEFT_COLS>,
    rhs: SquareMatrix<LEFT_COLS>,
) {
    let expected = lhs * rhs;
    lhs *= rhs;
    assert_close_matrix(expected, lhs, expected.norm());
}

fn test_product<const DIM: usize>(mats: Vec<SquareMatrix<DIM>>) {
    let expected = mats
        .iter()
        .copied()
        .fold(SquareMatrix::<DIM>::one(), |acc, mat| acc * mat);
    assert_close_matrix(expected, mats.into_iter().product(), expected.norm());
}

pub fn test_pow<const DIM: usize>(mut lhs: SquareMatrix<DIM>, rhs: u8) {
    lhs /= lhs.norm();
    let expected = std::iter::repeat(lhs).take(rhs as _).product();
    assert_close_matrix(expected, lhs.pow(rhs), expected.norm());
}

macro_rules! generate_tests {
    () => {
        generate_tests!(1, 2, 3, 4, 5);
    };
    ($($dim:literal),*) => {
        $(
            paste! {
                #[quickcheck]
                fn [< product $dim x $dim >](mats: Vec<SquareMatrix<$dim>>) {
                    test_product::<$dim>(mats)
                }

                #[quickcheck]
                fn [< pow $dim x $dim >](lhs: SquareMatrix<$dim>, rhs: u8) {
                    test_pow::<$dim>(lhs, rhs)
                }
            }

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
                fn [< matmul_assign_ $dim1 x $dim2 >](lhs: Matrix<$dim1, $dim2>, rhs: Matrix<$dim2, $dim2>) {
                    test_matmul_assign::<$dim1, $dim2>(lhs, rhs)
                }
            }

            generate_tests!(
                ($dim1, $dim2, 1),
                ($dim1, $dim2, 2),
                ($dim1, $dim2, 3),
                ($dim1, $dim2, 4),
                ($dim1, $dim2, 5)
            );
        )*
    };
    ($(($dim1:literal, $dim2:literal, $dim3:literal)),*) => {
        $(
            paste! {
                #[quickcheck]
                fn [< matmul_ $dim1 x $dim2 _by_ $dim2 x $dim3 >](lhs: Matrix<$dim1, $dim2>, rhs: Matrix<$dim2, $dim3>) {
                    test_matmul::<$dim1, $dim2, $dim3>(lhs, rhs)
                }
            }
        )*
    }
}
generate_tests!();
