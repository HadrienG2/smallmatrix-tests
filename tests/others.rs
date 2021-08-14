#![feature(const_generics, const_evaluatable_checked)]

mod common;

use self::common::*;
use num_traits::{One, Zero};
use paste::paste;
use quickcheck::TestResult;
use quickcheck_macros::quickcheck;
use smallmatrix_tests::{Matrix, Scalar, SquareMatrix, Vector, X, Y, Z};

fn test_clone<const ROWS: usize, const COLS: usize>(mat: Matrix<ROWS, COLS>) {
    let expected = Matrix::<ROWS, COLS>::from_col_major_elems(mat.into_col_major_elems());
    assert_bits_eq(expected, mat.clone());
}

// NOTE: PartialEq has been split into the separate "ops" test

#[quickcheck]
fn scalar_1x1_conv(x: Scalar) {
    let expected = Matrix::<1, 1>::from_col_major_elems(std::iter::once(x));
    assert_bits_eq(expected, Matrix::<1, 1>::from(x));
    assert_eq!(x.to_bits(), Scalar::from(expected).to_bits());
}

fn test_default<const ROWS: usize, const COLS: usize>() {
    let expected = Matrix::<ROWS, COLS>::from_col_major_elems(
        std::iter::repeat(Scalar::default()).take(ROWS * COLS),
    );
    assert_bits_eq(expected, Matrix::<ROWS, COLS>::default());
}

fn test_zero<const ROWS: usize, const COLS: usize>() {
    let expected =
        Matrix::<ROWS, COLS>::from_col_major_elems(std::iter::repeat(0.0).take(ROWS * COLS));
    assert_bits_eq(expected, Matrix::<ROWS, COLS>::zero());
}

fn test_unit<const DIM: usize>(idx: usize) {
    if idx < DIM {
        // Assert return type is right
        let unit: Vector<DIM> = Vector::<DIM>::unit(idx);
        for (idx2, elem) in unit.into_col_major_elems().enumerate() {
            assert_eq!(elem, (idx2 == idx) as u8 as Scalar);
        }
    } else {
        assert_panics(move || Vector::<DIM>::unit(idx))
    }
}

fn test_identity<const DIM: usize>() {
    // Assert return type is right
    let identity: SquareMatrix<DIM> = SquareMatrix::<DIM>::identity();
    for (col, col_vec) in identity.into_iter().enumerate() {
        for (row, elem) in col_vec.into_col_major_elems().enumerate() {
            assert_eq!(elem, (col == row) as u8 as Scalar);
        }
    }
}

fn test_one<const DIM: usize>() {
    assert_bits_eq(SquareMatrix::<DIM>::identity(), SquareMatrix::<DIM>::one());
}

// NOTE: Iterators have been split into the separate "iter" test
// NOTE: Matrix concatenation has been split into the separate "concat" test
// NOTE: Index operator is tested as a unit test because the test needs access
//       to the implementation, to avoid being a synonym of col-major iter tests
// NOTE: Matrix block extraction has been split into the separate "blocks" test

fn test_transpose<const ROWS: usize, const COLS: usize>(mat: Matrix<ROWS, COLS>) {
    // Assert return type is right
    let tr: Matrix<COLS, ROWS> = mat.transpose();
    for src_row in 0..ROWS {
        for src_col in 0..COLS {
            assert_eq!(
                mat[(src_row, src_col)].to_bits(),
                tr[(src_col, src_row)].to_bits()
            );
        }
    }
}

// NOTE: Arithmetic operators have been split out into the separate "ops" test
// NOTE: Matrix multiplication-like operations have been split into the separate
//       "matmul" test

fn test_dot<const ROWS: usize, const COLS: usize>(
    lhs: Matrix<ROWS, COLS>,
    rhs: Matrix<ROWS, COLS>,
) {
    let expected = lhs
        .into_col_major_elems()
        .zip(rhs.into_col_major_elems())
        .map(|(x, y)| x * y)
        .sum::<Scalar>();
    assert_close_scalar(expected, lhs.dot(rhs), lhs.norm() * rhs.norm());
}

fn test_squared_norm<const ROWS: usize, const COLS: usize>(mat: Matrix<ROWS, COLS>) {
    let expected = mat.dot(mat);
    assert_close_scalar(expected, mat.squared_norm(), mat.squared_norm());
}

fn test_norm<const ROWS: usize, const COLS: usize>(mat: Matrix<ROWS, COLS>) {
    let expected = mat.squared_norm().sqrt();
    assert_close_scalar(expected, mat.norm(), mat.norm());
}

#[quickcheck]
fn cross(lhs: Vector<3>, rhs: Vector<3>) {
    let expected = Vector::<3>::from_col_major_elems(
        [
            lhs[Y] * rhs[Z] - lhs[Z] * rhs[Y],
            lhs[Z] * rhs[X] - lhs[X] * rhs[Z],
            lhs[X] * rhs[Y] - lhs[Y] * rhs[X],
        ]
        .into_iter(),
    );
    assert_close_matrix(expected, lhs.cross(rhs), lhs.norm() * rhs.norm());
}

fn test_trace<const DIM: usize>(mat: SquareMatrix<DIM>) {
    let expected = (0..DIM).map(|idx| mat[(idx, idx)]).sum::<Scalar>();
    assert_close_scalar(expected, mat.trace(), mat.norm());
}

// NOTE: Matrix determinant, inverse, minor and cofactor have been split into
//       the separate "detinv" test

macro_rules! generate_tests {
    () => {
        generate_tests!(1, 2, 3, 4, 5, 6, 7, 8);
    };
    ($($dim:literal),*) => {
        $(
            paste! {
                #[test]
                fn [< unit $dim _in_range >]() {
                    for idx in 0..$dim {
                        test_unit::<$dim>(idx);
                    }
                }

                #[quickcheck]
                fn [< unit $dim _out_of_range >](idx: usize) -> TestResult {
                    if idx < $dim {
                        return TestResult::discard();
                    }
                    test_unit::<$dim>(idx);
                    TestResult::passed()
                }

                #[test]
                fn [< identity $dim >]() {
                    test_identity::<$dim>();
                }

                #[test]
                fn [< one $dim >]() {
                    test_one::<$dim>();
                }

                #[quickcheck]
                fn [< trace $dim x $dim >](mat: SquareMatrix<$dim>) {
                    test_trace::<$dim>(mat);
                }
            }

            generate_tests!(
                ($dim, 1),
                ($dim, 2),
                ($dim, 3),
                ($dim, 4),
                ($dim, 5),
                ($dim, 6),
                ($dim, 7),
                ($dim, 8)
            );
        )*
    };
    ($(($dim1:literal, $dim2:literal)),*) => {
        $(
            paste! {
                #[quickcheck]
                fn [< clone $dim1 x $dim2 >](mat: Matrix<$dim1, $dim2>) {
                    test_clone::<$dim1, $dim2>(mat);
                }

                #[test]
                fn [< default $dim1 x $dim2 >]() {
                    test_default::<$dim1, $dim2>();
                }

                #[test]
                fn [< zero $dim1 x $dim2 >]() {
                    test_zero::<$dim1, $dim2>();
                }

                #[quickcheck]
                fn [< transpose $dim1 x $dim2 >](mat: Matrix<$dim1, $dim2>) {
                    test_transpose::<$dim1, $dim2>(mat)
                }

                #[quickcheck]
                fn [< dot $dim1 x $dim2 >](lhs: Matrix<$dim1, $dim2>, rhs: Matrix<$dim1, $dim2>) {
                    test_dot::<$dim1, $dim2>(lhs, rhs);
                }

                #[quickcheck]
                fn [< squared_norm $dim1 x $dim2 >](mat: Matrix<$dim1, $dim2>) {
                    test_squared_norm::<$dim1, $dim2>(mat);
                }

                #[quickcheck]
                fn [< norm $dim1 x $dim2 >](mat: Matrix<$dim1, $dim2>) {
                    test_norm::<$dim1, $dim2>(mat);
                }
            }
        )*
    }
}
generate_tests!();
