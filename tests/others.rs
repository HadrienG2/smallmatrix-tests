#![feature(const_generics, const_evaluatable_checked)]

mod common;

use self::common::assert_panics;
use more_asserts::*;
use paste::paste;
use quickcheck::TestResult;
use quickcheck_macros::quickcheck;
use simd_tests::{Matrix, Scalar, SquareMatrix, Vector, X, Y, Z};

fn assert_close(expected: Scalar, result: Scalar) {
    if expected.is_nan() {
        assert_eq!(result.to_bits(), expected.to_bits());
    } else if expected.is_infinite() {
        assert_eq!(result, expected);
    } else {
        assert_le!((result - expected).abs(), Scalar::EPSILON * expected.abs());
    }
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

fn test_from_col_major_elems<const ROWS: usize, const COLS: usize>(elems: Vec<Scalar>) {
    if elems.len() == ROWS * COLS {
        // Assert return type is right
        let mat: Matrix<ROWS, COLS> =
            Matrix::<ROWS, COLS>::from_col_major_elems(elems.iter().copied());
        for (src, dest) in elems.into_iter().zip(mat.into_col_major_elems()) {
            assert_eq!(src.to_bits(), dest.to_bits());
        }
    } else {
        assert_panics(move || Matrix::<ROWS, COLS>::from_col_major_elems(elems.into_iter()));
    }
}

fn test_into_col_major_elems<const ROWS: usize, const COLS: usize>(mat: Matrix<ROWS, COLS>) {
    assert_eq!(mat.into_col_major_elems().count(), ROWS * COLS);
    for (idx, dest) in mat.into_col_major_elems().enumerate() {
        let (col, row) = (idx / ROWS, idx % ROWS);
        let src = mat[(row, col)];
        assert_eq!(src.to_bits(), dest.to_bits());
    }
}

fn test_col_major_elems<const ROWS: usize, const COLS: usize>(mat: Matrix<ROWS, COLS>) {
    assert_eq!(mat.col_major_elems().count(), ROWS * COLS);
    for (idx, dest_ref) in mat.col_major_elems().enumerate() {
        let (col, row) = (idx / ROWS, idx % ROWS);
        assert_eq!(&mat[(row, col)] as *const Scalar, dest_ref as *const Scalar);
    }
}

fn test_col_major_elems_mut<const ROWS: usize, const COLS: usize>(mut mat: Matrix<ROWS, COLS>) {
    assert_eq!(mat.col_major_elems_mut().count(), ROWS * COLS);
    let ptrs = mat
        .col_major_elems_mut()
        .map(|refmut| refmut as *mut Scalar)
        .collect::<Vec<_>>();
    for (idx, ptr) in ptrs.into_iter().enumerate() {
        let (col, row) = (idx / ROWS, idx % ROWS);
        assert_eq!(&mut mat[(row, col)] as *mut Scalar, ptr);
    }
}

// NOTE: Matrix concatenation has been split into the separate "concat" test
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

fn test_dot<const DIM: usize>(lhs: Vector<DIM>, rhs: Vector<DIM>) {
    // Assert return type is right
    let result: Scalar = lhs.dot(rhs);
    let expected = lhs
        .into_col_major_elems()
        .zip(rhs.into_col_major_elems())
        .map(|(x, y)| x * y)
        .sum::<Scalar>();
    assert_close(expected, result);
}

fn test_norm<const DIM: usize>(vec: Vector<DIM>) {
    // Assert return type is right
    let result: Scalar = vec.norm();
    let expected = vec.dot(vec).sqrt();
    assert_close(expected, result);
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
    // Assert return type is right
    let result: Vector<3> = lhs.cross(rhs);
    for (expected, result) in expected
        .into_col_major_elems()
        .zip(result.into_col_major_elems())
    {
        assert_close(expected, result);
    }
}

fn test_trace<const DIM: usize>(mat: SquareMatrix<DIM>) {
    let expected = (0..DIM).map(|idx| mat[(idx, idx)]).sum::<Scalar>();
    // Assert return type is right
    let result: Scalar = mat.trace();
    assert_close(expected, result);
}

macro_rules! generate_tests {
    () => {
        generate_tests!(1, 2, 3, 4, 5, 6, 7);
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

                #[quickcheck]
                fn [< dot $dim >](lhs: Vector<$dim>, rhs: Vector<$dim>) {
                    test_dot::<$dim>(lhs, rhs);
                }

                #[quickcheck]
                fn [< norm $dim >](vec: Vector<$dim>) {
                    test_norm::<$dim>(vec);
                }

                #[quickcheck]
                fn [< trace $dim >](mat: SquareMatrix<$dim>) {
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
                ($dim, 7)
            );
        )*
    };
    ($(($dim1:literal, $dim2:literal)),*) => {
        $(
            paste! {
                #[quickcheck]
                fn [< from_col_major_elems_ $dim1 x $dim2 >](elems: Vec<Scalar>) {
                    test_from_col_major_elems::<$dim1, $dim2>(elems);
                }

                #[quickcheck]
                fn [< into_col_major_elems_ $dim1 x $dim2 >](mat: Matrix<$dim1, $dim2>) {
                    test_into_col_major_elems::<$dim1, $dim2>(mat);
                }

                #[quickcheck]
                fn [< col_major_elems_ $dim1 x $dim2 >](mat: Matrix<$dim1, $dim2>) {
                    test_col_major_elems::<$dim1, $dim2>(mat);
                }

                #[quickcheck]
                fn [< col_major_elems_mut_ $dim1 x $dim2 >](mat: Matrix<$dim1, $dim2>) {
                    test_col_major_elems_mut::<$dim1, $dim2>(mat);
                }

                #[quickcheck]
                fn [< transpose $dim1 x $dim2 >](mat: Matrix<$dim1, $dim2>) {
                    test_transpose::<$dim1, $dim2>(mat)
                }
            }
        )*
    }
}
generate_tests!();
