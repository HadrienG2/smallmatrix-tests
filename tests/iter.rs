//! Iterator-related tests
//!
//! Split out from other tests to increase build parallelism

mod common;

use self::common::*;
use paste::paste;
use quickcheck_macros::quickcheck;
use simd_tests::{Matrix, Scalar};

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

fn test_col_iterator<const ROWS: usize, const COLS: usize>(mat: Matrix<ROWS, COLS>) {
    assert_eq!(mat.into_iter().count(), COLS);
    for (idx, col_vec) in mat.into_iter().enumerate() {
        let expected = mat.col(idx);
        assert_bits_eq(expected, col_vec);
    }
    assert_bits_eq(mat, Matrix::<ROWS, COLS>::from_iter(mat.into_iter()));
}

macro_rules! generate_tests {
    () => {
        generate_tests!(1, 2, 3, 4, 5, 6, 7, 8);
    };
    ($($dim:literal),*) => {
        $(
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
                fn [< col_iterator $dim1 x $dim2 >](mat: Matrix<$dim1, $dim2>) {
                    test_col_iterator::<$dim1, $dim2>(mat);
                }
            }
        )*
    }
}
generate_tests!();
