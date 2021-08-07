//! Matrix concatenation tests
//!
//! Split into their own codegen unit and operating on a reduced range of
//! matrix sizes and block dimensions for code bloat reasons.

#![feature(const_generics, const_evaluatable_checked)]

mod common;

use paste::paste;
use quickcheck_macros::quickcheck;
use simd_tests::{Matrix, Vector};

fn test_cat<const LEFT_DIM: usize, const RIGHT_DIM: usize>(
    lhs: Vector<LEFT_DIM>,
    rhs: Vector<RIGHT_DIM>,
) where
    [(); LEFT_DIM + RIGHT_DIM]: ,
{
    // Assert return type is right
    let out: Vector<{ LEFT_DIM + RIGHT_DIM }> = lhs.cat(rhs);
    for (src, dest) in lhs
        .into_col_major_elems()
        .chain(rhs.into_col_major_elems())
        .zip(out.into_col_major_elems())
    {
        assert_eq!(src.to_bits(), dest.to_bits());
    }
}

fn test_hcat<const LEFT_ROWS: usize, const LEFT_COLS: usize, const RIGHT_COLS: usize>(
    lhs: Matrix<LEFT_ROWS, LEFT_COLS>,
    rhs: Matrix<LEFT_ROWS, RIGHT_COLS>,
) where
    [(); LEFT_COLS + RIGHT_COLS]: ,
{
    // Assert return type is right
    let out: Matrix<LEFT_ROWS, { LEFT_COLS + RIGHT_COLS }> = lhs.hcat(rhs);
    for (src, dest) in lhs
        .into_col_major_elems()
        .chain(rhs.into_col_major_elems())
        .zip(out.into_col_major_elems())
    {
        assert_eq!(src.to_bits(), dest.to_bits());
    }
}

fn test_vcat<const LEFT_ROWS: usize, const LEFT_COLS: usize, const RIGHT_ROWS: usize>(
    lhs: Matrix<LEFT_ROWS, LEFT_COLS>,
    rhs: Matrix<RIGHT_ROWS, LEFT_COLS>,
) where
    [(); LEFT_ROWS + RIGHT_ROWS]: ,
{
    // Assert return type is right
    let out: Matrix<{ LEFT_ROWS + RIGHT_ROWS }, LEFT_COLS> = lhs.vcat(rhs);
    for ((col_lhs, col_rhs), col_dest) in lhs.into_iter().zip(rhs.into_iter()).zip(out.into_iter())
    {
        let col_src = col_lhs.cat(col_rhs);
        for (src, dest) in col_src
            .into_col_major_elems()
            .zip(col_dest.into_col_major_elems())
        {
            assert_eq!(src.to_bits(), dest.to_bits());
        }
    }
}

macro_rules! generate_tests {
    () => {
        generate_tests!(1, 2, 3, 4);
    };
    ($($dim:literal),*) => {
        $(
            generate_tests!(
                ($dim, 1),
                ($dim, 2),
                ($dim, 3),
                ($dim, 4)
            );
        )*
    };
    ($(($dim1:literal, $dim2:literal)),*) => {
        $(
            paste! {
                #[quickcheck]
                fn [< cat_vec $dim1 _vec $dim2 >](lhs: Vector<$dim1>, rhs: Vector<$dim2>) {
                    test_cat::<$dim1, $dim2>(lhs, rhs);
                }
            }

            generate_tests!(
                ($dim1, $dim2, 1),
                ($dim1, $dim2, 2),
                ($dim1, $dim2, 3),
                ($dim1, $dim2, 4)
            );
        )*
    };
    ($(($dim1:literal, $dim2:literal, $dim3:literal)),*) => {
        $(
            paste! {
                #[quickcheck]
                fn [< hcat_ $dim1 x $dim2 _ $dim1 x $dim3 >](lhs: Matrix<$dim1, $dim2>, rhs: Matrix<$dim1, $dim3>) {
                    test_hcat::<$dim1, $dim2, $dim3>(lhs, rhs);
                }

                #[quickcheck]
                fn [< vcat_ $dim1 x $dim2 _ $dim3 x $dim2 >](lhs: Matrix<$dim1, $dim2>, rhs: Matrix<$dim3, $dim2>) {
                    test_vcat::<$dim1, $dim2, $dim3>(lhs, rhs);
                }
            }
        )*
    }
}
generate_tests!();
