mod common;

use self::common::assert_panics;
use paste::paste;
use quickcheck_macros::quickcheck;
use simd_tests::{Matrix, Vector};
use std::panic::UnwindSafe;

// Signature of op asserts that return type is right
fn test_block<
    const ROWS: usize,
    const COLS: usize,
    const BLOCK_ROWS: usize,
    const BLOCK_COLS: usize,
>(
    mat: Matrix<ROWS, COLS>,
    start_row: usize,
    start_col: usize,
    op: Box<dyn FnOnce() -> Matrix<BLOCK_ROWS, BLOCK_COLS> + UnwindSafe>,
) {
    if BLOCK_ROWS <= ROWS.saturating_sub(start_row) && BLOCK_COLS <= COLS.saturating_sub(start_col)
    {
        let result = op();
        for dest_row in 0..BLOCK_ROWS {
            let src_row = dest_row + start_row;
            for dest_col in 0..BLOCK_COLS {
                let src_col = dest_col + start_col;
                assert_eq!(
                    result[(dest_row, dest_col)].to_bits(),
                    mat[(src_row, src_col)].to_bits()
                );
            }
        }
    } else {
        assert_panics(op)
    }
}

// Signature of op asserts that return type is right
fn test_rows<const ROWS: usize, const COLS: usize, const BLOCK_ROWS: usize>(
    mat: Matrix<ROWS, COLS>,
    start_row: usize,
    op: Box<dyn FnOnce() -> Matrix<BLOCK_ROWS, COLS> + UnwindSafe>,
) {
    test_block::<ROWS, COLS, BLOCK_ROWS, COLS>(mat, start_row, 0, op)
}

// Signature of op asserts that return type is right
fn test_segment<const DIM: usize, const SUB_DIM: usize>(
    vec: Vector<DIM>,
    idx: usize,
    op: Box<dyn FnOnce() -> Vector<SUB_DIM> + UnwindSafe>,
) {
    test_rows::<DIM, 1, SUB_DIM>(vec, idx, op)
}

// Signature of op asserts that output matrix has the right dimension at compile time
fn test_cols<const ROWS: usize, const COLS: usize, const BLOCK_COLS: usize>(
    mat: Matrix<ROWS, COLS>,
    start_col: usize,
    op: Box<dyn FnOnce() -> Matrix<ROWS, BLOCK_COLS> + UnwindSafe>,
) {
    test_block::<ROWS, COLS, ROWS, BLOCK_COLS>(mat, 0, start_col, op)
}

// Generate tests for specific matrix dimensions
macro_rules! generate_tests {
    () => {
        // Blocks tests have huge combinatorics, so we only generate them for a
        // small range of matrix dimensions, which is still large enough to
        // catch good old off-by-one issues
        generate_tests!(1, 2, 3);
    };
    ($($dim:literal),*) => {
        $(
            generate_tests!(
                ($dim, 1),
                ($dim, 2),
                ($dim, 3)
            );
        )*
    };
    ($(($dim1:literal, $dim2:literal)),*) => {
        $(
            paste! {
                #[quickcheck]
                fn [< segment $dim1 _of_vec $dim2 >](vec: Vector<$dim2>, idx: usize) {
                    test_segment::<$dim2, $dim1>(
                        vec,
                        idx,
                        Box::new(move || vec.segment::<$dim1>(idx))
                    )
                }

                #[quickcheck]
                fn [< head $dim1 _of_vec $dim2 >](vec: Vector<$dim2>) {
                    test_segment::<$dim2, $dim1>(
                        vec,
                        0,
                        Box::new(move || vec.head::<$dim1>())
                    )
                }

                #[quickcheck]
                fn [< tail $dim1 _of_vec $dim2 >](vec: Vector<$dim2>) {
                    test_segment::<$dim2, $dim1>(
                        vec,
                        ($dim2 as usize).saturating_sub(($dim1 as usize)),
                        Box::new(move || vec.tail::<$dim1>())
                    )
                }

                #[quickcheck]
                fn [< row $dim1 x $dim2 >](
                    mat: Matrix<$dim1, $dim2>,
                    row: usize
                ) {
                    test_rows::<$dim1, $dim2, 1>(
                        mat,
                        row,
                        Box::new(move || mat.row(row))
                    )
                }

                #[quickcheck]
                fn [< col $dim1 x $dim2 >](
                    mat: Matrix<$dim1, $dim2>,
                    col: usize
                ) {
                    test_cols::<$dim1, $dim2, 1>(
                        mat,
                        col,
                        Box::new(move || mat.col(col))
                    )
                }
            }

            generate_tests!(
                ($dim1, $dim2, 1),
                ($dim1, $dim2, 2),
                ($dim1, $dim2, 3)
            );
        )*
    };
    ($(($dim1:literal, $dim2:literal, $dim3:literal)),*) => {
        $(
            paste! {
                #[quickcheck]
                fn [< rows $dim1 _of_mat $dim2 x $dim3 >](
                    mat: Matrix<$dim2, $dim3>,
                    start_row: usize
                ) {
                    test_rows::<$dim2, $dim3, $dim1>(
                        mat,
                        start_row,
                        Box::new(move || mat.rows::<$dim1>(start_row))
                    )
                }

                #[quickcheck]
                fn [< top $dim1 _of_mat $dim2 x $dim3 >](
                    mat: Matrix<$dim2, $dim3>,
                ) {
                    test_rows::<$dim2, $dim3, $dim1>(
                        mat,
                        0,
                        Box::new(move || mat.top_rows::<$dim1>())
                    )
                }

                #[quickcheck]
                fn [< bottom $dim1 _of_mat $dim2 x $dim3 >](
                    mat: Matrix<$dim2, $dim3>,
                ) {
                    test_rows::<$dim2, $dim3, $dim1>(
                        mat,
                        ($dim2 as usize).saturating_sub($dim1),
                        Box::new(move || mat.bottom_rows::<$dim1>())
                    )
                }

                #[quickcheck]
                fn [< cols $dim1 _of_mat $dim2 x $dim3 >](
                    mat: Matrix<$dim2, $dim3>,
                    start_col: usize
                ) {
                    test_cols::<$dim2, $dim3, $dim1>(
                        mat,
                        start_col,
                        Box::new(move || mat.cols::<$dim1>(start_col))
                    )
                }

                #[quickcheck]
                fn [< left $dim1 _of_mat $dim2 x $dim3 >](
                    mat: Matrix<$dim2, $dim3>,
                ) {
                    test_cols::<$dim2, $dim3, $dim1>(
                        mat,
                        0,
                        Box::new(move || mat.left_cols::<$dim1>())
                    )
                }

                #[quickcheck]
                fn [< right $dim1 _of_mat $dim2 x $dim3 >](
                    mat: Matrix<$dim2, $dim3>,
                ) {
                    test_cols::<$dim2, $dim3, $dim1>(
                        mat,
                        ($dim3 as usize).saturating_sub($dim1),
                        Box::new(move || mat.right_cols::<$dim1>())
                    )
                }
            }

            generate_tests!(
                ($dim1, $dim2, $dim3, 1),
                ($dim1, $dim2, $dim3, 2),
                ($dim1, $dim2, $dim3, 3)
            );
        )*
    };
    ($(($dim1:literal, $dim2:literal, $dim3:literal, $dim4:literal)),*) => {
        $(
            paste! {
                #[quickcheck]
                fn [< block $dim1 x $dim2 _of_mat $dim3 x $dim4 >](
                    mat: Matrix<$dim3, $dim4>,
                    start_row: usize,
                    start_col: usize
                ) {
                    test_block::<$dim3, $dim4, $dim1, $dim2>(
                        mat,
                        start_row,
                        start_col,
                        Box::new(move || mat.block::<$dim1, $dim2>(start_row, start_col))
                    )
                }

                #[quickcheck]
                fn [< topleft $dim1 x $dim2 _of_mat $dim3 x $dim4 >](
                    mat: Matrix<$dim3, $dim4>
                ) {
                    test_block::<$dim3, $dim4, $dim1, $dim2>(
                        mat,
                        0,
                        0,
                        Box::new(move || mat.top_left_corner::<$dim1, $dim2>())
                    )
                }

                #[quickcheck]
                fn [< topright $dim1 x $dim2 _of_mat $dim3 x $dim4 >](
                    mat: Matrix<$dim3, $dim4>
                ) {
                    test_block::<$dim3, $dim4, $dim1, $dim2>(
                        mat,
                        0,
                        ($dim4 as usize).saturating_sub($dim2),
                        Box::new(move || mat.top_right_corner::<$dim1, $dim2>())
                    )
                }

                #[quickcheck]
                fn [< bottomleft $dim1 x $dim2 _of_mat $dim3 x $dim4 >](
                    mat: Matrix<$dim3, $dim4>
                ) {
                    test_block::<$dim3, $dim4, $dim1, $dim2>(
                        mat,
                        ($dim3 as usize).saturating_sub($dim1),
                        0,
                        Box::new(move || mat.bottom_left_corner::<$dim1, $dim2>())
                    )
                }

                #[quickcheck]
                fn [< bottomright $dim1 x $dim2 _of_mat $dim3 x $dim4 >](
                    mat: Matrix<$dim3, $dim4>
                ) {
                    test_block::<$dim3, $dim4, $dim1, $dim2>(
                        mat,
                        ($dim3 as usize).saturating_sub($dim1),
                        ($dim4 as usize).saturating_sub($dim2),
                        Box::new(move || mat.bottom_right_corner::<$dim1, $dim2>())
                    )
                }
            }
        )*
    };
}
generate_tests!();
