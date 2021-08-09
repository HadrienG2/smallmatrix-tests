#![feature(const_generics, const_evaluatable_checked)]

mod common;

use self::common::*;
use num_traits::{One, Zero};
use paste::paste;
use quickcheck::TestResult;
use quickcheck_macros::quickcheck;
use simd_tests::{Matrix, Scalar, SquareMatrix, Vector, X, Y, Z};

fn test_clone<const ROWS: usize, const COLS: usize>(mat: Matrix<ROWS, COLS>) {
    let expected = Matrix::<ROWS, COLS>::from_col_major_elems(mat.into_col_major_elems());
    assert_bits_eq(expected, mat.clone());
}

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
                #[quickcheck]
                fn [< partial_eq $dim1 x $dim2 >](mat1: Matrix<$dim1, $dim2>, mat2: Matrix<$dim1, $dim2>) {
                    test_partial_eq::<$dim1, $dim2>(mat1, mat2);
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

                #[quickcheck]
                fn [< transpose $dim1 x $dim2 >](mat: Matrix<$dim1, $dim2>) {
                    test_transpose::<$dim1, $dim2>(mat)
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
