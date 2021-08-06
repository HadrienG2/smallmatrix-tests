#![feature(const_generics, const_evaluatable_checked)]
#![feature(specialization)]

use core::{
    iter::{Product, Sum},
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
};
use genawaiter::yield_;
use more_asserts::*;
use num_traits::{One, Pow, Zero};
use quickcheck::Arbitrary;
use rand::{
    distributions::{Distribution, Standard},
    Fill, Rng,
};

// TODO: Add a layer of genericity over scalar type later on, and bring some
//       bidirectional From impls along the way
pub type Scalar = f64;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Matrix<const ROWS: usize, const COLS: usize>([[Scalar; ROWS]; COLS]);
pub type SquareMatrix<const DIM: usize> = Matrix<DIM, DIM>;

pub type ColVector<const ROWS: usize> = Matrix<ROWS, 1>;
pub type RowVector<const COLS: usize> = Matrix<1, COLS>;
// FIXME: Can't implement vector operations on row vectors due to 1x1 ambiguity.
//        May want to hack through this via specialization in the future.
pub type Vector<const DIM: usize> = ColVector<DIM>;

pub const X: usize = 0;
pub const Y: usize = 1;
pub const Z: usize = 2;

// 1x1 matrices are kinda sorta like scalars
impl From<Scalar> for Matrix<1, 1> {
    fn from(x: Scalar) -> Self {
        Self([[x]])
    }
}

impl From<Matrix<1, 1>> for Scalar {
    fn from(x: Matrix<1, 1>) -> Self {
        x.0[0][0]
    }
}

// NOTE: Can't derive Default as it isn't implemented for all arrays
impl<const ROWS: usize, const COLS: usize> Default for Matrix<ROWS, COLS> {
    fn default() -> Self {
        Self([[Scalar::default(); ROWS]; COLS])
    }
}

impl<const ROWS: usize, const COLS: usize> Zero for Matrix<ROWS, COLS> {
    fn zero() -> Self {
        Self([[Scalar::zero(); ROWS]; COLS])
    }

    fn is_zero(&self) -> bool {
        self.col_major_elems().all(Scalar::is_zero)
    }
}

// Random matrix generation
impl<const ROWS: usize, const COLS: usize> Fill for Matrix<ROWS, COLS> {
    fn try_fill<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<(), rand::Error> {
        for col in self.0.iter_mut() {
            col.try_fill(rng)?;
        }
        Ok(())
    }
}

impl<const ROWS: usize, const COLS: usize> Distribution<Matrix<ROWS, COLS>> for Standard
where
    [(); ROWS * COLS]: ,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Matrix<ROWS, COLS> {
        Matrix::<ROWS, COLS>::from_col_major_elems(rng.gen::<[Scalar; ROWS * COLS]>().into_iter())
    }
}

impl<const ROWS: usize, const COLS: usize> Arbitrary for Matrix<ROWS, COLS> {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        Self::from_col_major_elems((0..ROWS * COLS).map(|_| f64::arbitrary(g)))
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        let seed = self.clone();
        Box::new(
            genawaiter::rc::gen!({
                for (elem_idx, old_elem) in seed.into_col_major_elems().enumerate() {
                    for new_elem in old_elem.shrink() {
                        yield_!(Self::from_col_major_elems(
                            seed.into_col_major_elems()
                                .take(elem_idx)
                                .chain(core::iter::once(new_elem))
                                .chain(seed.into_col_major_elems().skip(elem_idx + 1))
                        ));
                    }
                }
            })
            .into_iter(),
        )
    }
}

impl<const DIM: usize> ColVector<DIM> {
    /// Unit vector
    pub fn unit(idx: usize) -> Self {
        assert_lt!(idx, DIM, "Requested coordinate is out of bounds");
        Self::from_col_major_elems((0..DIM).map(|idx2| (idx2 == idx) as u8 as _))
    }
}

impl<const DIM: usize> SquareMatrix<DIM> {
    /// Identity matrix
    pub fn identity() -> Self {
        Self::from_iter((0..DIM).map(|col| ColVector::<DIM>::unit(col)))
    }
}

impl<const DIM: usize> One for SquareMatrix<DIM> {
    fn one() -> Self {
        Self::identity()
    }
}

// TODO: Projection and rotation matrices

impl<const ROWS: usize, const COLS: usize> Matrix<ROWS, COLS> {
    /// Construction from column-major data
    pub fn from_col_major_elems<T: IntoIterator<Item = Scalar>>(input: T) -> Self {
        let mut result = Self::default();
        let mut targets = result.0.iter_mut().flat_map(|col| col.iter_mut());
        let mut sources = input.into_iter();
        loop {
            match (targets.next(), sources.next()) {
                (Some(target), Some(source)) => *target = source,
                (None, None) => break result,
                (Some(_), None) => panic!("Too few elements in input iterator"),
                (None, Some(_)) => panic!("Too many elements in input iterator"),
            }
        }
    }

    /// Turn into column-major data
    pub fn into_col_major_elems(self) -> impl Iterator<Item = Scalar> {
        self.0.into_iter().flat_map(|col| col.into_iter())
    }

    /// Iterate over inner column-major data
    pub fn col_major_elems(&self) -> impl Iterator<Item = &Scalar> {
        self.0.iter().flat_map(|col| col.iter())
    }

    /// Iterate over inner column-major data, allowing mutation
    pub fn col_major_elems_mut(&mut self) -> impl Iterator<Item = &mut Scalar> {
        self.0.iter_mut().flat_map(|col| col.iter_mut())
    }
}

// Iteration over columns and construction from a set of columns
impl<const ROWS: usize, const COLS: usize> FromIterator<ColVector<ROWS>> for Matrix<ROWS, COLS> {
    fn from_iter<T: IntoIterator<Item = ColVector<ROWS>>>(input: T) -> Self {
        let mut result = Self::default();
        let mut targets = result.0.iter_mut();
        let mut sources = input.into_iter();
        loop {
            match (targets.next(), sources.next()) {
                (Some(target), Some(source)) => *target = source.0[0],
                (None, None) => break result,
                (Some(_), None) => panic!("Too few elements in input iterator"),
                (None, Some(_)) => panic!("Too many elements in input iterator"),
            }
        }
    }
}

/// Iterator over the columns of a matrix
pub struct ColIter<const ROWS: usize, const COLS: usize>(
    core::array::IntoIter<[Scalar; ROWS], COLS>,
);

impl<const ROWS: usize, const COLS: usize> Iterator for ColIter<ROWS, COLS> {
    type Item = ColVector<ROWS>;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|arr| Matrix::<ROWS, 1>([arr]))
    }
}

impl<const ROWS: usize, const COLS: usize> IntoIterator for Matrix<ROWS, COLS> {
    type Item = ColVector<ROWS>;
    type IntoIter = ColIter<ROWS, COLS>;
    fn into_iter(self) -> Self::IntoIter {
        ColIter(self.0.into_iter())
    }
}

impl<const LEFT_DIM: usize> ColVector<LEFT_DIM> {
    /// Vector concatenation
    pub fn cat<const RIGHT_DIM: usize>(
        self,
        rhs: ColVector<RIGHT_DIM>,
    ) -> ColVector<{ LEFT_DIM + RIGHT_DIM }> {
        ColVector::<{ LEFT_DIM + RIGHT_DIM }>::from_col_major_elems(
            self.into_col_major_elems()
                .chain(rhs.into_col_major_elems()),
        )
    }
}

impl<const LEFT_ROWS: usize, const LEFT_COLS: usize> Matrix<LEFT_ROWS, LEFT_COLS> {
    /// Horizontal matrix concatenation
    pub fn hcat<const RIGHT_COLS: usize>(
        self,
        rhs: Matrix<LEFT_ROWS, RIGHT_COLS>,
    ) -> Matrix<LEFT_ROWS, { LEFT_COLS + RIGHT_COLS }> {
        Matrix::<LEFT_ROWS, { LEFT_COLS + RIGHT_COLS }>::from_iter(
            self.into_iter().chain(rhs.into_iter()),
        )
    }

    /// Vertical matrix concatenation
    pub fn vcat<const RIGHT_ROWS: usize>(
        self,
        rhs: Matrix<RIGHT_ROWS, LEFT_COLS>,
    ) -> Matrix<{ LEFT_ROWS + RIGHT_ROWS }, LEFT_COLS> {
        Matrix::<{ LEFT_ROWS + RIGHT_ROWS }, LEFT_COLS>::from_iter(
            self.into_iter()
                .zip(rhs.into_iter())
                .map(|(left_col, right_col)| left_col.cat(right_col)),
        )
    }
}

// TODO: Implement Display, LowerExp, UpperExp, and a matching custom Debug impl

// Element access
impl<const ROWS: usize, const COLS: usize> Index<(usize, usize)> for Matrix<ROWS, COLS> {
    type Output = Scalar;
    fn index(&self, (row, col): (usize, usize)) -> &Scalar {
        &self.0[col][row]
    }
}

impl<const ROWS: usize, const COLS: usize> IndexMut<(usize, usize)> for Matrix<ROWS, COLS> {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Scalar {
        &mut self.0[col][row]
    }
}

impl<const ROWS: usize> Index<usize> for ColVector<ROWS> {
    type Output = Scalar;
    fn index(&self, idx: usize) -> &Scalar {
        &self[(idx, 0)]
    }
}

impl<const ROWS: usize> IndexMut<usize> for ColVector<ROWS> {
    fn index_mut(&mut self, idx: usize) -> &mut Scalar {
        &mut self[(idx, 0)]
    }
}

// Vector slicing
impl<const DIM: usize> ColVector<DIM> {
    /// Extract a segment of a vector
    pub fn segment<const SUB_DIM: usize>(self, start_idx: usize) -> ColVector<SUB_DIM> {
        assert_le!(SUB_DIM, DIM, "Segment dimension is out of bounds");
        assert_le!(start_idx, DIM - SUB_DIM, "Start index is out of bounds");
        ColVector::<SUB_DIM>::from_col_major_elems(
            self.into_col_major_elems().skip(start_idx).take(SUB_DIM),
        )
    }

    /// Extract first elements of a vector
    pub fn head<const SUB_DIM: usize>(self) -> ColVector<SUB_DIM> {
        self.segment::<SUB_DIM>(0)
    }

    /// Extract last elements of a vector
    pub fn tail<const SUB_DIM: usize>(self) -> ColVector<SUB_DIM> {
        self.segment::<SUB_DIM>(DIM - SUB_DIM)
    }
}

// Matrix slicing
impl<const ROWS: usize, const COLS: usize> Matrix<ROWS, COLS> {
    /// Extract a block from a matrix
    pub fn block<const SUB_ROWS: usize, const SUB_COLS: usize>(
        self,
        start_row: usize,
        start_col: usize,
    ) -> Matrix<SUB_ROWS, SUB_COLS> {
        assert_le!(SUB_ROWS, ROWS, "Block rows are out of bounds");
        assert_le!(start_row, ROWS - SUB_ROWS, "Start row is out of bounds");
        assert_le!(SUB_COLS, COLS, "Block cols are out of bounds");
        assert_le!(start_col, COLS - SUB_COLS, "Start col is out of bounds");
        Matrix::<SUB_ROWS, SUB_COLS>::from_iter(
            self.into_iter()
                .skip(start_col)
                .take(SUB_COLS)
                .map(|col| col.segment::<SUB_ROWS>(start_row)),
        )
    }

    /// Extract the top-left corner of a matrix
    pub fn top_left_corner<const SUB_ROWS: usize, const SUB_COLS: usize>(
        self,
    ) -> Matrix<SUB_ROWS, SUB_COLS> {
        self.block::<SUB_ROWS, SUB_COLS>(0, 0)
    }

    /// Extract the top-right corner of a matrix
    pub fn top_right_corner<const SUB_ROWS: usize, const SUB_COLS: usize>(
        self,
    ) -> Matrix<SUB_ROWS, SUB_COLS> {
        self.block::<SUB_ROWS, SUB_COLS>(0, COLS - SUB_COLS)
    }

    /// Extract the bottom-left corner of a matrix
    pub fn bottom_left_corner<const SUB_ROWS: usize, const SUB_COLS: usize>(
        self,
    ) -> Matrix<SUB_ROWS, SUB_COLS> {
        self.block::<SUB_ROWS, SUB_COLS>(ROWS - SUB_ROWS, 0)
    }

    /// Extract the bottom-right corner of a matrix
    pub fn bottom_right_corner<const SUB_ROWS: usize, const SUB_COLS: usize>(
        self,
    ) -> Matrix<SUB_ROWS, SUB_COLS> {
        self.block::<SUB_ROWS, SUB_COLS>(ROWS - SUB_ROWS, COLS - SUB_COLS)
    }

    /// Extract a set of rows from a matrix
    pub fn rows<const SUB_ROWS: usize>(self, start_row: usize) -> Matrix<SUB_ROWS, COLS> {
        self.block::<SUB_ROWS, COLS>(start_row, 0)
    }

    /// Extract the N-th row of a matrix
    pub fn row(self, row: usize) -> RowVector<COLS> {
        self.rows::<1>(row)
    }

    /// Extract the top rows from a matrix
    pub fn top_rows<const SUB_ROWS: usize>(self) -> Matrix<SUB_ROWS, COLS> {
        self.rows::<SUB_ROWS>(0)
    }

    /// Extract the bottom rows from a matrix
    pub fn bottom_rows<const SUB_ROWS: usize>(self) -> Matrix<SUB_ROWS, COLS> {
        self.rows::<SUB_ROWS>(ROWS - SUB_ROWS)
    }

    /// Extract a set of columns from a matrix
    pub fn cols<const SUB_COLS: usize>(self, start_col: usize) -> Matrix<ROWS, SUB_COLS> {
        self.block::<ROWS, SUB_COLS>(0, start_col)
    }

    /// Extract the N-th column of a matrix
    pub fn col(self, col: usize) -> ColVector<ROWS> {
        self.cols::<1>(col)
    }

    /// Extract the left columns from a matrix
    pub fn left_cols<const SUB_COLS: usize>(self) -> Matrix<ROWS, SUB_COLS> {
        self.cols::<SUB_COLS>(0)
    }

    /// Extract the right columns from a matrix
    pub fn right_cols<const SUB_COLS: usize>(self) -> Matrix<ROWS, SUB_COLS> {
        self.cols::<SUB_COLS>(COLS - SUB_COLS)
    }
}

impl<const ROWS: usize, const COLS: usize> Matrix<ROWS, COLS> {
    /// Matrix transpose
    pub fn transpose(self) -> Matrix<COLS, ROWS> {
        // TODO: Avoid eager computation, use expression templates instead
        Matrix::<COLS, ROWS>::from_col_major_elems(
            (0..ROWS * COLS)
                .map(|idx| (idx / COLS, idx % COLS))
                .map(|row_col| self[row_col]),
        )
    }
}

// Negation
impl<const ROWS: usize, const COLS: usize> Neg for Matrix<ROWS, COLS> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::from_col_major_elems(self.into_col_major_elems().map(|elem| -elem))
    }
}

// Multiplication and division by scalar
impl<const ROWS: usize, const COLS: usize> Mul<Scalar> for Matrix<ROWS, COLS> {
    type Output = Self;
    fn mul(self, rhs: Scalar) -> Self {
        Self::from_col_major_elems(self.into_col_major_elems().map(|elem| elem * rhs))
    }
}

impl<const ROWS: usize, const COLS: usize> Mul<Matrix<ROWS, COLS>> for Scalar {
    type Output = Matrix<ROWS, COLS>;
    fn mul(self, rhs: Matrix<ROWS, COLS>) -> Matrix<ROWS, COLS> {
        rhs * self
    }
}

impl<const ROWS: usize, const COLS: usize> MulAssign<Scalar> for Matrix<ROWS, COLS> {
    fn mul_assign(&mut self, rhs: Scalar) {
        for elem in self.col_major_elems_mut() {
            *elem *= rhs;
        }
    }
}

impl<const ROWS: usize, const COLS: usize> Div<Scalar> for Matrix<ROWS, COLS> {
    type Output = Self;
    fn div(self, rhs: Scalar) -> Self {
        Self::from_col_major_elems(self.into_col_major_elems().map(|elem| elem / rhs))
    }
}

impl<const ROWS: usize, const COLS: usize> DivAssign<Scalar> for Matrix<ROWS, COLS> {
    fn div_assign(&mut self, rhs: Scalar) {
        for elem in self.col_major_elems_mut() {
            *elem /= rhs;
        }
    }
}

// Matrix addition
impl<const ROWS: usize, const COLS: usize> Add for Matrix<ROWS, COLS> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::from_col_major_elems(
            self.into_col_major_elems()
                .zip(rhs.into_col_major_elems())
                .map(|(lhs, rhs)| lhs + rhs),
        )
    }
}

impl<const ROWS: usize, const COLS: usize> AddAssign for Matrix<ROWS, COLS> {
    fn add_assign(&mut self, rhs: Self) {
        for (lhs, rhs) in self.col_major_elems_mut().zip(rhs.into_col_major_elems()) {
            *lhs += rhs
        }
    }
}

impl<const ROWS: usize, const COLS: usize> Sum for Matrix<ROWS, COLS> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl<const ROWS: usize, const COLS: usize> Sub for Matrix<ROWS, COLS> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::from_col_major_elems(
            self.into_col_major_elems()
                .zip(rhs.into_col_major_elems())
                .map(|(lhs, rhs)| lhs - rhs),
        )
    }
}

impl<const ROWS: usize, const COLS: usize> SubAssign for Matrix<ROWS, COLS> {
    fn sub_assign(&mut self, rhs: Self) {
        for (lhs, rhs) in self.col_major_elems_mut().zip(rhs.into_col_major_elems()) {
            *lhs -= rhs
        }
    }
}

// Matrix multiplication
impl<const LEFT_ROWS: usize, const LEFT_COLS: usize, const RIGHT_COLS: usize>
    Mul<Matrix<LEFT_COLS, RIGHT_COLS>> for Matrix<LEFT_ROWS, LEFT_COLS>
{
    type Output = Matrix<LEFT_ROWS, RIGHT_COLS>;
    fn mul(self, rhs: Matrix<LEFT_COLS, RIGHT_COLS>) -> Matrix<LEFT_ROWS, RIGHT_COLS> {
        Matrix::<LEFT_ROWS, RIGHT_COLS>::from_iter(rhs.into_iter().map(|right_col| {
            self.into_iter()
                .zip(right_col.into_col_major_elems())
                .map(|(left_col, right_elem)| left_col * right_elem)
                .sum()
        }))
    }
}

impl<const ROWS: usize, const COLS: usize> MulAssign<SquareMatrix<COLS>> for Matrix<ROWS, COLS> {
    fn mul_assign(&mut self, rhs: SquareMatrix<COLS>) {
        // Cannot optimize in-place product beyond this because each column of
        // the destination matrix uses every column from the source matrix
        *self = self.clone() * rhs;
    }
}

impl<const DIM: usize> Product for SquareMatrix<DIM> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

impl<RHS: Into<usize>, const DIM: usize> Pow<RHS> for SquareMatrix<DIM> {
    type Output = Self;
    fn pow(self, rhs: RHS) -> Self {
        num_traits::pow::pow(self, rhs.into())
    }
}

impl<const DIM: usize> ColVector<DIM> {
    /// Vector dot product
    pub fn dot(self, rhs: Self) -> Scalar {
        (self.transpose() * rhs).into()
    }

    /// Vector norm
    pub fn norm(self) -> Scalar {
        self.dot(self).sqrt()
    }
}

impl ColVector<3> {
    /// 3D vector cross product
    pub fn cross(self, rhs: Self) -> Self {
        Self::from_col_major_elems((X..=Z).map(|idx| {
            self[(idx + 1) % 3] * rhs[(idx + 2) % 3] - self[(idx + 2) % 3] * rhs[(idx + 1) % 3]
        }))
    }
}

impl<const DIM: usize> SquareMatrix<DIM> {
    /// Matrix trace
    pub fn trace(self) -> Scalar {
        (0..DIM).map(|idx| self[(idx, idx)]).sum()
    }
}

// Minor, cofactor, determinant and inverse have a dimension-dependent
// definition. Unfortunately, in the current state of Rust const generics, this
// requires a rather copious amount of dirty type system hackery.
//
// First of all, the type system must be convinced that a determinant definition
// exists for all matrix sizes. Otherwise, it will not compile a recursive
// definition of NxN matrix determinant, which, for N>=2, is based on the
// (N-1)x(N-1) determinant, because it does not manage to prove that another
// implementation for 1x1 matrices is enough to terminate the recursion.
//
// But the actual definition must change depending on matrix size, as otherwise
// we recurse forever until the point where N-1 underflows.
//
// So we need specialization, which means that we need a trait, that we shall
// carefully hide so that the poor user does not need to know about it.
#[doc(hidden)]
trait DeterminantInverse {
    /// Matrix determinant
    fn det_impl(self) -> Scalar;

    /// Matrix inverse
    fn inverse_impl(self) -> Self;
}
//
// The default implementation of that trait handles the 0x0 and 1x1 cases
impl<const DIM: usize> DeterminantInverse for SquareMatrix<DIM> {
    default fn det_impl(self) -> Scalar {
        match DIM {
            0 => 0.0,
            1 => self[(0, 0)],
            _ => unreachable!("Should be taken over by specialization"),
        }
    }

    default fn inverse_impl(self) -> Self {
        match DIM {
            0 => self,
            1 => {
                let inner = self[(0, 0)];
                assert_ne!(inner, 0.0, "Matrix is not invertible");
                Self::from_col_major_elems(core::iter::once(1.0 / inner))
            }
            _ => unreachable!("Should be taken over by specialization"),
        }
    }
}
//
// Then we introduce a dirty trick to only implement methods when a const
// generic parameter matches a certain predicate...
//
/// Hackish emulation of trait bounds for const parameters
pub struct ConstCheck<const CHECK: bool>;
/// Trait implemented by ConstCheck when its argument is true
pub trait True {}
impl True for ConstCheck<true> {}
//
// ...and we use that to implement matrix minors and cofactors on the matrices
// where it makes sense, that is to say, for square matrices larger than 2x2.
impl<const DIM: usize> SquareMatrix<DIM>
where
    [(); DIM - 1]: ,
    ConstCheck<{ DIM >= 2 }>: True,
{
    /// Matrix element minor
    pub fn minor(self, row: usize, col: usize) -> Scalar {
        assert_le!(row, DIM, "Row is out of bounds");
        assert_le!(col, DIM, "Column is out of bounds");
        SquareMatrix::<{ DIM - 1 }>::from_iter(
            self.into_iter()
                .take(col)
                .chain(self.into_iter().skip(col + 1))
                .map(|col| {
                    ColVector::<{ DIM - 1 }>::from_col_major_elems(
                        col.into_col_major_elems()
                            .take(row)
                            .chain(col.into_col_major_elems().skip(row + 1)),
                    )
                }),
        )
        .det_impl()
    }

    /// Matrix element cofactor
    pub fn cofactor(self, row: usize, col: usize) -> Scalar {
        (-1.0f64).powi((row + col) as _) * self.minor(row, col)
    }
}
//
// Using minors and cofactors, we can then implement a "specialized" determinant
// for matrix sizes above 2 that uses Laplace expansion.
impl<const DIM: usize> DeterminantInverse for SquareMatrix<DIM>
where
    [(); DIM - 1]: ,
    ConstCheck<{ DIM >= 2 }>: True,
{
    // Matrix determinant computation based on Laplace expansion
    // TODO: Switch to something like LU decomposition when DIM becomes high
    fn det_impl(self) -> Scalar {
        let left_col = self.left_cols::<1>();
        left_col
            .into_col_major_elems()
            .enumerate()
            .map(|(row, elem)| elem * self.cofactor(row, 0))
            .sum()
    }

    /// Matrix inverse computation based on Cramer's rule
    // TODO: Switch to something like LU decomposition when DIM becomes high
    fn inverse_impl(self) -> Self {
        let det = self.det_impl();
        assert_ne!(det, 0.0, "Matrix is not invertible");
        let cofactor_matrix = Self::from_col_major_elems(
            (0..DIM * DIM).map(|idx| self.cofactor(idx % DIM, idx / DIM)),
        );
        cofactor_matrix.transpose() / det
    }
}
//
// And finally, we expose an inherent determinant method for matrices of size
// above 1x1, so that users need not mess with our weird trait.
impl<const DIM: usize> SquareMatrix<DIM> {
    /// Matrix determinant
    pub fn det(self) -> Scalar {
        self.det_impl()
    }

    /// Matrix inverse
    pub fn inverse(self) -> Self {
        self.inverse_impl()
    }
}

// TODO: Linear solver, Gram-Schmidt, eigenvalues and eigenvectors...
// TODO: Special matrices (diagonal, tridiagonal, upper/lower triangular,
//       symmetric, hermitic) and unit vectors, with specialized handling.

#[cfg(test)]
mod tests {
    use super::*;
    use paste::paste;
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;
    use std::panic::UnwindSafe;

    fn panics<F: FnOnce() -> R + UnwindSafe, R>(f: F) -> bool {
        std::panic::catch_unwind(f).is_err()
    }

    fn assert_close(expected: Scalar, result: Scalar) {
        if expected.is_nan() {
            assert_eq!(result.to_bits(), expected.to_bits());
        } else if expected.is_infinite() {
            assert_eq!(result, expected);
        } else {
            assert_le!((result - expected).abs(), Scalar::EPSILON * expected.abs());
        }
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
        // Assert that the output has the right dimension
        let result: Vector<3> = lhs.cross(rhs);
        for (expected, result) in expected
            .into_col_major_elems()
            .zip(result.into_col_major_elems())
        {
            assert_close(expected, result);
        }
    }

    macro_rules! generate_tests {
        () => {
            // FIXME: Had to disable some test configurations to keep compile times reasonable.
            //        Do a measureme run to make sure we understand what's going on.
            generate_tests!(1, 2, 3, /*4, 5, 6, 7, */ 8);
        };
        ($($dim:literal),*) => {
            $(
                paste! {
                    #[test]
                    fn [< unit $dim _in_range >]() {
                        for idx in 0..$dim {
                            let unit = Vector::<$dim>::unit(idx);
                            for (idx2, elem) in unit.into_col_major_elems().enumerate() {
                                assert_eq!(elem, (idx2 == idx) as u8 as Scalar);
                            }
                        }
                    }

                    #[quickcheck]
                    fn [< unit $dim _out_of_range >](idx: usize) -> TestResult {
                        if idx < $dim {
                            return TestResult::discard();
                        }
                        TestResult::from_bool(panics(|| Vector::<$dim>::unit(idx)))

                    }

                    #[test]
                    fn [< identity $dim >]() {
                        let identity = SquareMatrix::<$dim>::identity();
                        for (col, col_vec) in identity.into_iter().enumerate() {
                            for (row, elem) in col_vec.into_col_major_elems().enumerate() {
                                assert_eq!(elem, (col == row) as u8 as Scalar);
                            }
                        }
                    }

                    #[quickcheck]
                    fn [< dot $dim >](lhs: Vector<$dim>, rhs: Vector<$dim>) {
                        let result = lhs.dot(rhs);
                        let expected =
                            lhs.into_col_major_elems()
                               .zip(rhs.into_col_major_elems())
                               .map(|(x, y)| x * y)
                               .sum::<Scalar>();
                        assert_close(expected, result);
                    }

                    #[quickcheck]
                    fn [< norm $dim >](vec: Vector<$dim>) {
                        let result = vec.norm();
                        let expected = vec.dot(vec).sqrt();
                        assert_close(expected, result);
                    }

                    #[quickcheck]
                    fn [< trace $dim >](mat: SquareMatrix<$dim>) {
                        let expected = (0..$dim).map(|idx| mat[(idx, idx)]).sum::<Scalar>();
                        let result = mat.trace();
                        assert_close(expected, result);
                    }
                }

                // FIXME: Had to disable some test configurations to keep compile times reasonable.
                //        Do a measureme run to make sure we understand what's going on.
                generate_tests!(
                    ($dim, 1),
                    ($dim, 2),
                    ($dim, 3),
                    // ($dim, 4),
                    // ($dim, 5),
                    // ($dim, 6),
                    // ($dim, 7),
                    ($dim, 8)
                );
            )*
        };
        ($(($dim1:literal, $dim2:literal)),*) => {
            $(
                paste! {
                    #[quickcheck]
                    fn [< from_col_major_elems_ $dim1 x $dim2 >](elems: Vec<Scalar>) -> TestResult {
                        if elems.len() == $dim1 * $dim2 {
                            let matrix = Matrix::<$dim1, $dim2>::from_col_major_elems(elems.iter().copied());
                            for (src, dest) in elems.into_iter().zip(matrix.into_col_major_elems()) {
                                assert_eq!(src.to_bits(), dest.to_bits());
                            }
                            TestResult::passed()
                        } else {
                            TestResult::from_bool(panics(|| Matrix::<$dim1, $dim2>::from_col_major_elems(elems.into_iter())))
                        }
                    }

                    #[quickcheck]
                    fn [< into_col_major_elems_ $dim1 x $dim2 >](matrix: Matrix<$dim1, $dim2>) {
                        assert_eq!(matrix.into_col_major_elems().count(), $dim1 * $dim2);
                        for (idx, dest) in matrix.into_col_major_elems().enumerate() {
                            let (col, row) = (idx / $dim1, idx % $dim1);
                            let src = matrix[(row, col)];
                            assert_eq!(src.to_bits(), dest.to_bits());
                        }
                    }

                    #[quickcheck]
                    fn [< col_major_elems_ $dim1 x $dim2 >](matrix: Matrix<$dim1, $dim2>) {
                        assert_eq!(matrix.col_major_elems().count(), $dim1 * $dim2);
                        for (idx, dest_ref) in matrix.col_major_elems().enumerate() {
                            let (col, row) = (idx / $dim1, idx % $dim1);
                            assert_eq!(&matrix[(row, col)] as *const Scalar, dest_ref as *const Scalar);
                        }
                    }

                    #[quickcheck]
                    fn [< col_major_elems_mut_ $dim1 x $dim2 >](mut matrix: Matrix<$dim1, $dim2>) {
                        assert_eq!(matrix.col_major_elems_mut().count(), $dim1 * $dim2);
                        let ptrs = matrix.col_major_elems_mut().map(|refmut| refmut as *mut Scalar).collect::<Vec<_>>();
                        for (idx, ptr) in ptrs.into_iter().enumerate() {
                            let (col, row) = (idx / $dim1, idx % $dim1);
                            assert_eq!(&mut matrix[(row, col)] as *mut Scalar, ptr);
                        }
                    }

                    #[quickcheck]
                    fn [< cat_vec $dim1 _vec $dim2 >](lhs: Vector<$dim1>, rhs: Vector<$dim2>) {
                        // Assert that the output has the right dimension
                        let out: Vector<{ $dim1 + $dim2 }> = lhs.cat(rhs);
                        for (src, dest) in lhs.into_col_major_elems().chain(rhs.into_col_major_elems()).zip(out.into_col_major_elems()) {
                            assert_eq!(src.to_bits(), dest.to_bits());
                        }
                    }

                    // Signature of op asserts that output vector has the right dimension at compile time
                    fn [< test_segment $dim1 _of_vec $dim2 >](
                        vec: Vector<$dim2>,
                        idx: usize,
                        op: impl FnOnce() -> Vector<$dim1> + UnwindSafe
                    ) -> bool {
                        if $dim1 <= ($dim2 as usize).saturating_sub(idx) {
                            for (src, dest) in vec.into_col_major_elems().skip(idx).zip(op().into_col_major_elems()) {
                                assert_eq!(src.to_bits(), dest.to_bits());
                            }
                            true
                        } else {
                            panics(op)
                        }
                    }

                    #[quickcheck]
                    fn [< segment $dim1 _of_vec $dim2 >](vec: Vector<$dim2>, idx: usize) -> bool {
                        [< test_segment $dim1 _of_vec $dim2 >](
                            vec,
                            idx,
                            || vec.segment::<$dim1>(idx)
                        )
                    }

                    #[quickcheck]
                    fn [< head $dim1 _of_vec $dim2 >](vec: Vector<$dim2>) -> bool {
                        [< test_segment $dim1 _of_vec $dim2 >](
                            vec,
                            0,
                            || vec.head::<$dim1>()
                        )
                    }

                    #[quickcheck]
                    fn [< tail $dim1 _of_vec $dim2 >](vec: Vector<$dim2>) -> bool {
                        [< test_segment $dim1 _of_vec $dim2 >](
                            vec,
                            ($dim2 as usize).saturating_sub(($dim1 as usize)),
                            || vec.tail::<$dim1>()
                        )
                    }

                    #[quickcheck]
                    fn [< row $dim1 x $dim2 >](
                        mat: Matrix<$dim1, $dim2>,
                        row: usize
                    ) -> bool {
                        [< test_rows1_of_mat $dim1 x $dim2 >](
                            mat,
                            row,
                            || mat.row(row)
                        )
                    }

                    #[quickcheck]
                    fn [< col $dim1 x $dim2 >](
                        mat: Matrix<$dim1, $dim2>,
                        col: usize
                    ) -> bool {
                        [< test_cols1_of_mat $dim1 x $dim2 >](
                            mat,
                            col,
                            || mat.col(col)
                        )
                    }

                    #[quickcheck]
                    fn [< transpose $dim1 x $dim2 >](mat: Matrix<$dim1, $dim2>) {
                        // Assert that the output has the right dimension
                        let tr: Matrix<$dim2, $dim1> = mat.transpose();
                        for src_row in 0..$dim1 {
                            for src_col in 0..$dim2 {
                                assert_eq!(
                                    mat[(src_row, src_col)].to_bits(),
                                    tr[(src_col, src_row)].to_bits()
                                );
                            }
                        }
                    }
                }

                // FIXME: Had to disable some test configurations to keep compile times reasonable.
                //        Do a measureme run to make sure we understand what's going on.
                generate_tests!(
                    ($dim1, $dim2, 1),
                    ($dim1, $dim2, 2),
                    ($dim1, $dim2, 3),
                    // ($dim1, $dim2, 4),
                    // ($dim1, $dim2, 5),
                    // ($dim1, $dim2, 6),
                    // ($dim1, $dim2, 7),
                    ($dim1, $dim2, 8)
                );
            )*
        };
        ($(($dim1:literal, $dim2:literal, $dim3:literal)),*) => {
            $(
                paste! {
                    #[quickcheck]
                    fn [< hcat_ $dim1 x $dim2 _ $dim1 x $dim3 >](lhs: Matrix<$dim1, $dim2>, rhs: Matrix<$dim1, $dim3>) {
                        // Assert that the output has the right dimension
                        let out: Matrix<$dim1, { $dim2 + $dim3 }> = lhs.hcat(rhs);
                        for (src, dest) in lhs.into_col_major_elems().chain(rhs.into_col_major_elems()).zip(out.into_col_major_elems()) {
                            assert_eq!(src.to_bits(), dest.to_bits());
                        }
                    }

                    #[quickcheck]
                    fn [< vcat_ $dim1 x $dim2 _ $dim3 x $dim2 >](lhs: Matrix<$dim1, $dim2>, rhs: Matrix<$dim3, $dim2>) {
                        // Assert that the output has the right dimension
                        let out: Matrix<{ $dim1 + $dim3 }, $dim2> = lhs.vcat(rhs);
                        for ((col_lhs, col_rhs), col_dest) in lhs.into_iter().zip(rhs.into_iter()).zip(out.into_iter()) {
                            let col_src = col_lhs.cat(col_rhs);
                            for (src, dest) in col_src.into_col_major_elems().zip(col_dest.into_col_major_elems()) {
                                assert_eq!(src.to_bits(), dest.to_bits());
                            }
                        }
                    }

                    // Signature of op asserts that output matrix has the right dimension at compile time
                    fn [< test_rows $dim1 _of_mat $dim2 x $dim3 >](
                        mat: Matrix<$dim2, $dim3>,
                        start_row: usize,
                        op: impl FnOnce() -> Matrix<$dim1, $dim3> + UnwindSafe
                    ) -> bool {
                        [< test_block $dim1 x $dim3 _of_mat $dim2 x $dim3 >](mat, start_row, 0, op)
                    }

                    #[quickcheck]
                    fn [< rows $dim1 _of_mat $dim2 x $dim3 >](
                        mat: Matrix<$dim2, $dim3>,
                        start_row: usize
                    ) -> bool {
                        [< test_rows $dim1 _of_mat $dim2 x $dim3 >](
                            mat,
                            start_row,
                            || mat.rows::<$dim1>(start_row)
                        )
                    }

                    #[quickcheck]
                    fn [< top $dim1 _of_mat $dim2 x $dim3 >](
                        mat: Matrix<$dim2, $dim3>,
                    ) -> bool {
                        [< test_rows $dim1 _of_mat $dim2 x $dim3 >](
                            mat,
                            0,
                            || mat.top_rows::<$dim1>()
                        )
                    }

                    #[quickcheck]
                    fn [< bottom $dim1 _of_mat $dim2 x $dim3 >](
                        mat: Matrix<$dim2, $dim3>,
                    ) -> bool {
                        [< test_rows $dim1 _of_mat $dim2 x $dim3 >](
                            mat,
                            ($dim2 as usize).saturating_sub($dim1),
                            || mat.bottom_rows::<$dim1>()
                        )
                    }

                    // Signature of op asserts that output matrix has the right dimension at compile time
                    fn [< test_cols $dim1 _of_mat $dim2 x $dim3 >](
                        mat: Matrix<$dim2, $dim3>,
                        start_col: usize,
                        op: impl FnOnce() -> Matrix<$dim2, $dim1> + UnwindSafe
                    ) -> bool {
                        [< test_block $dim2 x $dim1 _of_mat $dim2 x $dim3 >](mat, 0, start_col, op)
                    }

                    #[quickcheck]
                    fn [< cols $dim1 _of_mat $dim2 x $dim3 >](
                        mat: Matrix<$dim2, $dim3>,
                        start_col: usize
                    ) -> bool {
                        [< test_cols $dim1 _of_mat $dim2 x $dim3 >](
                            mat,
                            start_col,
                            || mat.cols::<$dim1>(start_col)
                        )
                    }

                    #[quickcheck]
                    fn [< left $dim1 _of_mat $dim2 x $dim3 >](
                        mat: Matrix<$dim2, $dim3>,
                    ) -> bool {
                        [< test_cols $dim1 _of_mat $dim2 x $dim3 >](
                            mat,
                            0,
                            || mat.left_cols::<$dim1>()
                        )
                    }

                    #[quickcheck]
                    fn [< right $dim1 _of_mat $dim2 x $dim3 >](
                        mat: Matrix<$dim2, $dim3>,
                    ) -> bool {
                        [< test_cols $dim1 _of_mat $dim2 x $dim3 >](
                            mat,
                            ($dim3 as usize).saturating_sub($dim1),
                            || mat.right_cols::<$dim1>()
                        )
                    }
                }

                // FIXME: Had to disable some test configurations to keep compile times reasonable.
                //        Do a measureme run to make sure we understand what's going on.
                generate_tests!(
                    ($dim1, $dim2, $dim3, 1),
                    ($dim1, $dim2, $dim3, 2),
                    ($dim1, $dim2, $dim3, 3),
                    // ($dim1, $dim2, $dim3, 4),
                    // ($dim1, $dim2, $dim3, 5),
                    // ($dim1, $dim2, $dim3, 6),
                    // ($dim1, $dim2, $dim3, 7),
                    ($dim1, $dim2, $dim3, 8)
                );
            )*
        };
        ($(($dim1:literal, $dim2:literal, $dim3:literal, $dim4:literal)),*) => {
            $(
                paste! {
                    // Signature of op asserts that output matrix has the right dimension at compile time
                    fn [< test_block $dim1 x $dim2 _of_mat $dim3 x $dim4 >](
                        mat: Matrix<$dim3, $dim4>,
                        start_row: usize,
                        start_col: usize,
                        op: impl FnOnce() -> Matrix<$dim1, $dim2> + UnwindSafe
                    ) -> bool {
                        if $dim1 <= ($dim3 as usize).saturating_sub(start_row)
                            && $dim2 <= ($dim4 as usize).saturating_sub(start_col)
                        {
                            for (src, dest) in
                                mat.into_iter().skip(start_col).take($dim2)
                                   .flat_map(|col| col.into_col_major_elems().skip(start_row).take($dim1))
                                   .zip(op().into_col_major_elems())
                            {
                                assert_eq!(src.to_bits(), dest.to_bits());
                            }
                            true
                        } else {
                            panics(op)
                        }
                    }

                    #[quickcheck]
                    fn [< block $dim1 x $dim2 _of_mat $dim3 x $dim4 >](
                        mat: Matrix<$dim3, $dim4>,
                        start_row: usize,
                        start_col: usize
                    ) -> bool {
                        [< test_block $dim1 x $dim2 _of_mat $dim3 x $dim4 >](
                            mat,
                            start_row,
                            start_col,
                            || mat.block::<$dim1, $dim2>(start_row, start_col)
                        )
                    }

                    #[quickcheck]
                    fn [< topleft $dim1 x $dim2 _of_mat $dim3 x $dim4 >](
                        mat: Matrix<$dim3, $dim4>
                    ) -> bool {
                        [< test_block $dim1 x $dim2 _of_mat $dim3 x $dim4 >](
                            mat,
                            0,
                            0,
                            || mat.top_left_corner::<$dim1, $dim2>()
                        )
                    }

                    #[quickcheck]
                    fn [< topright $dim1 x $dim2 _of_mat $dim3 x $dim4 >](
                        mat: Matrix<$dim3, $dim4>
                    ) -> bool {
                        [< test_block $dim1 x $dim2 _of_mat $dim3 x $dim4 >](
                            mat,
                            0,
                            ($dim4 as usize).saturating_sub($dim2),
                            || mat.top_right_corner::<$dim1, $dim2>()
                        )
                    }

                    #[quickcheck]
                    fn [< bottomleft $dim1 x $dim2 _of_mat $dim3 x $dim4 >](
                        mat: Matrix<$dim3, $dim4>
                    ) -> bool {
                        [< test_block $dim1 x $dim2 _of_mat $dim3 x $dim4 >](
                            mat,
                            ($dim3 as usize).saturating_sub($dim1),
                            0,
                            || mat.bottom_left_corner::<$dim1, $dim2>()
                        )
                    }

                    #[quickcheck]
                    fn [< bottomright $dim1 x $dim2 _of_mat $dim3 x $dim4 >](
                        mat: Matrix<$dim3, $dim4>
                    ) -> bool {
                        [< test_block $dim1 x $dim2 _of_mat $dim3 x $dim4 >](
                            mat,
                            ($dim3 as usize).saturating_sub($dim1),
                            ($dim4 as usize).saturating_sub($dim2),
                            || mat.bottom_right_corner::<$dim1, $dim2>()
                        )
                    }
                }
            )*
        };
    }
    generate_tests!();
}
