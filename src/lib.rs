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
        {
            let mut targets = result.col_major_elems_mut();
            let mut sources = input.into_iter();
            loop {
                match (targets.next(), sources.next()) {
                    (Some(target), Some(source)) => *target = source,
                    (None, None) => break,
                    (Some(_), None) => panic!("Too few elements in input iterator"),
                    (None, Some(_)) => panic!("Too many elements in input iterator"),
                }
            }
        }
        result
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
#[derive(Clone, Debug)]
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
        // Code bloat optimization
        fn validate_block(
            start_row: usize,
            sub_rows: usize,
            rows: usize,
            start_col: usize,
            sub_cols: usize,
            cols: usize,
        ) {
            assert_le!(sub_rows, rows, "Block rows are out of bounds");
            assert_le!(start_row, rows - sub_rows, "Start row is out of bounds");
            assert_le!(sub_cols, cols, "Block cols are out of bounds");
            assert_le!(start_col, cols - sub_cols, "Start col is out of bounds");
        }
        validate_block(start_row, SUB_ROWS, ROWS, start_col, SUB_COLS, COLS);
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
        // the destination matrix uses every column from the source matrix, so
        // we can only start updating source columns once we're done computing
        // the entire matrix product
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
//
impl<const DIM: usize> SquareMatrix<DIM> {
    /// Raise a square matrix to an integer power
    pub fn pow<RHS: Into<usize>>(self, rhs: RHS) -> Self {
        <Self as Pow<RHS>>::pow(self, rhs)
    }
}

impl<const ROWS: usize, const COLS: usize> Matrix<ROWS, COLS> {
    /// Froebenius dot product
    pub fn dot(self, rhs: Self) -> Scalar {
        self.into_col_major_elems()
            .zip(rhs.into_col_major_elems())
            .map(|(x, y)| x * y)
            .sum()
    }

    /// Froebenius squared norm
    pub fn squared_norm(self) -> Scalar {
        self.dot(self)
    }

    /// Froebenius norm
    pub fn norm(self) -> Scalar {
        self.squared_norm().sqrt()
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

// Indexing tests are implemented as a unit test because they must have access
// to the implementation. An implementation in terms of the elements iterator
// would also be possible, but then the indexing and iterator test would
// mutually rely on their test subject's correctness
#[cfg(test)]
mod tests {
    use super::*;
    use paste::paste;
    use quickcheck_macros::quickcheck;

    fn test_index_mat<const ROWS: usize, const COLS: usize>(
        mut mat: Matrix<ROWS, COLS>,
        row: usize,
        col: usize,
    ) {
        if row >= ROWS || col >= COLS {
            assert!(std::panic::catch_unwind(move || &mat[(row, col)] as *const _).is_err());
            assert!(std::panic::catch_unwind(move || &mut mat[(row, col)] as *mut _).is_err());
        } else {
            assert_eq!(
                &mat.0[col][row] as *const Scalar,
                &mat[(row, col)] as *const Scalar
            );
            assert_eq!(
                &mut mat.0[col][row] as *mut Scalar,
                &mut mat[(row, col)] as *mut Scalar
            );
        }
    }

    fn test_index_vec<const DIM: usize>(mut vec: ColVector<DIM>, idx: usize) {
        if idx >= DIM {
            assert!(std::panic::catch_unwind(move || &vec[idx] as *const _).is_err());
            assert!(std::panic::catch_unwind(move || &mut vec[idx] as *mut _).is_err());
        } else {
            assert_eq!(&vec.0[0][idx] as *const Scalar, &vec[idx] as *const Scalar);
            assert_eq!(
                &mut vec.0[0][idx] as *mut Scalar,
                &mut vec[idx] as *mut Scalar
            );
        }
    }

    macro_rules! generate_tests {
        () => {
            generate_tests!(1, 2, 3, 4, 5, 6, 7, 8);
        };
        ($($dim:literal),*) => {
            $(
                paste! {
                    #[quickcheck]
                    fn [< index_vec $dim >](vec: ColVector<$dim>, idx: usize) {
                        test_index_vec::<$dim>(vec, idx);
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
                    fn [< index_mat $dim1 x $dim2 >](mat: Matrix<$dim1, $dim2>, row: usize, col: usize) {
                        test_index_mat::<$dim1, $dim2>(mat, row, col);
                    }
                }
            )*
        }
    }
    generate_tests!();
}
