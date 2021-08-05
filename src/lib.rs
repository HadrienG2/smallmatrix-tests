#![feature(const_generics, const_evaluatable_checked)] // Needed for hcat and vcat

use core::{
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
};
use more_asserts::*;

// TODO: Add a layer of genericity over scalar type later on
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

// NOTE: Can't derive Default as it isn't implemented for all arrays
impl<const ROWS: usize, const COLS: usize> Default for Matrix<ROWS, COLS> {
    fn default() -> Self {
        Self([[Scalar::default(); ROWS]; COLS])
    }
}

// TODO: Implement useful num_traits (at least Zero), rand, and quickcheck's Arbitrary

impl<const ROWS: usize, const COLS: usize> Matrix<ROWS, COLS> {
    /// Construction from column-major data
    fn from_col_major_elems<T: IntoIterator<Item = Scalar>>(input: T) -> Self {
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
    fn into_col_major_elems(self) -> impl Iterator<Item = Scalar> {
        self.0.into_iter().flat_map(|col| col.into_iter())
    }

    /// Iterate over inner column-major data
    #[allow(unused)]
    fn col_major_elems(&self) -> impl Iterator<Item = &Scalar> {
        self.0.iter().flat_map(|col| col.iter())
    }

    /// Iterate over inner column-major data, allowing mutation
    fn col_major_elems_mut(&mut self) -> impl Iterator<Item = &mut Scalar> {
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

// TODO: Projection and rotation matrices

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

impl<const DIM: usize> ColVector<DIM> {
    // Extract a segment of a vector
    pub fn segment<const SUB_DIM: usize>(self, start_idx: usize) -> ColVector<SUB_DIM> {
        assert_le!(SUB_DIM, DIM, "Segment dimension is out of bounds");
        assert_le!(start_idx, DIM - SUB_DIM, "Start index is out of bounds");
        ColVector::<SUB_DIM>::from_col_major_elems(
            self.into_col_major_elems().skip(start_idx).take(SUB_DIM),
        )
    }

    // Extract first elements of a vector
    pub fn head<const SUB_DIM: usize>(self) -> ColVector<SUB_DIM> {
        self.segment::<SUB_DIM>(0)
    }

    // Extract last elements of a vector
    pub fn tail<const SUB_DIM: usize>(self) -> ColVector<SUB_DIM> {
        self.segment::<SUB_DIM>(DIM - SUB_DIM)
    }
}

// TODO: Matrix slicing (block, (top|bottom)?rows, (left|right)?cols, (top|bottom)(left|right)corner)

impl<const ROWS: usize, const COLS: usize> Matrix<ROWS, COLS> {
    /// Matrix transpose (TODO: avoid using expression templates)
    pub fn transpose(self) -> Matrix<COLS, ROWS> {
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
        iter.fold(Self::default(), |acc, x| acc + x)
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

impl<const DIM: usize> ColVector<DIM> {
    /// Dot product
    pub fn dot(self, rhs: Self) -> Scalar {
        (self.transpose() * rhs).into()
    }

    // Norm
    pub fn norm(self) -> Scalar {
        self.dot(self).sqrt()
    }
}

impl ColVector<3> {
    /// Cross product
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

    // TODO: Matrix determinant (needs hcat and bottom(left|right)corner)
    // TODO: Matrix inversion (needs determinant)
}

// TODO: Maybe linear solver, Gram-Schmidt, eigenvalues and eigenvectors...
// TODO: Special matrices (diagonal, tridiagonal, upper/lower triangular,
//       symmetric, hermitic) and unit vectors, with specialized handling.

/* // SIMD processing parameters
pub type Simd<const LANES: usize> = SimdF64<LANES>;
pub type Mask<const LANES: usize> = Mask64<LANES>;

const fn simd_lanes(simd_bits: usize) -> usize {
    // assert_eq!(simd_bits % 8, 0);
    let simd_bytes = simd_bits / 8;
    let scalar_bytes = std::mem::size_of::<Scalar>();
    // assert_gt!(simd_bytes, scalar_bytes);
    simd_bytes / scalar_bytes
}
pub const SSE: usize = simd_lanes(128);
pub const AVX: usize = simd_lanes(256);
// pub const AVX512: usize = simd_lanes(512);
pub const WIDEST: usize = AVX; /* AVX512 */

// Divide X by Y, rounding upwards
const fn div_round_up(num: usize, denom: usize) -> usize {
    num / denom + (num % denom != 0) as usize
}

// Set up a storage buffer in a SIMD-friendly layout
pub fn allocate_simd<const LANES: usize>(min_size: usize) -> Vec<Simd<LANES>>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let num_vecs = div_round_up(min_size, LANES);
    vec![[0.0; LANES].into(); num_vecs]
}

// Convert existing scalar data into a SIMD-friendly layout
pub fn simdize<const LANES: usize>(input: &[Scalar]) -> Vec<Simd<LANES>>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    // Collect all complete SIMD chunks from input
    let mut chunks = input
        .array_chunks::<LANES>()
        .copied()
        .map(Simd::<LANES>::from_array)
        .collect::<Vec<_>>();

    // If there is a remaining incomplete chunk, zero-pad it
    debug_assert_ge!(input.len(), chunks.len() * LANES);
    if input.len() > chunks.len() * LANES {
        let remainder = &input[chunks.len() * LANES..];
        let mut last_chunk = [0.0; LANES];
        last_chunk[..input.len() % LANES].copy_from_slice(remainder);
        chunks.push(Simd::<LANES>::from_array(last_chunk));
    }
    chunks
}

// Degrade SIMD data into scalar data
pub fn scalarize<const LANES: usize>(slice: &[Simd<LANES>]) -> &[Scalar]
where
    LaneCount<LANES>: SupportedLaneCount,
{
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const Scalar, LANES * slice.len()) }
}

pub fn scalarize_mut<const LANES: usize>(slice: &mut [Simd<LANES>]) -> &mut [Scalar]
where
    LaneCount<LANES>: SupportedLaneCount,
{
    unsafe {
        std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut Scalar, LANES * slice.len())
    }
}

// Iterator summation algorithm that minimizes dependency chain length
//
// Inlining is very important, and therefore forced, as we want the summation to
// be carried out in registers whenever possible.
//
#[inline(always)]
fn smart_sum<T: Copy + Default + AddAssign<T>, I: Iterator<Item = T>, const KERNEL_LEN: usize>(
    products: I,
) -> T {
    // Collect the inputs
    assert_eq!(products.size_hint().0, KERNEL_LEN);
    let mut buffer = [T::default(); KERNEL_LEN];
    for (dest, src) in buffer.iter_mut().zip(products) {
        *dest = src;
    }

    // Perform the summation using a binary tree algorithm
    let mut stride = KERNEL_LEN.next_power_of_two() / 2;
    while stride > 0 {
        for i in 0..stride.min(KERNEL_LEN - stride) {
            buffer[i] += buffer[i + stride];
        }
        stride /= 2;
    }
    buffer[0]
}

// Perform convolution using a scalar algorithm, let compiler autovectorize it
//
// Inlining is very important, and therefore forced, as the compiler can produce
// much better code when it knows the convolution kernel.
//
#[inline(always)]
pub fn convolve_autovec<const LANES: usize, const KERNEL_LEN: usize, const SMART_SUM: bool>(
    input: &[Simd<LANES>],
    kernel: &[Scalar; KERNEL_LEN],
    output: &mut [Simd<LANES>],
) where
    LaneCount<LANES>: SupportedLaneCount,
{
    let input = scalarize(input);
    let output = scalarize_mut(output);
    assert_ge!(input.len(), kernel.len());
    assert_ge!(output.len(), input.len() - kernel.len() + 1);
    for (data, output) in input.array_windows::<KERNEL_LEN>().zip(output.iter_mut()) {
        let products = data.iter().zip(kernel.iter()).map(|(&x, &k)| x * k);
        *output = if SMART_SUM {
            smart_sum::<_, _, KERNEL_LEN>(products)
        } else {
            products.sum()
        };
    }
}

// Manually vectorized version of the autovectorized code
//
// Slightly faster because the autovectorizer failed to take advantage of data
// alignment, which the manually vectorized version does leverage.
//
struct Manual<const LANES: usize>;

macro_rules! gen_manual_convolve {
    ($($lanes:expr),*) => {
        $( paste!{
            impl Manual<$lanes> {
                #[inline(always)]
                fn convolve<const KERNEL_LEN: usize, const SMART_SUM: bool>(
                    input: &[Simd<$lanes>],
                    kernel: &[Scalar; KERNEL_LEN],
                    output: &mut [Simd<$lanes>]
                ) {
                    // Validate inputs
                    assert_ge!(input.len()*$lanes, KERNEL_LEN, "Convolution input is smaller than convolution kernel length");
                    assert_ge!(output.len()*$lanes, input.len()*$lanes - KERNEL_LEN + 1, "Convolution output buffer is smaller than output length");

                    // Perform vectorized convolution
                    let kernel_len_vecs = div_round_up(KERNEL_LEN-1, $lanes) + 1;
                    for (out_vec, in_aligned_vecs) in output.iter_mut().zip(input.windows(kernel_len_vecs)) {
                        // Collect unaligned convolution inputs
                        let in_vecs_iter = scalarize(in_aligned_vecs)
                            .array_windows::<$lanes>()
                            .copied()
                            .map(Simd::<$lanes>::from_array);

                        // Compute products with kernel
                        let products = kernel.iter().zip(in_vecs_iter).map(|(&kernel_elem, in_vec)| {
                            in_vec * Simd::<$lanes>::splat(kernel_elem)
                        });

                        // Sum inputs using an algorithm of variable cleverness
                        *out_vec = if SMART_SUM {
                            smart_sum::<_, _, KERNEL_LEN>(products)
                        } else {
                            products.sum()
                        };
                    }

                    // Compute tail elements (if any) using scalar algorithm
                    let num_out_vecs = output.len();
                    let num_ins_windows = input.windows(kernel_len_vecs).count();
                    debug_assert_ge!(num_out_vecs, num_ins_windows);
                    if num_out_vecs > num_ins_windows {
                        convolve_autovec::<$lanes, KERNEL_LEN, false>(&input[num_ins_windows..], kernel, &mut output[num_ins_windows..]);
                    }
                }
            }
        } )*
    };
}

gen_manual_convolve!(4, 8 /*, 16*/);

// Element-shifting interpolation between two consecutive vectors
// TODO: Should codegen this using macros

#[allow(unused)]
fn simd_shift_2(left: Simd<2>, right: Simd<2>, shift: usize) -> Simd<2> {
    match shift {
        0 => left,
        1 => left.shuffle::<{ [1, 2] }>(right),
        2 => right,
        _ => unreachable!("Bad shift value"),
    }
}

#[allow(unused)]
fn simd_shift_4(left: Simd<4>, right: Simd<4>, shift: usize) -> Simd<4> {
    match shift {
        0 => left,
        1 => left.shuffle::<{ [1, 2, 3, 4] }>(right),
        2 => left.shuffle::<{ [2, 3, 4, 5] }>(right),
        3 => left.shuffle::<{ [3, 4, 5, 6] }>(right),
        4 => right,
        _ => unreachable!("Bad shift value"),
    }
}

fn simd_shift_8(left: Simd<8>, right: Simd<8>, shift: usize) -> Simd<8> {
    match shift {
        0 => left,
        1 => left.shuffle::<{ [1, 2, 3, 4, 5, 6, 7, 8] }>(right),
        2 => left.shuffle::<{ [2, 3, 4, 5, 6, 7, 8, 9] }>(right),
        3 => left.shuffle::<{ [3, 4, 5, 6, 7, 8, 9, 10] }>(right),
        4 => left.shuffle::<{ [4, 5, 6, 7, 8, 9, 10, 11] }>(right),
        5 => left.shuffle::<{ [5, 6, 7, 8, 9, 10, 11, 12] }>(right),
        6 => left.shuffle::<{ [6, 7, 8, 9, 10, 11, 12, 13] }>(right),
        7 => left.shuffle::<{ [7, 8, 9, 10, 11, 12, 13, 14] }>(right),
        8 => right,
        _ => unreachable!("Bad shift value"),
    }
}

// TODO: Should add a _16 version for AVX-512

// Variant of the convolution algorithm that minimizes the number of loads and
// makes the loads as efficient as possible, at the cost of using a lot more
// shuffles. On Zen 2, this is not a good tradeoff.
struct MinimalLoads<const LANES: usize>;

macro_rules! gen_minimal_loads_convolve {
    ($($lanes:expr),*) => {
        $( paste!{
            impl MinimalLoads<$lanes> {
                #[inline(always)]
                fn convolve<const KERNEL_LEN: usize, const SMART_SUM: bool>(
                    input: &[Simd<$lanes>],
                    kernel: &[Scalar; KERNEL_LEN],
                    output: &mut [Simd<$lanes>]
                ) {
                    // Validate inputs
                    assert_ge!(input.len()*$lanes, KERNEL_LEN, "Convolution input is smaller than convolution kernel length");
                    assert_ge!(output.len()*$lanes, input.len()*$lanes - KERNEL_LEN + 1, "Convolution output buffer is smaller than output length");

                    // Perform vectorized convolution
                    let kernel_len_vecs = div_round_up(KERNEL_LEN-1, $lanes) + 1;
                    for (out_vec, in_aligned_vecs) in output.iter_mut().zip(input.windows(kernel_len_vecs)) {
                        let products = kernel.iter().enumerate().map(|(shift, &kernel_elem)| {
                            let vec_shift = shift / $lanes;
                            let elem_shift = shift % $lanes;
                            let left_vec = in_aligned_vecs[vec_shift];
                            let in_vec = if elem_shift == 0 {
                                left_vec
                            } else {
                                let right_vec = in_aligned_vecs[vec_shift + 1];
                                [< simd_shift_ $lanes >](left_vec, right_vec, elem_shift)
                            };
                            in_vec * Simd::<$lanes>::splat(kernel_elem)
                        });
                        *out_vec = if SMART_SUM {
                            smart_sum::<_, _, KERNEL_LEN>(products)
                        } else {
                            products.sum()
                        };
                    }

                    // Compute tail elements (if any) using scalar algorithm
                    let num_out_vecs = output.len();
                    let num_ins_windows = input.windows(kernel_len_vecs).count();
                    debug_assert_ge!(num_out_vecs, num_ins_windows);
                    if num_out_vecs > num_ins_windows {
                        convolve_autovec::<$lanes, KERNEL_LEN, false>(&input[num_ins_windows..], kernel, &mut output[num_ins_windows..]);
                    }
                }
            }
        } )*
    };
}

gen_minimal_loads_convolve!(4, 8 /*, 16*/);

// A shuffle pattern that is efficient on x86 CPUs with SSE/AVX
// TODO: Should codegen this using macros

#[allow(unused)]
fn simd_shuf2_4(base: Simd<4>, shift4: Simd<4>) -> Simd<4> {
    // base   is [ 0 1 2 3 ]
    // shift4 is [ 4 5 6 7 ]
    // output is [ 2 3 4 5 ]
    base.shuffle::<{ [2, 3, 4, 5] }>(shift4)
}

fn simd_shuf2_8(base: Simd<8>, shift4: Simd<8>) -> Simd<8> {
    // base   is [ 0 1 2 3 | 4 5 6 7 ]
    // shift4 is [ 4 5 6 7 | 8 9 10 11 ]
    // output is [ 2 3 4 5 | 6 7 8 9 ]
    base.shuffle::<{ [2, 3, 8, 9, 6, 7, 12, 13] }>(shift4)
}

// TODO: Should add a _16 version for AVX-512

// Variant of the convolution algorithm that tries to strike a balance between
// reducing the number of unaligned loads (which bound the performance of the
// naive algorithm) and adding as few shuffles as possible (since these compete
// with adds an muls for SIMD execution ports on Zen2).
//
// Interestingly enough, this is still not beneficial, because it results in
// turning memory operands into MOVs, which also has a cost.
//
struct Shuf2Loadu<const LANES: usize>;

macro_rules! gen_shuf2_loadu_convolve {
    ($($lanes:expr),*) => {
        $( paste!{
            impl Shuf2Loadu<$lanes> {
                #[inline(always)]
                fn convolve<const KERNEL_LEN: usize, const SMART_SUM: bool>(
                    input: &[Simd<$lanes>],
                    kernel: &[Scalar; KERNEL_LEN],
                    output: &mut [Simd<$lanes>]
                ) {
                    // Validate inputs
                    assert_ge!(input.len()*$lanes, KERNEL_LEN, "Convolution input is smaller than convolution kernel length");
                    assert_ge!(output.len()*$lanes, input.len()*$lanes - KERNEL_LEN + 1, "Convolution output buffer is smaller than output length");

                    // Perform vectorized convolution
                    let kernel_len_vecs = div_round_up(KERNEL_LEN-1, $lanes) + 1;
                    for (out_vec, in_aligned_vecs) in output.iter_mut().zip(input.windows(kernel_len_vecs)) {
                        // Collect unaligned convolution inputs
                        let in_vecs_iter = scalarize(in_aligned_vecs)
                            .array_windows::<$lanes>()
                            .copied()
                            .map(Simd::<$lanes>::from_array);
                        let mut in_vecs = [Default::default(); KERNEL_LEN];
                        for (in_vec, data) in in_vecs.iter_mut().zip(in_vecs_iter) {
                            *in_vec = data;
                        }

                        // Compute products with kernel, only using a subset of inputs
                        let products = kernel.iter().enumerate().map(|(idx, &kernel_elem)| {
                            let elem_idx = idx % $lanes;
                            let in_vec = if (elem_idx == 2 || elem_idx == 3) && idx < KERNEL_LEN-2 {
                                // Inputs 2 and 3 are interpolated from inputs (0, 1, 4, 5) using shuffles
                                [< simd_shuf2_ $lanes >](in_vecs[idx-2], in_vecs[idx+2])
                            } else {
                                // Other inputs are accessed using unaligned loads
                                in_vecs[idx]
                            };
                            in_vec * Simd::<$lanes>::splat(kernel_elem)
                        });

                        // Sum inputs using an algorithm of variable cleverness
                        *out_vec = if SMART_SUM {
                            smart_sum::<_, _, KERNEL_LEN>(products)
                        } else {
                            products.sum()
                        };
                    }

                    // Compute tail elements (if any) using scalar algorithm
                    let num_out_vecs = output.len();
                    let num_ins_windows = input.windows(kernel_len_vecs).count();
                    debug_assert_ge!(num_out_vecs, num_ins_windows);
                    if num_out_vecs > num_ins_windows {
                        convolve_autovec::<$lanes, KERNEL_LEN, false>(&input[num_ins_windows..], kernel, &mut output[num_ins_windows..]);
                    }
                }
            }
        } )*
    };
}

gen_shuf2_loadu_convolve!(4, 8 /*, 16*/);

// TODO: On Intel processors, FMA might perform better than mul + add (while the
//       reverse is expected on AMD Zen processors)

// Examples of usage (use cargo asm to show assembly)
pub const FINITE_DIFF: [Scalar; 2] = [-1.0, 1.0];
pub const SHARPEN3: [Scalar; 3] = [-0.5, 2.0, -0.5];
pub const SMOOTH5: [Scalar; 5] = [0.1, 0.2, 0.4, 0.2, 0.1];
pub const ANTISYM8: [Scalar; 8] = [0.1, -0.2, 0.4, -0.8, 0.8, -0.4, 0.2, -0.1];
// TODO: For AVX-512, could also test a 16-wide kernel

// Generate example implementations using above kernels
macro_rules! generate_examples {
    ($impl:ident) => {
        generate_examples!($impl, SSE);
        generate_examples!($impl, AVX);
        // generate_examples!($impl, AVX512);
    };
    ($impl:ident, $width:ident) => {
        generate_examples!($impl, $width, FINITE_DIFF);
        generate_examples!($impl, $width, SHARPEN3);
        generate_examples!($impl, $width, SMOOTH5);
        generate_examples!($impl, $width, ANTISYM8);
    };
    ($impl:ident, $width:ident, $kernel:ident) => {
        generate_examples!($impl, $width, $kernel, false, basic);
        generate_examples!($impl, $width, $kernel, true, smart);
    };
    (autovec, $width:ident, $kernel:ident, $smart:expr, $suffix:ident) => {
        paste!{
            #[inline(never)]
            pub fn [<$kernel:lower _autovec_ $suffix _ $width:lower>](input: &[Simd<$width>], output: &mut [Simd<$width>]) {
                convolve_autovec::<$width, { $kernel.len() }, $smart>(input, &$kernel, output);
            }
        }
    };
    ($simd_impl:ident, $width:ident, $kernel:ident, $smart:expr, $suffix:ident) => {
        paste!{
            #[inline(never)]
            pub fn [<$kernel:lower _ $simd_impl:snake:lower _ $suffix _ $width:lower>](input: &[Simd<$width>], output: &mut [Simd<$width>]) {
                $simd_impl::<$width>::convolve::<{ $kernel.len() }, $smart>(input, &$kernel, output);
            }
        }
    }
}

// No need to support multiple vector widths for autovectorized version, it will
// use max-width AVX with unaligned operands anyway (and that's okay)
generate_examples!(autovec, WIDEST);
generate_examples!(Manual, WIDEST);
generate_examples!(Shuf2Loadu, WIDEST);
generate_examples!(MinimalLoads, WIDEST); */

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;

    /* // Test division with upwards rounding
    #[quickcheck]
    fn div_round_up(num: usize, denom: usize) -> TestResult {
        if denom == 0 {
            return TestResult::discard();
        }
        let result = super::div_round_up(num, denom);
        if num % denom == 0 {
            TestResult::from_bool(result == num / denom)
        } else {
            TestResult::from_bool(result == num / denom + 1)
        }
    }

    // Test SIMD-friendly allocation
    fn allocate_simd<const LANES: usize>(size: u16) -> bool
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        let size = size as usize;
        let vec = super::allocate_simd::<LANES>(size);
        vec.len() == super::div_round_up(size, LANES)
    }

    #[quickcheck]
    fn allocate_simd_sse(size: u16) -> bool {
        allocate_simd::<SSE>(size)
    }

    #[quickcheck]
    fn allocate_simd_avx(size: u16) -> bool {
        allocate_simd::<AVX>(size)
    }

    // Test conversion of scalar data to SIMD data and back
    fn simdize_scalarize<const LANES: usize>(input: &[Scalar]) -> bool
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        let output_simd = super::simdize::<LANES>(input);
        assert_eq!(output_simd.len(), super::div_round_up(input.len(), LANES));
        let output_scalar = super::scalarize(&output_simd);
        for (input_elem, output_elem) in input.iter().zip(output_scalar.iter()) {
            if input_elem.is_nan() {
                assert!(output_elem.is_nan());
            } else {
                assert_eq!(input_elem, output_elem);
            }
        }
        for tail_elem in &output_scalar[input.len()..] {
            assert_eq!(*tail_elem, 0.0);
        }
        true
    }

    #[quickcheck]
    fn simdize_scalarize_sse(input: Vec<Scalar>) -> bool {
        simdize_scalarize::<SSE>(&input)
    }

    #[quickcheck]
    fn simdize_scalarize_avx(input: Vec<Scalar>) -> bool {
        simdize_scalarize::<AVX>(&input)
    }

    // Generic test of convolution implementations
    #[inline(always)]
    fn convolve<
        Convolution: FnOnce(&[Simd<LANES>], &[Scalar; KERNEL_LEN], &mut [Simd<LANES>]),
        const LANES: usize,
        const KERNEL_LEN: usize,
    >(
        convolution: Convolution,
        input: Vec<Scalar>,
        kernel: &[Scalar; KERNEL_LEN],
    ) -> TestResult
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        // Reject unreasonable test inputs
        if input.len() < KERNEL_LEN || !input.iter().copied().all(Scalar::is_normal) {
            return TestResult::discard();
        }

        // Normalize input magnitude to [eps, 1] range for easier error analysis
        let input_magnitude = input
            .iter()
            .fold(Scalar::MIN_POSITIVE, |acc, x| acc.max(x.abs()));
        let input = input
            .into_iter()
            .map(|x| {
                let mut normalized = x / input_magnitude;
                if normalized.abs() < Scalar::EPSILON {
                    normalized = Scalar::EPSILON.copysign(x);
                }
                normalized
            })
            .collect::<Vec<_>>();

        // Prepare and perform the convolution
        let input_simd = super::simdize::<LANES>(&input);
        let output_len = input_simd.len() * LANES - KERNEL_LEN + 1;
        let mut output_simd = super::allocate_simd::<LANES>(output_len);
        convolution(&input_simd, &kernel, &mut output_simd);

        // Check convolution results against a basic reference implementation
        let output = scalarize(&output_simd);
        for (output_idx, (out, ins)) in output
            .into_iter()
            .zip(input.array_windows::<KERNEL_LEN>())
            .enumerate()
        {
            let expected = ins
                .iter()
                .zip(kernel.iter())
                .map(|(&x, &k)| x * k)
                .sum::<Scalar>();
            assert_le!(
                (*out - expected).abs(),
                2.0 * Scalar::EPSILON,
                "At output index {}/{}, fed inputs {:?} (from input indices {}->{}/{}) into kernel {:?} and got output {} instead of expected {}",
                output_idx,
                output_len-1,
                ins,
                output_idx,
                output_idx + KERNEL_LEN-1,
                input.len()-1,
                kernel,
                out,
                expected
            );
        }
        TestResult::passed()
    }

    // Generate tests for a given convolution implementation, for all SIMD
    // widths, example kernels, and summation algorithms
    macro_rules! generate_tests {
        ($impl:ident) => {
            generate_tests!($impl, SSE);
            generate_tests!($impl, AVX);
            // generate_tests!($impl, AVX512);
        };
        ($impl:ident, $width:ident) => {
            generate_tests!($impl, $width, FINITE_DIFF);
            generate_tests!($impl, $width, SHARPEN3);
            generate_tests!($impl, $width, SMOOTH5);
            generate_tests!($impl, $width, ANTISYM8);
        };
        ($impl:ident, $width:ident, $kernel:ident) => {
            generate_tests!($impl, $width, $kernel, false, basic);
            generate_tests!($impl, $width, $kernel, true, smart);
        };
        (autovec, $width:ident, $kernel:ident, $smart:expr, $suffix:ident) => {
            paste!{
                #[quickcheck]
                fn [<$kernel:lower _autovec_ $suffix _ $width:lower>](input: Vec<Scalar>) -> TestResult {
                    convolve(
                        super::convolve_autovec::<$width, { $kernel.len() }, $smart>,
                        input,
                        &$kernel,
                    )
                }
            }
        };
        ($simd_impl:ident, $width:ident, $kernel:ident, $smart:expr, $suffix:ident) => {
            paste!{
                #[quickcheck]
                fn [<$kernel:lower _ $simd_impl:snake:lower _ $suffix _ $width:lower>](input: Vec<Scalar>) -> TestResult {
                    convolve(
                        super::$simd_impl::<$width>::convolve::<{ $kernel.len() }, $smart>,
                        input,
                        &$kernel,
                    )
                }
            }
        }
    }
    generate_tests!(autovec);
    generate_tests!(Manual);
    generate_tests!(MinimalLoads);
    generate_tests!(Shuf2Loadu); */
}
