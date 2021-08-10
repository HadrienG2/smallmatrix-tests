#![feature(const_generics, const_evaluatable_checked)]

use criterion::{
    black_box, criterion_group, criterion_main, measurement::Measurement, BenchmarkGroup,
    Criterion, Throughput,
};
use paste::paste;
use rand::prelude::*;
use simd_tests::{Scalar, SquareMatrix, Vector};

// Enable calling a function with a tuple of arguments
trait CallWith<Tuple, Result> {
    fn call(&self, t: Tuple) -> Result;
}
//
macro_rules! impl_call_with {
    () => {
        impl_call_with!(
            (),
            (A),
            (A, B),
            (A, B, C),
            (A, B, C, D)
        );
    };
    (
        $(
            ( $( $tuple_elem:ident ),* )
        ),*
    ) => {
        $(
            impl<
                $( $tuple_elem, )*
                R,
                F: Fn( $( $tuple_elem ),* ) -> R
            > CallWith<( $( $tuple_elem, )* ), R> for F {
                #[allow(non_snake_case)]
                fn call(
                    &self,
                    ( $( $tuple_elem, )* ) : ( $( $tuple_elem, )* )
                ) -> R {
                    (*self)( $( $tuple_elem ),* )
                }
            }
        )*
    }
}
//
impl_call_with!();

// The number of operations carried out by a benchmark may either be constant
// or vary depending on the object's dimension
trait NumOps {
    fn num_ops(&self, dim: usize) -> usize;
}
//
impl NumOps for usize {
    fn num_ops(&self, _dim: usize) -> usize {
        *self
    }
}
//
impl<F: Fn(usize) -> usize> NumOps for F {
    fn num_ops(&self, dim: usize) -> usize {
        (*self)(dim)
    }
}

// TODO: Deduplicate these functions
fn bench_output_bytes<Inputs: Clone, Output, M: Measurement>(
    group: &mut BenchmarkGroup<M>,
    method: impl CallWith<Inputs, Output>,
    input: Inputs,
    name: &str,
) {
    let output_bytes = std::mem::size_of::<Output>();
    if output_bytes > 0 {
        group.throughput(Throughput::Bytes(output_bytes as _));
        group.bench_function(name, |b| b.iter(|| method.call(black_box(input.clone()))));
    }
}

fn bench_input_bytes<Inputs: Clone, Output, M: Measurement>(
    group: &mut BenchmarkGroup<M>,
    method: impl CallWith<Inputs, Output>,
    input: Inputs,
    name: &str,
) {
    let input_bytes = std::mem::size_of::<Inputs>();
    if input_bytes > 0 {
        group.throughput(Throughput::Bytes(input_bytes as _));
        group.bench_function(name, |b| b.iter(|| method.call(black_box(input.clone()))));
    }
}

fn bench_ops<Inputs: Clone, Output, M: Measurement>(
    group: &mut BenchmarkGroup<M>,
    method: impl CallWith<Inputs, Output>,
    input: Inputs,
    name: &str,
    num_ops: usize,
) {
    if num_ops > 0 {
        group.throughput(Throughput::Elements(num_ops as _));
        group.bench_function(name, |b| b.iter(|| method.call(black_box(input.clone()))));
    }
}

// Execute a certain statement for a set of supported benchmark dimensions,
// which are accessible to the statement as a constant called "DIM".
macro_rules! for_each_dim {
    ($action:stmt) => {
        for_each_dim!([1, 2, 3, 4, 5, 6, 7, 8], $action)
    };
    ([ $( $dim:literal ),* ], $action:stmt) => {
        $(
            {
                const DIM: usize = $dim;
                $action
            }
        )*
    }
}

// Generate a set of Criterion benchmarks for Vector/SquareMatrix methods
macro_rules! generate_benchmarks {
    (
        $(
            ($object:ty, $method:ident, $num_muls:expr, $num_adds:expr $(, $input_generator:ident)* )
        ),*
        $( , )?
    ) => {
        paste! {
            $(
                fn [< $object:lower _ $method >](c: &mut Criterion) {
                    let mut group = c.benchmark_group(format!("{}::{}", stringify!($object), stringify!($method)));
                    for_each_dim!(
                        bench_output_bytes(
                            &mut group,
                            $object::<DIM>::$method,
                            ( $( $input_generator::<DIM>(), )* ),
                            &format!("OutputBytes/{}D", DIM)
                        )
                    );
                    for_each_dim!(
                        bench_input_bytes(
                            &mut group,
                            $object::<DIM>::$method,
                            ( $( $input_generator::<DIM>(), )* ),
                            &format!("InputBytes/{}D", DIM)
                        )
                    );
                    for_each_dim!(
                        bench_ops(
                            &mut group,
                            $object::<DIM>::$method,
                            ( $( $input_generator::<DIM>(), )* ),
                            &format!("Muls/{}D", DIM),
                            NumOps::num_ops(&$num_muls, DIM)
                        )
                    );
                    for_each_dim!(
                        bench_ops(
                            &mut group,
                            $object::<DIM>::$method,
                            ( $( $input_generator::<DIM>(), )* ),
                            &format!("Adds/{}D", DIM),
                            NumOps::num_ops(&$num_adds, DIM)
                        )
                    );
                    group.finish();
                }
            )*
            criterion_group!(benches, $( [< $object:lower _ $method >] ),*);
            criterion_main!(benches);
        }
    };
}

// Benchmark input generators
fn coord_idx<const DIM: usize>() -> usize {
    rand::thread_rng().gen_range(0..DIM)
}
//
fn mat_elems_iter<const DIM: usize>() -> impl Iterator<Item = Scalar> + Clone
where
    [(); DIM * DIM]: ,
{
    rand::thread_rng().gen::<[Scalar; DIM * DIM]>().into_iter()
}
//
fn vector<const DIM: usize>() -> Vector<DIM>
where
    [(); DIM * 1]: ,
{
    rand::random()
}
//
fn matrix<const DIM: usize>() -> SquareMatrix<DIM>
where
    [(); DIM * DIM]: ,
{
    rand::random()
}

// Computational complexity calculations
fn factorial(x: usize) -> usize {
    (2..x).product()
}
//
fn det_muls(dim: usize) -> usize {
    factorial(dim)
}
//
fn det_adds(dim: usize) -> usize {
    det_muls(dim) - 1
}
//
fn num_mat_elems(dim: usize) -> usize {
    dim * dim
}

// Generate all the benchmarks
// FIXME: Find a way to split this across multiple compilation units
generate_benchmarks!(
    (Vector, unit, 0, 0, coord_idx),
    (SquareMatrix, identity, 0, 0),
    (SquareMatrix, from_col_major_elems, 0, 0, mat_elems_iter),
    (Vector, cat, 0, 0, vector, vector),
    (SquareMatrix, hcat, 0, 0, matrix, matrix),
    (SquareMatrix, vcat, 0, 0, matrix, matrix),
    // TODO: Benchmark a number of blocks
    (SquareMatrix, transpose, 0, 0, matrix),
    // TODO: Benchmark a number of matrix powers
    (Vector, dot, 0, 0, vector, vector),
    (Vector, squared_norm, 0, 0, vector),
    (Vector, norm, 0, 0, vector),
    // FIXME: Figure out a way to benchmark cross, which is restricted to Vector<3>
    (SquareMatrix, trace, 0, |dim| dim - 1, matrix),
    // FIXME: Should have implementation-specific det and inverse methods, so
    //        we can keep the number of add/mul accurate as impl evolves
    (SquareMatrix, det, det_muls, det_adds, matrix),
    (
        SquareMatrix,
        inverse,
        // In Cramer's rule, one computes one cofactor per matrix element, then
        // computes the matrix determinant and divide the matrix (ergo, every
        // matrix element) by said determinant.
        |dim| num_mat_elems(dim) * (det_muls(dim - 1) + 1) + det_muls(dim),
        |dim| num_mat_elems(dim) * det_adds(dim - 1) + det_adds(dim),
        matrix
    ),
    // TODO: Add more ops, those that go by traits
);
