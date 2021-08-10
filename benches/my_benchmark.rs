use criterion::{
    black_box, criterion_group, criterion_main, measurement::Measurement, BenchmarkGroup,
    Criterion, Throughput,
};
use more_asserts::*;
use rand::prelude::*;
use simd_tests::{Scalar, SquareMatrix, Vector};

#[derive(Copy, Clone, Debug)]
enum FigureOfMerit {
    InputBytes,
    OutputBytes,
    Muls,
    Adds,
}

fn benchmark_impl<Inputs: Clone, Output, M: Measurement>(
    group: &mut BenchmarkGroup<M>,
    figure_of_merit: FigureOfMerit,
    num_muls: usize,
    num_adds: usize,
    mut function: impl FnMut(Inputs) -> Output,
    inputs: Inputs,
    name: &str,
) {
    let output_bytes = std::mem::size_of::<Output>();
    let input_bytes = std::mem::size_of::<Inputs>();
    match figure_of_merit {
        FigureOfMerit::OutputBytes => {
            assert_gt!(output_bytes, 0);
            group.throughput(Throughput::Bytes(output_bytes as _));
        }
        FigureOfMerit::InputBytes => {
            if input_bytes == 0 {
                return;
            } else {
                group.throughput(Throughput::Bytes(input_bytes as _));
            }
        }
        FigureOfMerit::Muls => {
            if num_muls == 0 {
                return;
            } else {
                group.throughput(Throughput::Elements(num_muls as _));
            }
        }
        FigureOfMerit::Adds => {
            if num_adds == 0 {
                return;
            } else {
                group.throughput(Throughput::Elements(num_adds as _));
            }
        }
    };
    group.bench_function(name, |b| b.iter(|| function(black_box(inputs.clone()))));
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    for figure_of_merit in [
        FigureOfMerit::OutputBytes,
        FigureOfMerit::InputBytes,
        FigureOfMerit::Muls,
        FigureOfMerit::Adds,
    ] {
        let mut group = c.benchmark_group(format!("{:?}", figure_of_merit));
        macro_rules! benchmark {
            ($num_muls:expr, $num_adds:expr, $function:path $(, $input:ident)*) => {
                #[allow(unused_parens)]
                benchmark_impl(
                    &mut group,
                    figure_of_merit,
                    $num_muls,
                    $num_adds,
                    |($($input),*)| $function($($input),*),
                    ($($input),*),
                    stringify!($function)
                );
            };
        }
        macro_rules! generate_benchmarks {
            () => { generate_benchmarks!(1, 2, 3, 4, 5, 6, 7, 8); };
            ($($dim:literal),*) => {
                $(
                    // FIXME: Find a way to split this across multiple
                    //        compilation units
                    // FIXME: Iterate over benchmarks, then over dimensions

                    let dim_idx = rng.gen_range(0..$dim);
                    benchmark!(0, 0, Vector::<$dim>::unit, dim_idx);

                    benchmark!(0, 0, SquareMatrix::<$dim>::identity);

                    let vec_elems = rng.gen::<[Scalar; $dim]>().into_iter();
                    benchmark!(0, 0, Vector::<$dim>::from_col_major_elems, vec_elems);

                    let mat_elems = rng.gen::<[Scalar; $dim * $dim]>().into_iter();
                    benchmark!(0, 0, SquareMatrix::<$dim>::from_col_major_elems, mat_elems);

                    let v1 = rng.gen::<Vector<$dim>>();
                    let v2 = rng.gen::<Vector<$dim>>();
                    benchmark!(0, 0, Vector::<$dim>::cat, v1, v2);

                    let m1 = rng.gen::<SquareMatrix<$dim>>();
                    let m2 = rng.gen::<SquareMatrix<$dim>>();
                    benchmark!(0, 0, SquareMatrix::<$dim>::hcat, m1, m2);
                    benchmark!(0, 0, SquareMatrix::<$dim>::vcat, m1, m2);

                    // TODO: Benchmark a number of blocks

                    benchmark!(0, 0, Vector::<$dim>::transpose, v1);
                    benchmark!(0, 0, SquareMatrix::<$dim>::transpose, m1);

                    // TODO: Benchmark a number of matrix powers

                    benchmark!($dim, $dim-1, Vector::<$dim>::dot, v1, v2);
                    benchmark!($dim * $dim, $dim * $dim - 1, SquareMatrix::<$dim>::dot, m1, m2);

                    benchmark!($dim, $dim-1, Vector::<$dim>::squared_norm, v1);
                    benchmark!($dim, $dim-1, Vector::<$dim>::norm, v1);
                    benchmark!($dim * $dim, $dim * $dim - 1, SquareMatrix::<$dim>::squared_norm, m1);
                    benchmark!($dim * $dim, $dim * $dim - 1, SquareMatrix::<$dim>::norm, m1);

                    if $dim == 3 {
                        let v3_1 = rng.gen::<Vector<3>>();
                        let v3_2 = rng.gen::<Vector<3>>();
                        benchmark!(6, 3, Vector::<3>::cross, v3_1, v3_2);
                    }

                    benchmark!(0, $dim - 1, SquareMatrix::<$dim>::trace, m1);

                    // FIXME: Should have implementation-specific det, inverse
                    //        methods, so we can keep number of add/mul accurate
                    let dim_min_1_fact = (2..$dim).product::<usize>();
                    let dim_fact = dim_min_1_fact * $dim;
                    benchmark!(dim_fact, dim_fact - 1, SquareMatrix::<$dim>::det, m1);
                    let num_mat_elems = $dim * $dim;
                    benchmark!(
                        dim_fact + num_mat_elems * dim_min_1_fact,
                        dim_fact - 1 + num_mat_elems * (dim_min_1_fact - 1),
                        SquareMatrix::<$dim>::inverse,
                        m1
                    );

                    // TODO: Add more ops, those that go by traits
                )*
            }
        }
        generate_benchmarks!();
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
