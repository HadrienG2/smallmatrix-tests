use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use more_asserts::*;
use rand::prelude::*;
use simd_tests::{SquareMatrix, Vector};

#[derive(Debug)]
enum FigureOfMerit {
    InputBytes,
    OutputBytes,
    Muls,
    Adds,
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    for figure_of_merit in [
        FigureOfMerit::InputBytes,
        FigureOfMerit::OutputBytes,
        FigureOfMerit::Muls,
        FigureOfMerit::Adds,
    ] {
        let mut group = c.benchmark_group(format!("{:?}", figure_of_merit));
        // Measure throughput as bytes of input processed per second
        macro_rules! generate_benchmarks {
            () => { generate_benchmarks!(1, 2, 3, 4, 5, 6, 7, 8); };
            ($($dim:literal),*) => {
                $(
                    // FIXME: Specific to Vector<$dim>::unit", generalize to other ops
                    let output_bytes = std::mem::size_of::<Vector<$dim>>();
                    let input_bytes = std::mem::size_of::<usize>();
                    let num_muls = 0;
                    let num_adds = 0;

                    match figure_of_merit {
                        FigureOfMerit::OutputBytes => {
                            assert_gt!(output_bytes, 0);
                            group.throughput(Throughput::Bytes(output_bytes as _));
                        }
                        FigureOfMerit::InputBytes => {
                            if input_bytes == 0 {
                                continue;
                            } else {
                                group.throughput(Throughput::Bytes(input_bytes as _));
                            }
                        }
                        FigureOfMerit::Muls => {
                            if num_muls == 0 {
                                continue;
                            } else {
                                group.throughput(Throughput::Elements(num_muls as _));
                            }
                        }
                        FigureOfMerit::Adds => {
                            if num_adds == 0 {
                                group.throughput(Throughput::Elements(num_adds as _));
                            }
                        }
                    };

                    let input = rng.gen_range(0..$dim);
                    group.bench_function(&format!("Vector<{}>::unit()", $dim),
                        |b| { b.iter(|| Vector::<$dim>::unit(black_box(input))) }
                    );
                )*
            }
        }
        generate_benchmarks!();
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
