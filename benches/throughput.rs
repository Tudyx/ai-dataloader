use ai_dataloader::indexable::DataLoader;
use ai_dataloader::{Dataset, GetSample, Len};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use ndarray::Array3;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

const NUM_CLASS: usize = 20;
const IMAGE_SIZE: usize = 50;
const DATASET_LEN: usize = 500;

/// Dataset that return the same random image each time.
pub struct RandomUnique {
    image: Array3<u8>,
}

impl Default for RandomUnique {
    fn default() -> Self {
        Self {
            image: Array3::random((IMAGE_SIZE, IMAGE_SIZE, 3), Uniform::new_inclusive(0, 255)),
        }
    }
}

impl Dataset for RandomUnique {}

impl Len for RandomUnique {
    fn len(&self) -> usize {
        50_000
    }
}

impl GetSample for RandomUnique {
    type Sample = (Array3<u8>, i32);

    fn get_sample(&self, index: usize) -> Self::Sample {
        (self.image.clone(), (index % NUM_CLASS) as i32)
    }
}

fn iter_all_dataset(loader: &DataLoader<RandomUnique>) -> usize {
    let mut num_sample = 0;
    for (_sample, label) in loader.iter() {
        num_sample += label.len();
    }
    num_sample
}

fn bench(c: &mut Criterion) {
    let loader = DataLoader::builder(RandomUnique::default())
        .batch_size(16)
        .build();

    const BYTES: u64 = DATASET_LEN as u64 * IMAGE_SIZE as u64 * IMAGE_SIZE as u64 * 3;

    let mut group = c.benchmark_group("throughput-example");
    group.throughput(Throughput::Bytes(BYTES));
    group.bench_function("iter_all_dataset", |b| b.iter(|| iter_all_dataset(&loader)));
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
