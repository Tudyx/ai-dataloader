use crate::Len;

mod batch_sampler;
mod random_sampler;
mod sequential_sampler;

pub use batch_sampler::{BatchIterator, BatchSampler};
pub use random_sampler::RandomSampler;
pub use sequential_sampler::SequentialSampler;

pub type DefaultSampler = SequentialSampler;

// TODO: maybe copy is too restrictive and should be replaced by Clone?
/// Every Sampler is iterable and has a length.
pub trait Sampler: Len + IntoIterator<Item = usize> + Copy {
    /// Create a new sampler form the dataset length.
    fn new(data_source_len: usize) -> Self;
}

#[derive(Clone, Copy)]
pub struct DumbSampler;

impl Sampler for DumbSampler {
    #[allow(unused_variables)]
    fn new(data_source_len: usize) -> Self {
        DumbSampler {}
    }
}

impl IntoIterator for DumbSampler {
    type Item = usize;
    type IntoIter = std::ops::Range<usize>;
    fn into_iter(self) -> Self::IntoIter {
        2..12
    }
}

impl Len for DumbSampler {
    fn len(&self) -> usize {
        (2..12).len()
    }
}
