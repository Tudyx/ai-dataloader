//! Defines the strategy to draw samples from the dataset.
//!

use crate::Len;

mod batch_sampler;
mod random_sampler;
mod sequential_sampler;

pub use batch_sampler::{BatchIterator, BatchSampler};
pub use random_sampler::RandomSampler;
pub use sequential_sampler::SequentialSampler;

// TODO: maybe copy is too restrictive and should be replaced by Clone?
/// Every Sampler is iterable and has a length.
pub trait Sampler: Len + IntoIterator<Item = usize> + Copy {
    /// Create a new sampler form the dataset length.
    fn new(data_source_len: usize) -> Self;
}
