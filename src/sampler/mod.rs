use self::sequential_sampler::SequentialSampler;

pub mod batch_sampler;
pub mod random_sampler;
pub mod sequential_sampler;

pub type DefaultSampler = SequentialSampler;

pub trait HasLength {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// maybe copy is too restrictive and should be replaced by Clone
pub trait Sampler: HasLength + IntoIterator<Item = usize> + Copy {
    fn new(data_source_len: usize) -> Self;
}
// Even if vector have it, we should do it.
// That's because there is no standard trait for len
impl<T> HasLength for Vec<T> {
    fn len(&self) -> usize {
        self.len()
    }
}
