use self::sequential_sampler::SequentialSampler;

pub mod batch_sampler;
pub mod random_sampler;
pub mod sequential_sampler;

pub type DefaultSampler = SequentialSampler;

/// Basic trait for anything that could have a length.
/// Even if a lot of struc have a `len()` methode in the standard library,
/// to my knowledge this function is not included into any standard trait.
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

#[derive(Clone, Copy)]
pub struct DumbSampler;

impl Sampler for DumbSampler {
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

impl HasLength for DumbSampler {
    fn len(&self) -> usize {
        (2..12).len()
    }
}
