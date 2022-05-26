use self::sequential_sampler::SequentialSampler;

pub mod batch_sampler;
pub mod random_sampler;
pub mod sequential_sampler;

pub type DefaultSampler = SequentialSampler;

/// Basic trait for anything that could have a length.
/// Even if a lot of struc have a `len()` method in the standard library,
/// to my knowledge this function is not included into any standard trait.
pub trait HasLength {
    /// Returns the number of elements in the collection, also referred to
    /// as its 'length'.
    fn len(&self) -> usize;
    /// Return `true` if the collection has no element
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// TODO: maybe copy is too restrictive and should be replaced by Clone?
/// Every Sampler is iterable and has a length
pub trait Sampler: HasLength + IntoIterator<Item = usize> + Copy {
    /// Create a new sampler form the dataset length
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

impl HasLength for DumbSampler {
    fn len(&self) -> usize {
        (2..12).len()
    }
}
