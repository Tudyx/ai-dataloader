pub mod ndarray_dataset;
use crate::sampler::HasLength;

/// A dataset is just something that has a length and is indexable.
/// A vec of dataset collate output must also be collatable
///
/// We use a custom [GetItem] trait instead of `std::ops::Index` because
/// it provides more flexibility.
/// Indeed we could have provide this implementation:
/// ```
/// use dataloader_rs::collate::Collate;
/// use dataloader_rs::sampler::HasLength;
///
/// pub trait Dataset<T>: HasLength + std::ops::Index<usize>
/// where
/// T: Collate<Vec<Self::Output>>,
/// Self::Output: Sized,
/// {
/// }
/// ```
/// But as `Index::Output` must refer as something exist, it will not cover most of our use cases.
/// For instance if the dataset is something like that:
/// ```
/// struct Dataset {
///     labels: Vec<i32>,
///     texts: Vec<String>,
/// }
/// ```
/// And we want to return a tuple (label, text) when indexing, it will no be possible with `std:ops::Index`
pub trait Dataset: HasLength + GetSample {}

/// Dataset could become something like that when functor trait will be available
#[doc(hidden)]
trait FunctorDataset<F>: HasLength + GetSample
where
    F: Fn(Vec<Self::Sample>) -> Self::CollateOutput,
{
    type CollateOutput;
}

/// Return a sample from the dataset at a given index
pub trait GetSample {
    /// Type of one sample of the dataset
    type Sample: Sized;
    /// Return the dataset sample corresponding to the index
    fn get_sample(&self, index: usize) -> Self::Sample;
}

// TODO: Does a blanket implementation of Dataset for type that have implemented std::ops::Index
// will work in that case?
impl<T> Dataset for Vec<T> where T: Clone {}

impl<T: Clone> GetSample for Vec<T> {
    type Sample = T;
    fn get_sample(&self, index: usize) -> Self::Sample {
        self[index].clone()
    }
}
