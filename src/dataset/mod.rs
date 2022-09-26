use std::collections::VecDeque;

mod len;
pub use len::Len;
mod ndarray_dataset;
pub use ndarray_dataset::NdarrayDataset;
mod get_sample;
pub use get_sample::GetSample;

/// A dataset is just something that has a length and is indexable.
/// A vec of dataset collate output must also be collatable.
///
/// We use a custom [`GetSample`] trait instead of `std::ops::Index` because
/// it provides more flexibility.
/// Indeed we could have provide this implementation:
///
/// ```
/// use dataloader_rs::collate::Collate;
/// use dataloader_rs::Len;
///
/// pub trait Dataset<T>: Len + std::ops::Index<usize>
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
/// And we want to return a tuple (label, text) when indexing, it will no be possible with `std:ops::Index`.
pub trait Dataset: Len + GetSample {}

/// Dataset could become something like that when functor trait will be available.
#[doc(hidden)]
trait FunctorDataset<F>: Len + GetSample
where
    F: Fn(Vec<Self::Sample>) -> Self::CollateOutput,
{
    type CollateOutput;
}

impl<T> Dataset for Vec<T> where T: Clone {}
impl<T> Dataset for VecDeque<T> where T: Clone {}
