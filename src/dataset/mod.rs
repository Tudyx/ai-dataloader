pub mod ndarray_dataset;
use crate::collate::default_collate::DefaultCollate;
use crate::collate::Collate;
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
pub trait Dataset<C = DefaultCollate>: HasLength + GetItem
where
    C: Collate<Vec<Self::Output>>,
{
    // it's kind of weird that the dataset is link to a collator, a dataset shouldn't even know that collator exist
}

/// Dataset could become something like that when functor trait will be available
#[doc(hidden)]
trait FunctorDataset<F>: HasLength + GetItem
where
    F: Fn(Vec<Self::Output>) -> Self::CollateOutput,
{
    type CollateOutput;
}

/// Return an item of the dataset
pub trait GetItem {
    /// Type of one sample of the dataset
    type Output: Sized;
    /// Return the dataset sample corresponding to the index
    fn get_item(&self, index: usize) -> Self::Output;
}

// TODO: Does a blanket implementation of Dataset for type that have implemented std::ops::Index
// will work in that case?
impl<T, C> Dataset<C> for Vec<T>
where
    T: Clone,
    C: Collate<Vec<Self::Output>>,
{
}

impl<T: Clone> GetItem for Vec<T> {
    type Output = T;
    fn get_item(&self, index: usize) -> Self::Output {
        self[index].clone()
    }
}
