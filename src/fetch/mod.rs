use std::marker::PhantomData;

use crate::{
    collate::{Collate, DefaultCollate},
    Dataset,
};

/// A Fetcher will fetch data from the dataset.
/// Fetcher will be implemented for `MapDataset` (i.e. indexable dataset)
/// and for iterable dataset.
pub(crate) trait Fetcher<D, C = DefaultCollate>
where
    D: Dataset,
    C: Collate<D::Sample>,
{
    /// Given a batch of index, return the result of the collate function on them.
    fn fetch(&self, possibly_batched_index: Vec<usize>) -> C::Output;
}

/// Fetcher for map-style dataset. Simply call the collate function on all the batch of elements.
pub(crate) struct MapDatasetFetcher<'dataset, D: Dataset, C = DefaultCollate>
where
    C: Collate<D::Sample>,
{
    /// The dataset data will be fetch from.
    pub dataset: &'dataset D,
    /// The function (generic struct) used to collate data together.
    pub collate_fn: PhantomData<C>,
}

impl<'dataset, D, C> Fetcher<D, C> for MapDatasetFetcher<'dataset, D, C>
where
    D: Dataset,
    C: Collate<D::Sample>,
{
    fn fetch(&self, possibly_batched_index: Vec<usize>) -> C::Output {
        // As the batch length can vary depending on if the last element is dropped or not, we can't use
        // a fix sized array.
        let mut data = Vec::with_capacity(possibly_batched_index.len());
        for idx in possibly_batched_index {
            data.push(self.dataset.get_sample(idx));
        }
        C::collate(data)
    }
}
