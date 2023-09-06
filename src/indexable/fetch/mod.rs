use crate::{
    collate::{Collate, DefaultCollate},
    Dataset,
};
use std::sync::Arc;

#[cfg(feature = "rayon")]
use crate::THREAD_POOL;

#[cfg(feature = "rayon")]
use rayon::iter::ParallelIterator;

#[cfg(feature = "rayon")]
use rayon::prelude::IntoParallelIterator;

// FIXME: a fetcher trait doesn't make sens anymore.

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
#[derive(Debug)]
pub(crate) struct MapDatasetFetcher<D, C = DefaultCollate>
where
    D: Dataset,
    C: Collate<D::Sample>,
{
    /// The dataset data will be fetch from.
    pub(crate) dataset: Arc<D>,
    /// The function (generic struct) used to collate data together.
    pub(crate) collate_fn: Arc<C>,
}

impl<D, C> Fetcher<D, C> for MapDatasetFetcher<D, C>
where
    D: Dataset + Sync + Send,
    C: Collate<D::Sample> + Sync + Send,
    D::Sample: Send,
{
    fn fetch(&self, possibly_batched_index: Vec<usize>) -> C::Output {
        // As the batch length can vary depending on if the last element is dropped or not, we can't use a fix len array to
        // collect the data.
        #[cfg(feature = "rayon")]
        let data = THREAD_POOL
            .get()
            .expect("thread pool is initialized")
            .install(|| {
                possibly_batched_index
                    .into_par_iter()
                    .map(|idx| self.dataset.get_sample(idx))
                    .collect()
            });
        #[cfg(not(feature = "rayon"))]
        let data = possibly_batched_index
            .into_iter()
            .map(|idx| self.dataset.get_sample(idx))
            .collect();

        self.collate_fn.collate(data)
    }
}
