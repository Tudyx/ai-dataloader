// // Contains definitions of the methods used by the _BaseDataLoaderIter to fetch
// // data from an iterable-style or map-style dataset. This logic is shared in both
// // single- and multi-processing data loading.

use crate::collate::default_collate::DefaultCollator;
use crate::collate::Collate;
use crate::dataset::Dataset;

/// A Fetcher will fetch data from the dataset
pub trait Fetcher<D, C = DefaultCollator>
where
    D: Dataset<C>,
    C: Collate<Vec<D::Output>>,
{
    fn fetch(&self, possibly_batched_index: Vec<usize>) -> C::Output;
}

// <Type as Trait>::function(receiver_if_method, next_arg, ...);
// pub struct ExperimentalFetcher<D, F>
// where
//     D: Dataset,
//     // F: Fn(Vec<<D as GetItem<usize>>::Output>) -> <F as Fn(Vec<<D as GetItem<usize>>::Output>)>::Output,
//     F: Fn(Vec<<D as GetItem<usize>>::Output>) -> i32,
//     DefaultCollector: Collect<Vec<<D as GetItem<usize>>::Output>>,
// {
//     pub dataset: D,
//     pub collecate_fn: F,
// }
pub struct MapDatasetFetcher<'dataset, D: Dataset<C>, C = DefaultCollator>
where
    C: Collate<Vec<D::Output>>,
{
    pub dataset: &'dataset D,
    pub collate_fn: C,
}

impl<'dataset, D, C> Fetcher<D, C> for MapDatasetFetcher<'dataset, D, C>
where
    D: Dataset<C>,
    C: Collate<Vec<D::Output>>,
{
    fn fetch(&self, possibly_batched_index: Vec<usize>) -> C::Output {
        let mut data = Vec::with_capacity(possibly_batched_index.len());
        for idx in possibly_batched_index {
            data.push(self.dataset.get_item(idx));
        }
        C::collate(data)
    }
}
