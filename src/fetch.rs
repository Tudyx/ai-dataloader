use crate::collate::default_collate::DefaultCollate;
use crate::collate::Collate;
use crate::dataset::Dataset;

/// A Fetcher will fetch data from the dataset
/// Fetcher will be implemented for MapDataset (i.e. indexable dataset)
/// and for iterable dataset
pub trait Fetcher<D, C = DefaultCollate>
where
    D: Dataset<C>,
    C: Collate<D::Output>,
{
    /// Given a batch of index, return the result of the collate function on them
    fn fetch(&self, possibly_batched_index: Vec<usize>) -> C::Output;
}

// <Type as Trait>::function(receiver_if_method, next_arg, ...);
// pub struct ExperimentalFetcher<D, F>
// where
//     D: Dataset,
//     // F: Fn(Vec<<D as GetItem>::Output>) -> <F as Fn(Vec<<D as GetItem>::Output>)>::Output,
//     F: Fn(Vec<<D as GetItem>::Output>) -> i32,
//     DefaultCollector: Collect<Vec<<D as GetItem>::Output>>,
// {
//     pub dataset: D,
//     pub collecate_fn: F,
// }

/// Fetcher for map-style dataset. Simply calll the collate function on all the batch of elements
pub struct MapDatasetFetcher<'dataset, D: Dataset<C>, C = DefaultCollate>
where
    C: Collate<D::Output>,
{
    /// The dataset data will be fetch from
    pub dataset: &'dataset D,
    /// The function (generic struct) used to collate data together
    pub collate_fn: C,
}

impl<'dataset, D, C> Fetcher<D, C> for MapDatasetFetcher<'dataset, D, C>
where
    D: Dataset<C>,
    C: Collate<D::Output>,
{
    fn fetch(&self, possibly_batched_index: Vec<usize>) -> C::Output {
        // As the batch len can vary depending if the last element are drop or not, we can't use
        // a fix sized array.
        let mut data = Vec::with_capacity(possibly_batched_index.len());
        for idx in possibly_batched_index {
            data.push(self.dataset.get_item(idx));
        }
        C::collate(data)
    }
}
