use std::marker::PhantomData;

use crate::{
    collate::{Collate, DefaultCollate},
    sampler::{BatchSampler, DefaultSampler, Sampler},
    DataLoader, Dataset,
};

/// Basic builder for creating dataloader.
#[derive(Debug, Clone, PartialEq, PartialOrd, Hash, Eq, Ord)]
pub struct DataLoaderBuilder<D, S = DefaultSampler, C = DefaultCollate>
where
    D: Dataset,
    S: Sampler,
    C: Collate<D::Sample>,
{
    /// The dataset from which the loader will yield the data.
    dataset: D,
    /// Number of element in a batch.
    batch_size: usize,
    /// Used to sample the dataset (randomly or not).
    sampler: Option<S>,
    /// The sampler userd to gather element of the batch together.
    batch_sampler: Option<BatchSampler<S>>,
    /// if we drop sample at the end that doesn't fullfill a whole batch.
    drop_last: bool,
    /// Used to collate the data together.
    collate_fn: Option<C>,
}
impl<D, S, C> DataLoaderBuilder<D, S, C>
where
    D: Dataset,
    S: Sampler,
    C: Collate<D::Sample>,
{
    pub fn new(dataset: D) -> Self {
        Self {
            dataset,
            batch_size: 1,
            sampler: None,
            batch_sampler: None,
            drop_last: false,
            collate_fn: None,
        }
    }
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
    pub fn with_collate_fn(mut self, collate_fn: C) -> Self {
        self.collate_fn = Some(collate_fn);
        self
    }
    pub fn with_sampler(mut self, sampler: S) -> Self {
        self.sampler = Some(sampler);
        self
    }

    pub fn build(mut self) -> DataLoader<D, S, C> {
        if self.batch_sampler.is_some() && self.batch_size != 0
            || self.sampler.is_some()
            || self.drop_last
        {
            panic!("batch_sampler option is mutually exclusive with batch_size,  sampler, and drop_last'")
        }

        let sampler = self.sampler.unwrap_or_else(|| S::new(self.dataset.len()));

        if self.batch_sampler.is_none() {
            self.batch_sampler = Some(BatchSampler {
                sampler, // because sampler implement the copy trait
                batch_size: self.batch_size,
                drop_last: self.drop_last,
            })
        }
        DataLoader {
            dataset: self.dataset,
            batch_sampler: self.batch_sampler.unwrap(),
            phantom: PhantomData,
        }
    }
}
