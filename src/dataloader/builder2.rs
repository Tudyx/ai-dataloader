use std::marker::PhantomData;

use crate::{
    collate::{Collate, DefaultCollate},
    sampler::{BatchSampler, DefaultSampler, RandomSampler, Sampler, SequentialSampler},
    DataLoader, Dataset,
};

/// Basic builder for creating dataloader.
#[derive(Debug, Clone, PartialEq, PartialOrd, Hash, Eq, Ord)]
pub struct DataLoaderBuilder2<D, S = DefaultSampler, C = DefaultCollate>
where
    // D: Dataset,
    S: Sampler,
    // C: Collate<D::Sample>,
{
    /// The dataset from which the loader will yield the data.
    dataset: D,
    /// The sampler userd to gather element of the batch together.
    batch_sampler: BatchSampler<S>,
    /// Used to collate the data together.
    collate_fn: C,
}

impl<D> DataLoaderBuilder2<D, SequentialSampler, DefaultCollate>
where
    D: Dataset,
{
    pub fn new(dataset: D) -> DataLoaderBuilder2<D> {
        let dataset_len = dataset.len();
        Self {
            dataset,
            batch_sampler: BatchSampler {
                sampler: SequentialSampler::new(dataset_len),
                batch_size: 1,
                drop_last: false,
            },
            collate_fn: DefaultCollate,
        }
    }
}

impl<D, S, C> DataLoaderBuilder2<D, S, C>
where
    D: Dataset,
    S: Sampler,
    // TODO: verify we can't produce invalide dataloader because the line below is commented
    // C: Collate<D::Sample>,
{
    pub fn shuffle(self) -> DataLoaderBuilder2<D, RandomSampler, C> {
        self.with_sampler::<RandomSampler>()
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_sampler.batch_size = batch_size;
        self
    }

    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.batch_sampler.drop_last = drop_last;
        self
    }

    pub fn with_collate_fn<CF>(self, collate_fn: CF) -> DataLoaderBuilder2<D, S, CF> {
        DataLoaderBuilder2 {
            dataset: self.dataset,

            batch_sampler: self.batch_sampler,
            collate_fn,
        }
    }

    pub fn with_sampler<SA>(self) -> DataLoaderBuilder2<D, SA, C>
    where
        SA: Sampler,
    {
        let sampler: SA = SA::new(self.dataset.len());
        DataLoaderBuilder2 {
            dataset: self.dataset,
            batch_sampler: BatchSampler {
                sampler,
                batch_size: self.batch_sampler.batch_size,
                drop_last: self.batch_sampler.drop_last,
            },

            collate_fn: self.collate_fn,
        }
    }

    pub fn build(self) -> DataLoader<D, S, C> {
        DataLoader {
            dataset: self.dataset,
            batch_sampler: self.batch_sampler,
            phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collate::NoOpCollate;

    #[test]
    fn api() {
        let _loader = DataLoaderBuilder2::new(vec![1, 2, 3, 4]).build();
        let _loader = DataLoaderBuilder2::new(vec![1, 2, 3, 4]).shuffle().build();

        let _loader = DataLoaderBuilder2::new(vec![1, 2, 3, 4])
            .with_batch_size(2)
            .build();

        let _loader = DataLoaderBuilder2::new(vec![1, 2, 3, 4])
            .with_batch_size(2)
            .drop_last(true)
            .build();

        let _loader = DataLoaderBuilder2::new(vec![1, 2, 3, 4])
            .with_batch_size(2)
            .drop_last(true)
            .with_collate_fn(NoOpCollate)
            .build();

        let _loader = DataLoaderBuilder2::new(vec![1, 2, 3, 4])
            .with_batch_size(2)
            .drop_last(true)
            .with_sampler::<RandomSampler>()
            .build();

        let _loader = DataLoaderBuilder2::new(vec![1, 2, 3, 4])
            .with_batch_size(2)
            .drop_last(true)
            .with_sampler::<RandomSampler>()
            .with_collate_fn(NoOpCollate)
            .build();

        let _loader = DataLoaderBuilder2::new(vec![1, 2, 3, 4])
            .shuffle()
            .with_batch_size(2)
            .drop_last(true)
            .with_collate_fn(NoOpCollate)
            .build();
    }
}
