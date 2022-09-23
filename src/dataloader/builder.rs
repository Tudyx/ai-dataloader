use std::marker::PhantomData;

use crate::{
    collate::DefaultCollate,
    sampler::{BatchSampler, RandomSampler, Sampler, SequentialSampler},
    DataLoader, Dataset,
};

/// Basic builder for creating dataloader.
#[derive(Debug, Clone, PartialEq, PartialOrd, Hash, Eq, Ord)]
pub struct DataLoaderBuilder<D, S = SequentialSampler, C = DefaultCollate>
where
    D: Dataset,
    S: Sampler,
    C: Collate<D::Sample>,
{
    /// The dataset from which the loader will yield the data.
    dataset: D,
    /// The sampler userd to gather element of the batch together.
    batch_sampler: BatchSampler<S>,
    /// Used to collate the data together.
    collate_fn: C,
}

impl<D> DataLoaderBuilder<D, SequentialSampler, DefaultCollate>
where
    D: Dataset,
    DefaultCollate: Collate<D::Sample>,
{
    pub fn new(dataset: D) -> DataLoaderBuilder<D> {
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
use crate::collate::Collate;

impl<D, S, C> DataLoaderBuilder<D, S, C>
where
    D: Dataset,
    S: Sampler,
    C: Collate<D::Sample>,
{
    pub fn shuffle(self) -> DataLoaderBuilder<D, RandomSampler, C> {
        self.sampler::<RandomSampler>()
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_sampler.batch_size = batch_size;
        self
    }

    pub fn drop_last(mut self) -> Self {
        self.batch_sampler.drop_last = true;
        self
    }

    pub fn collate_fn<CF>(self, collate_fn: CF) -> DataLoaderBuilder<D, S, CF>
    where
        CF: Collate<D::Sample>,
    {
        DataLoaderBuilder {
            dataset: self.dataset,

            batch_sampler: self.batch_sampler,
            collate_fn,
        }
    }

    pub fn sampler<SA>(self) -> DataLoaderBuilder<D, SA, C>
    where
        SA: Sampler,
    {
        let sampler: SA = SA::new(self.dataset.len());
        DataLoaderBuilder {
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
        let _loader = DataLoaderBuilder::new(vec![1, 2, 3, 4]).build();
        let _loader = DataLoaderBuilder::new(vec![1, 2, 3, 4]).shuffle().build();

        let _loader = DataLoaderBuilder::new(vec![1, 2, 3, 4])
            .batch_size(2)
            .build();

        let _loader = DataLoaderBuilder::new(vec![1, 2, 3, 4])
            .batch_size(2)
            .drop_last()
            .build();

        let _loader = DataLoaderBuilder::new(vec![1, 2, 3, 4])
            .batch_size(2)
            .drop_last()
            .collate_fn(NoOpCollate)
            .build();

        let _loader = DataLoaderBuilder::new(vec![1, 2, 3, 4])
            .batch_size(2)
            .drop_last()
            .sampler::<RandomSampler>()
            .build();

        let _loader = DataLoaderBuilder::new(vec![1, 2, 3, 4])
            .batch_size(2)
            .drop_last()
            .sampler::<RandomSampler>()
            .collate_fn(NoOpCollate)
            .build();

        // TODO: checker la syntax des builder dans la STL, voir s'il utilise "with_", des verbe, etc..

        let _loader = DataLoaderBuilder::new(vec![1, 2, 3, 4])
            .shuffle()
            .batch_size(2)
            .drop_last()
            .collate_fn(NoOpCollate)
            .build();
    }
}
