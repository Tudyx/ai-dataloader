use crate::{
    collate::{Collate, DefaultCollate},
    sampler::{BatchSampler, RandomSampler, Sampler, SequentialSampler},
    Dataset,
};
use std::cmp::max;

#[cfg(feature = "rayon")]
use crate::THREAD_POOL;

use super::DataLoader;

/// Basic builder for creating dataloader from a type that implement `IntoIterator`.
/// add a dataloader for all type that implement `IntoIterator`.
/// If the iterator `Item` is not supported by default collate, you must provide your own collate function
#[must_use]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash, Ord)]
pub struct Builder<D, S = SequentialSampler, C = DefaultCollate>
where
    D: Dataset,
    S: Sampler,
    C: Collate<D::Sample>,
{
    /// The dataset from which the loader will yield the data.
    dataset: D,
    /// The sampler used to gather elements of the batch together.
    batch_sampler: BatchSampler<S>,
    /// Used to collate the data together.
    collate_fn: C,
    #[cfg(feature = "rayon")]
    /// Number of threads to use.
    num_threads: usize,
    /// Prefetch buffer size.
    prefetch_size: usize,
}

// FIXME: kind of strange that we require DefaultCollatte even if in the end we may won't use it
impl<D> Builder<D, SequentialSampler, DefaultCollate>
where
    D: Dataset,
    DefaultCollate: Collate<D::Sample>,
{
    /// Create a new [`Builder`], with default fields.
    /// By default the [`Builder`] is sequential and have a `batch_size` of one.
    pub fn new(dataset: D) -> Self {
        #[cfg(feature = "rayon")]
        let num_threads = std::thread::available_parallelism()
            .unwrap_or(std::num::NonZeroUsize::new(1).unwrap())
            .get();

        let dataset_len = dataset.len();
        Self {
            dataset,
            batch_sampler: BatchSampler {
                sampler: SequentialSampler::new(dataset_len),
                batch_size: 1,
                drop_last: false,
            },
            collate_fn: DefaultCollate,
            #[cfg(feature = "rayon")]
            num_threads,
            prefetch_size: 0,
        }
    }
}

impl<D, S, C> Builder<D, S, C>
where
    D: Dataset,
    S: Sampler,
    C: Collate<D::Sample>,
{
    /// Use a random sampler.
    pub fn shuffle(self) -> Builder<D, RandomSampler, C> {
        self.sampler::<RandomSampler>()
    }
    /// Set the number of elements in a batch.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_sampler.batch_size = max(batch_size, 1);
        self
    }

    /// Set the number of threads to use.
    #[cfg(feature = "rayon")]
    pub fn num_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }

    /// Set the size of the prefetch buffer.
    pub fn prefetch_size(mut self, prefetch_size: usize) -> Self {
        self.prefetch_size = prefetch_size;
        self
    }

    /// Drop the lasts element if they don't feat into a batch. For instance if a dataset have 13
    /// samples and a `batch_size` of 5, the last 3 samples will be dropped.
    pub fn drop_last(mut self) -> Self {
        self.batch_sampler.drop_last = true;
        self
    }

    /// Set a custom collate function.
    pub fn collate_fn<CF>(self, collate_fn: CF) -> Builder<D, S, CF>
    where
        CF: Collate<D::Sample>,
    {
        Builder {
            dataset: self.dataset,

            batch_sampler: self.batch_sampler,
            collate_fn,
            #[cfg(feature = "rayon")]
            num_threads: self.num_threads,
            prefetch_size: self.prefetch_size,
        }
    }

    /// Set a custom [`Sampler`].
    pub fn sampler<SA>(self) -> Builder<D, SA, C>
    where
        SA: Sampler,
    {
        let sampler: SA = SA::new(self.dataset.len());
        Builder {
            dataset: self.dataset,
            batch_sampler: BatchSampler {
                sampler,
                batch_size: self.batch_sampler.batch_size,
                drop_last: self.batch_sampler.drop_last,
            },

            collate_fn: self.collate_fn,
            #[cfg(feature = "rayon")]
            num_threads: self.num_threads,
            prefetch_size: self.prefetch_size,
        }
    }
    /// Create a `Dataloader` from a [`Builder`].
    pub fn build(self) -> DataLoader<D, S, C> {
        #[cfg(feature = "rayon")]
        if let Some(pool) = THREAD_POOL.get() {
            if pool.current_num_threads() != self.num_threads {
                // We reset the threadpool because we can't modify the number of
                // threads of an existing thread pool.
                #[cfg(feature = "rayon")]
                THREAD_POOL
                    .set(
                        rayon::ThreadPoolBuilder::new()
                            .num_threads(self.num_threads)
                            .build()
                            .expect("could not spawn threads"),
                    )
                    .ok();
            }
        } else {
            THREAD_POOL
                .set(
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(self.num_threads)
                        .build()
                        .expect("could not spawn threads"),
                )
                .ok();
        }

        DataLoader {
            dataset: self.dataset,
            batch_sampler: self.batch_sampler,
            collate_fn: self.collate_fn,
            prefetch_size: self.prefetch_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collate::NoOpCollate;

    #[test]
    fn api() {
        let _loader = Builder::new(vec![1, 2, 3, 4]).build();
        let _loader = Builder::new(vec![1, 2, 3, 4]).shuffle().build();

        let _loader = Builder::new(vec![1, 2, 3, 4]).batch_size(2).build();

        let _loader = Builder::new(vec![1, 2, 3, 4])
            .batch_size(2)
            .drop_last()
            .build();

        let _loader = Builder::new(vec![1, 2, 3, 4])
            .batch_size(2)
            .drop_last()
            .collate_fn(NoOpCollate)
            .build();

        let _loader = Builder::new(vec![1, 2, 3, 4])
            .batch_size(2)
            .drop_last()
            .sampler::<RandomSampler>()
            .build();

        let _loader = Builder::new(vec![1, 2, 3, 4])
            .batch_size(2)
            .drop_last()
            .sampler::<RandomSampler>()
            .collate_fn(NoOpCollate)
            .build();

        let _loader = Builder::new(vec![1, 2, 3, 4])
            .shuffle()
            .batch_size(2)
            .drop_last()
            .collate_fn(NoOpCollate)
            .build();

        let _loader = Builder::new(vec![1, 2, 3, 4])
            .collate_fn(NoOpCollate)
            .batch_size(2)
            .build();

        let _loader = Builder::new(vec![1, 2, 3, 4])
            .collate_fn(|x| x)
            .batch_size(2)
            .build();
    }
}
