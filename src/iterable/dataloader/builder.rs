use std::marker::PhantomData;

use crate::collate::{Collate, DefaultCollate};

use super::DataLoader;

/// Basic builder for creating dataloader.
#[must_use]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash, Ord)]
pub struct Builder<D, C = DefaultCollate>
where
    D: IntoIterator,
    DefaultCollate: Collate<D::Item>,
{
    /// The dataset from which the loader will yield the data.
    dataset: D,

    batch_size: usize,

    drop_last: bool,
    /// Used to collate the data together.
    collate_fn: C,

    shuffle: bool,
}

impl<D> Builder<D, DefaultCollate>
where
    D: IntoIterator,
    DefaultCollate: Collate<D::Item>,
{
    /// Create a new [`Builder`], with default fields.
    /// By default the [`Builder`] is sequential and have a `batch_size` of one.
    pub fn new(dataset: D) -> Self {
        Self {
            dataset,
            batch_size: 1,
            drop_last: false,
            collate_fn: DefaultCollate,
            shuffle: false,
        }
    }
}

impl<D, C> Builder<D, C>
where
    D: IntoIterator,
    DefaultCollate: Collate<D::Item>,
{
    /// Use a random sampler.
    pub fn shuffle(mut self) -> Builder<D, C> {
        self.shuffle = true;
        self
    }
    /// Set the number of elements in a batch.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Drop the lasts element if they don't feat into a batch. For instance if a dataset have 13
    /// samples and a `batch_size` of 5, the last 3 samples will be droped.
    pub fn drop_last(mut self) -> Self {
        self.drop_last = true;
        self
    }

    /// Set a custom collate function.
    pub fn collate_fn<CF>(self, collate_fn: CF) -> Builder<D, CF>
    where
        CF: Collate<D::Item>,
    {
        Builder {
            dataset: self.dataset,
            batch_size: self.batch_size,
            drop_last: self.drop_last,
            collate_fn,
            shuffle: self.shuffle,
        }
    }

    /// Create a `Dataloader` from a [`Builder`].
    pub fn build(self) -> DataLoader<D, C> {
        DataLoader {
            dataset: self.dataset,
            batch_size: self.batch_size,
            drop_last: self.drop_last,
            collate_fn: PhantomData,
            shuffle: self.shuffle,
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
            .shuffle()
            .batch_size(2)
            .drop_last()
            .collate_fn(NoOpCollate)
            .build();
    }
}
