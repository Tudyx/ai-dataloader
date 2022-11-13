//! # Iterable `DataLoader`
//!
//! Iterable `DataLoader` is a `DataLoader` that operate on an iterable dataset.
//! An iterable dataset is just a type that implement `IntoIterator`.

mod builder;
use builder::Builder;
use rand::{seq::SliceRandom, thread_rng};

use std::marker::PhantomData;

use crate::collate::{Collate, DefaultCollate};

/// For iterable dataset, the `datalaoder` will yield until the underlying iterator is `None`.
/// As the iteration over the dataset can be done mutliple time, depending if the underlying dataset iterator consume the dataset or not.
#[derive(Debug)]
pub struct DataLoader<D, C> {
    /// The dataset we will iterate over.
    dataset: D,
    /// The number of sample a batch will contain.
    batch_size: usize,
    /// If `true`, the sampler will drop the last batch if
    /// its size were less than `batch_size`.
    drop_last: bool,
    /// Just here because collate has no data members.
    collate_fn: PhantomData<C>,
    /// If `true` the sample in the batch will be shuffled
    shuffle: bool,
}

impl<D> DataLoader<D, DefaultCollate>
where
    D: IntoIterator,
    DefaultCollate: Collate<D::Item>,
{
    /// return a [`DataLoader`] builder.
    pub fn builder(dataset: D) -> Builder<D, DefaultCollate> {
        Builder::new(dataset)
    }
}

// we want to use dataloader in for loop
// A dataset is something we can turn into an iterator.
// We make a an iterator that consume this iterator and yield only batches of it.
impl<D, C> IntoIterator for DataLoader<D, C>
where
    D: IntoIterator,
    C: Collate<<D as IntoIterator>::Item>,
{
    // We yield batch of dataset element (which can be transformed by the collate function).
    type Item = C::Output;
    type IntoIter = IntoIter<D::IntoIter, C>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            batch_size: self.batch_size,
            dataset_iter: self.dataset.into_iter(),
            drop_last: self.drop_last,
            collate_fn: PhantomData,
            shuffle: self.shuffle,
        }
    }
}

/// Iterator returned by `into_iter` function.
#[derive(Debug)]
pub struct IntoIter<D, C> {
    batch_size: usize,
    dataset_iter: D,
    drop_last: bool,
    collate_fn: PhantomData<C>,
    shuffle: bool,
}

impl<D, C> Iterator for IntoIter<D, C>
where
    D: Iterator,
    C: Collate<D::Item>,
{
    type Item = C::Output;
    fn next(&mut self) -> Option<Self::Item> {
        let mut batch = self
            .dataset_iter
            .by_ref()
            .take(self.batch_size)
            .collect::<Vec<_>>();

        if batch.is_empty() {
            return None;
        }

        if batch.len() == self.batch_size || (batch.len() != self.batch_size && !self.drop_last) {
            if self.shuffle {
                batch.shuffle(&mut thread_rng());
            }
            return Some(C::collate(batch));
        }
        None
    }
}

/// Iterator returned by `iter` function.
#[derive(Debug)]
pub struct Iter<D, C> {
    batch_size: usize,
    dataset_iter: D,
    drop_last: bool,
    collate_fn: PhantomData<C>,
    shuffle: bool,
}

impl<'dataset, D, C> IntoIterator for &'dataset DataLoader<D, C>
where
    D: 'dataset,
    &'dataset D: IntoIterator,
    C: Collate<<&'dataset D as IntoIterator>::Item>,
{
    type Item = C::Output;
    type IntoIter = Iter<<&'dataset D as IntoIterator>::IntoIter, C>;

    fn into_iter(self) -> Self::IntoIter {
        Iter {
            batch_size: self.batch_size,
            dataset_iter: self.dataset.into_iter(),
            drop_last: self.drop_last,
            collate_fn: PhantomData,
            shuffle: self.shuffle,
        }
    }
}

impl<'dataset, D, C> DataLoader<D, C>
where
    D: 'dataset,
    &'dataset D: IntoIterator,
    C: Collate<<&'dataset D as IntoIterator>::Item>,
{
    /// Iterate over the dataloader without consuming the underlying dataset.
    /// As it make no sens to collate reference into a tensor, by default element are copied.
    pub fn iter(&'dataset self) -> Iter<<&'dataset D as IntoIterator>::IntoIter, C> {
        Iter {
            batch_size: self.batch_size,
            dataset_iter: self.dataset.into_iter(),
            drop_last: self.drop_last,
            collate_fn: PhantomData,
            shuffle: self.shuffle,
        }
    }
}

impl<D, C> Iterator for Iter<D, C>
where
    D: Iterator,
    C: Collate<D::Item>,
{
    type Item = C::Output;
    fn next(&mut self) -> Option<Self::Item> {
        let mut batch = self
            .dataset_iter
            .by_ref()
            .take(self.batch_size)
            .collect::<Vec<_>>();

        if batch.is_empty() {
            return None;
        }

        if batch.len() == self.batch_size || (batch.len() != self.batch_size && !self.drop_last) {
            if self.shuffle {
                batch.shuffle(&mut thread_rng());
            }
            return Some(C::collate(batch));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::collate::NoOpCollate;
    use ndarray::array;

    #[test]
    fn multiple_iteration() {
        let dataset = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let loader = DataLoader::builder(dataset).batch_size(2).build();

        for sample in loader.iter() {
            dbg!(sample);
        }

        for sample in &loader {
            dbg!(sample);
        }

        let mut into_iter = loader.into_iter();
        assert_eq!(into_iter.next(), Some(array![0, 1]));
        assert_eq!(into_iter.next(), Some(array![2, 3]));
        assert_eq!(into_iter.next(), Some(array![4, 5]));
        assert_eq!(into_iter.next(), Some(array![6, 7]));
        assert_eq!(into_iter.next(), Some(array![8, 9]));
        assert_eq!(into_iter.next(), Some(array![10]));
        assert_eq!(into_iter.next(), None);
    }

    #[test]
    fn drop_last() {
        let dataset = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let loader = DataLoader::builder(dataset)
            .batch_size(2)
            .drop_last()
            .build();

        let mut into_iter = loader.into_iter();
        assert_eq!(into_iter.next(), Some(array![0, 1]));
        assert_eq!(into_iter.next(), Some(array![2, 3]));
        assert_eq!(into_iter.next(), Some(array![4, 5]));
        assert_eq!(into_iter.next(), Some(array![6, 7]));
        assert_eq!(into_iter.next(), Some(array![8, 9]));
        assert_eq!(into_iter.next(), None);
    }

    #[test]
    fn custom_collate() {
        let dataset = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let loader = DataLoader::builder(dataset)
            .batch_size(2)
            .collate_fn(NoOpCollate)
            .build();

        let mut into_iter = loader.into_iter();
        assert_eq!(into_iter.next(), Some(vec![0, 1]));
        assert_eq!(into_iter.next(), Some(vec![2, 3]));
        assert_eq!(into_iter.next(), Some(vec![4, 5]));
        assert_eq!(into_iter.next(), Some(vec![6, 7]));
        assert_eq!(into_iter.next(), Some(vec![8, 9]));
        assert_eq!(into_iter.next(), Some(vec![10]));
        assert_eq!(into_iter.next(), None);
    }
}
