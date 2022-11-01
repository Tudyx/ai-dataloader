use super::{Sampler, SequentialSampler};
use crate::Len;

/// Wraps another sampler to yield a mini-batch of indices.
/// # Arguments
///
/// * `sampler` - Base sampler.
/// * `batch_size` - Size of mini-batch.
/// * `drop_last` - If `true`, the sampler will drop the last batch if its size would be less than `batch_size`.
///
///
/// # Examples:
/// ```
/// use ai_dataloader::sampler::SequentialSampler;
/// use ai_dataloader::sampler::BatchSampler;
///
/// let dataset = vec![0, 1, 2, 3];
/// let batch_sampler = BatchSampler {
///     sampler: SequentialSampler {
///     data_source_len: dataset.len(),
///     },
///     batch_size: 2,
///     drop_last: false,
/// };
/// let mut iter = batch_sampler.iter();
/// assert_eq!(iter.next(), Some(vec![0, 1]));
/// assert_eq!(iter.next(), Some(vec![2, 3]));
/// ```
#[derive(Debug, Clone, PartialEq, PartialOrd, Hash, Eq, Ord)]
pub struct BatchSampler<S = SequentialSampler> {
    /// Base sampler.
    pub sampler: S,
    /// Size of mini batch.
    pub batch_size: usize,
    /// If `true`, the sampler will drop the last batch if
    /// its size were less than ``batch_size``.
    pub drop_last: bool,
}

impl<S: Sampler> Len for BatchSampler<S> {
    /// Returns the number of batch.
    ///
    /// If `drop_last` is set to false, even an incomplete batch will be counted.
    fn len(&self) -> usize {
        if self.drop_last {
            self.sampler.len() / self.batch_size
        } else {
            (self.sampler.len() + self.batch_size - 1) / self.batch_size
        }
    }
}
impl<S: Sampler> BatchSampler<S> {
    /// Return an iterator over the [`BatchSampler`].
    pub fn iter(&self) -> BatchIterator<S::IntoIter> {
        BatchIterator {
            sampler: self.sampler.into_iter(),
            batch_size: self.batch_size,
            drop_last: self.drop_last,
        }
    }
}

/// An iterator for the batch. Yield a sequence of index at each iteration.
#[derive(Debug)]
pub struct BatchIterator<I>
where
    I: Iterator<Item = usize>,
{
    /// The underlying sampler.
    sampler: I,
    /// The size of one batch.
    batch_size: usize,
    /// Weither to drop the laste elements or not.
    drop_last: bool,
}

impl<I> Iterator for BatchIterator<I>
where
    I: Iterator<Item = usize>,
{
    type Item = Vec<usize>;
    fn next(&mut self) -> Option<Self::Item> {
        let mut batch = Vec::with_capacity(self.batch_size);

        // We can't use a classic for loop here because it will
        // try to move the `&mut`.
        let mut current_idx = self.sampler.next();
        while let Some(idx) = current_idx {
            batch.push(idx);
            if batch.len() == self.batch_size {
                return Some(batch);
            }
            current_idx = self.sampler.next();
        }
        if !batch.is_empty() && !self.drop_last {
            return Some(batch);
        }
        None
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basics() {
        let dataset = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let batch_sampler = BatchSampler {
            sampler: SequentialSampler {
                data_source_len: dataset.len(),
            },
            batch_size: 3,
            drop_last: false,
        };
        for (i, batch_indices) in batch_sampler.iter().enumerate() {
            println!("Batch #{} indices: {:?}", i, batch_indices);
        }
        let mut iter = batch_sampler.iter();
        assert_eq!(iter.next(), Some(vec![0, 1, 2]));
        assert_eq!(iter.next(), Some(vec![3, 4, 5]));
        assert_eq!(iter.next(), Some(vec![6, 7, 8]));
    }
    #[test]
    fn batch_sampler() {
        // TODO : test from pytorch, need to support custom batch sampler
        let mut batches = Vec::new();
        for i in (0..20).step_by(5) {
            batches.push([i..i + 2]);
            batches.push([i + 2..i + 5]);
        }
    }
    #[test]
    fn len() {
        let dataset = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let batch_sampler = BatchSampler {
            sampler: SequentialSampler {
                data_source_len: dataset.len(),
            },
            batch_size: 2,
            drop_last: false,
        };
        assert_eq!(batch_sampler.len(), 5);

        let dataset = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let batch_sampler = BatchSampler {
            sampler: SequentialSampler {
                data_source_len: dataset.len(),
            },
            batch_size: 2,
            drop_last: false,
        };
        assert_eq!(batch_sampler.len(), 6);

        let dataset = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let batch_sampler = BatchSampler {
            sampler: SequentialSampler {
                data_source_len: dataset.len(),
            },
            batch_size: 2,
            drop_last: true,
        };
        assert_eq!(batch_sampler.len(), 5);
    }
}
