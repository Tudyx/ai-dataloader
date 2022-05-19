use super::{DefaultSampler, HasLength, Sampler};

#[derive(Debug, Clone)]
pub struct BatchSampler<S: Sampler = DefaultSampler> {
    pub sampler: S,
    pub batch_size: usize,
    pub drop_last: bool,
}

impl<S: Sampler> HasLength for BatchSampler<S> {
    // return the number of batch
    // if drop_last is set to false, even an incomplete batch will be counted
    fn len(&self) -> usize {
        if self.drop_last {
            self.sampler.len() / self.batch_size
        } else {
            (self.sampler.len() + self.batch_size - 1) / self.batch_size
        }
    }
}
impl<S: Sampler> BatchSampler<S> {
    pub fn iter(&self) -> BatchIterator<S::IntoIter> {
        BatchIterator {
            sampler: self.sampler.into_iter(),
            batch_size: self.batch_size,
            drop_last: self.drop_last,
        }
    }
}

pub struct BatchIterator<I>
where
    I: Iterator<Item = usize>,
{
    sampler: I,
    batch_size: usize,
    drop_last: bool,
}

impl<I> Iterator for BatchIterator<I>
where
    I: Iterator<Item = usize>,
{
    type Item = Vec<usize>;
    fn next(&mut self) -> Option<Self::Item> {
        let mut batch = Vec::new();

        // We can't use a classic for loop here because it will
        // try to move the &mut
        let mut current_idx = self.sampler.next();
        while let Some(idx) = current_idx {
            batch.push(idx);
            if batch.len() == self.batch_size {
                return Some(batch);
            }
            current_idx = self.sampler.next();
        }
        if batch.len() > 0 && !self.drop_last {
            return Some(batch);
        }
        None
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::sequential_sampler::SequentialSampler;

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
        let mut vec = Vec::new();
        for i in (0..20).step_by(5) {
            vec.push(vec![i..i + 2]);
            vec.push(vec![i + 2..i + 5]);
            // println!("{}", i);
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
