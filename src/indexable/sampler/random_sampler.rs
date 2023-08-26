use rand::seq::SliceRandom;
use rand::thread_rng;

use super::{Len, Sampler};

/// Sampler that returns random index between zero and `data_source_len`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Hash, Eq, Ord)]
pub struct RandomSampler {
    /// The length of the data source.
    data_source_len: usize,
    /// Whether the sample is replaced or not.
    /// If it's replaced, we can have 2 times the same sample.
    replacement: bool,
}

impl Sampler for RandomSampler {
    fn new(data_source_len: usize) -> Self {
        Self {
            data_source_len,
            replacement: false,
        }
    }
}
impl Len for RandomSampler {
    fn len(&self) -> usize {
        self.data_source_len
    }
}
impl IntoIterator for RandomSampler {
    type IntoIter = RandomSamplerIter;
    type Item = usize;
    fn into_iter(self) -> Self::IntoIter {
        RandomSamplerIter::new(self.data_source_len, self.replacement)
    }
}
/// Iterator that returns random index between zero and `data_source_len`.
#[derive(Debug)]
pub struct RandomSamplerIter {
    /// The length of the data source.
    data_source_len: usize,
    /// A permutation over the datasets indexes.
    indexes: Vec<usize>,
    /// The current index.
    idx: usize,
}

impl RandomSamplerIter {
    /// Create a new `RandomSamplerIter`.
    ///
    /// # Arguments
    ///
    /// * `data_source_len` - The length of the dataset.
    /// * `replacement` - Whether we can have the same sample twice over one iteration or not.
    fn new(data_source_len: usize, replacement: bool) -> Self {
        if replacement {
            todo!()
        } else {
            let mut vec: Vec<usize> = (0..data_source_len).collect();
            vec.shuffle(&mut thread_rng());
            Self {
                data_source_len,
                indexes: vec,
                idx: 0,
            }
        }
    }
}
impl Iterator for RandomSamplerIter {
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.data_source_len {
            self.idx += 1;
            Some(self.indexes[self.idx - 1])
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_sampler() {
        let random_sampler = RandomSampler {
            data_source_len: 10,
            replacement: false,
        };
        for idx in random_sampler {
            println!("{idx}");
        }
    }
}
