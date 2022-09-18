use std::ops::Range;

use crate::{sampler::Sampler, Len};

/// Yield index from 0 to `data_source_len` in ascending order
#[derive(Debug, Clone, Copy)]
pub struct SequentialSampler {
    /// The length of the dataset that will be sampled
    pub data_source_len: usize,
}
impl Sampler for SequentialSampler {
    fn new(data_source_len: usize) -> Self {
        SequentialSampler { data_source_len }
    }
}

impl Len for SequentialSampler {
    fn len(&self) -> usize {
        self.data_source_len
    }
}
impl IntoIterator for SequentialSampler {
    type IntoIter = Range<usize>;
    type Item = usize;
    fn into_iter(self) -> Self::IntoIter {
        0..self.data_source_len
    }
}

#[test]
fn sequential_sampler() {
    let dataset = vec![1, 2, 3];
    let sampler = SequentialSampler {
        data_source_len: dataset.len(),
    };
    let mut iter = sampler.into_iter();
    assert_eq!(iter.next(), Some(0));
    assert_eq!(iter.next(), Some(1));
    assert_eq!(iter.next(), Some(2));
    assert_eq!(iter.next(), None);
}
