use std::ops::Range;

use super::{HasLength, Sampler};

#[derive(Debug, Clone, Copy)]
pub struct SequentialSampler {
    pub data_source_len: usize,
}
impl Sampler for SequentialSampler {
    fn new(data_source_len: usize) -> Self {
        SequentialSampler { data_source_len }
    }
}

impl HasLength for SequentialSampler {
    fn len(&self) -> usize {
        self.data_source_len
    }
}
impl IntoIterator for SequentialSampler {
    type IntoIter = Range<usize>;
    type Item = usize;
    fn into_iter(self) -> Self::IntoIter {
        println!("Create a vec of len {}", self.data_source_len);
        (0..self.data_source_len).into_iter()
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
