use rand::seq::SliceRandom;
use rand::thread_rng;

use super::{HasLength, Sampler};

/// Sampler that return random index between 0 and `data_source_len`
#[derive(Debug, Clone, Copy)]
pub struct RandomSampler {
    data_source_len: usize,
    /// Whether the sample is replace or not
    /// If it replace, we can have 2 times the same sample
    replacement: bool,
}

impl Sampler for RandomSampler {
    fn new(data_source_len: usize) -> Self {
        RandomSampler {
            data_source_len,
            replacement: false,
        }
    }
}
impl HasLength for RandomSampler {
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
/// Iterator that redurn random index between 0 and `data_source_len`
pub struct RandomSamplerIter {
    data_source_len: usize,
    indexes: Vec<usize>,
    idx: usize,
}

impl RandomSamplerIter {
    /// Create a new `RandomSamplerIter`
    ///
    /// # Arguments
    ///
    /// * `data_source_len` - The len of the dataset.
    /// * `replacement` - Weither we can have the same sample twice over one iteration or not
    fn new(data_source_len: usize, replacement: bool) -> RandomSamplerIter {
        if replacement {
            todo!()
        } else {
            let mut vec: Vec<usize> = (0..data_source_len).collect();
            vec.shuffle(&mut thread_rng());
            RandomSamplerIter {
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

#[test]
fn test_random_sampler() {}
// TODO
// def _test_batch_sampler(self, **kwargs):
//     # [(0, 1), (2, 3, 4), (5, 6), (7, 8, 9), ...]
//     batches = []  # using a regular iterable
//     for i in range(0, 20, 5):
//         batches.append(tuple(range(i, i + 2)))
//         batches.append(tuple(range(i + 2, i + 5)))

//     dl = self._get_data_loader(self.dataset, batch_sampler=batches, **kwargs)
//     self.assertEqual(len(dl), 8)
//     for i, (input, _target) in enumerate(dl):
//         if i % 2 == 0:
//             offset = i * 5 // 2
//             self.assertEqual(len(input), 2)
//             self.assertEqual(input, self.data[offset:offset + 2])
//         else:
//             offset = i * 5 // 2
//             self.assertEqual(len(input), 3)
//             self.assertEqual(input, self.data[offset:offset + 3])
