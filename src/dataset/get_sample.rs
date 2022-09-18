use std::collections::VecDeque;

/// Return a sample from the dataset at a given index
pub trait GetSample {
    /// Type of one sample of the dataset
    type Sample: Sized;
    /// Return the dataset sample corresponding to the index
    fn get_sample(&self, index: usize) -> Self::Sample;
}

impl<T: Clone> GetSample for Vec<T> {
    type Sample = T;
    fn get_sample(&self, index: usize) -> Self::Sample {
        self[index].clone()
    }
}

impl<T: Clone> GetSample for VecDeque<T> {
    type Sample = T;
    fn get_sample(&self, index: usize) -> Self::Sample {
        self[index].clone()
    }
}
