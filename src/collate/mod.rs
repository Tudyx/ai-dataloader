pub mod default_collate;

/// Any collate gather samples from one batch together.
/// This trait can be seen as a functor.
/// The `default trait` make it possible to create the functor.
pub trait Collate<T>: Default {
    /// The type of the collate function's output
    type Output;
    /// Take a batch of samples and collate them
    fn collate(batch: Vec<T>) -> Self::Output;
}

/// Simple collator that doesn't change the batch of samples.
#[derive(Default)]
pub struct NoOpCollator;

impl<T> Collate<T> for NoOpCollator {
    type Output = Vec<T>;
    fn collate(batch: Vec<T>) -> Self::Output {
        batch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_op_collate() {
        assert_eq!(NoOpCollator::collate(vec![1, 2]), vec![1, 2]);
    }
}
