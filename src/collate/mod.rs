//! Merges a list of samples to form a batch.
//!

mod default_collate;
pub use default_collate::DefaultCollate;

/// Any collate gather samples from one batch together.
///
/// A `DefaultCollate` struct is provided which will cover most of the use cases.
///
///
/// This trait is used instead of `Fn` because [we can not currently `impl Fn*` on struct on stable rust](https://github.com/rust-lang/rust/issues/29625).
pub trait Collate<T>: Default {
    /// The type of the collate function's output
    type Output;
    /// Take a batch of samples and collate them
    fn collate(batch: Vec<T>) -> Self::Output;
}

/// Simple Collate that doesn't change the batch of samples.
#[derive(Default)]
pub struct NoOpCollate;

impl<T> Collate<T> for NoOpCollate {
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
        assert_eq!(NoOpCollate::collate(vec![1, 2]), vec![1, 2]);
    }
}
