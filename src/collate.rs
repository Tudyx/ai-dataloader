//! Merges a list of samples to form a batch.
//!

mod default_collate;
pub use default_collate::DefaultCollate;

#[cfg(feature = "tch")]
#[cfg_attr(docsrs, doc(cfg(feature = "tch")))]
mod torch_collate;
#[cfg(feature = "tch")]
#[cfg_attr(docsrs, doc(cfg(feature = "tch")))]
pub use torch_collate::TorchCollate;

/// Any collate gather samples from one batch together.
///
/// A `DefaultCollate` struct is provided which will cover most of the use cases.
///
///
/// This trait is used instead of `Fn` because [we cannot currently `impl Fn*` on struct on stable rust](https://github.com/rust-lang/rust/issues/29625).
pub trait Collate<T> {
    /// The type of the collate function's output
    type Output;
    /// Take a batch of samples and collate them
    fn collate(&self, batch: Vec<T>) -> Self::Output;
}

// Allow user to specify closure as collate function.
impl<T, F, O> Collate<T> for F
where
    F: Fn(Vec<T>) -> O,
{
    type Output = O;
    fn collate(&self, batch: Vec<T>) -> Self::Output {
        (self)(batch)
    }
}

/// Simple Collate that doesn't change the batch of samples.
#[derive(Default, Debug)]
pub struct NoOpCollate;

impl<T> Collate<T> for NoOpCollate {
    type Output = Vec<T>;
    fn collate(&self, batch: Vec<T>) -> Self::Output {
        batch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_op_collate() {
        assert_eq!(NoOpCollate.collate(vec![1, 2]), vec![1, 2]);
    }

    #[test]
    fn no_op_collate_closure() {
        let collate = |x| x;
        assert_eq!(collate.collate(vec![1, 2]), vec![1, 2]);
    }
}
