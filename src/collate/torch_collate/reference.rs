use super::TorchCollate;
use crate::collate::Collate;

/// We think it makes no sense to but a bench of reference into a Tensor. That's why if the dataset yield reference and they
/// are cheap to copy, we copy them.
/// It is useful for having a non-consuming `Iterator` over the `Dataloader`.
impl<T> Collate<&T> for TorchCollate
where
    T: Copy,
    Self: Collate<T>,
{
    type Output = <Self as Collate<T>>::Output;
    fn collate(batch: Vec<&T>) -> Self::Output {
        TorchCollate::collate(batch.into_iter().cloned().collect())
    }
}
