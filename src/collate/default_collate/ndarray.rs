use super::super::Collate;
use super::DefaultCollate;
use ndarray::{stack, Array, ArrayBase, ArrayView, Axis, Dimension, RemoveAxis};

impl<A, D> Collate<Array<A, D>> for DefaultCollate
where
    A: Clone,
    D: Dimension,
    D::Larger: RemoveAxis,
{
    type Output = Array<A, <D as Dimension>::Larger>;
    fn collate(batch: Vec<Array<A, D>>) -> Self::Output {
        // Convert it to a `Vec` of view.
        let vec_of_view: Vec<ArrayView<'_, A, D>> = batch.iter().map(ArrayBase::view).collect();
        stack(Axis(0), vec_of_view.as_slice())
            .expect("Make sure you're items from the dataset have the same shape.")
    }
}
