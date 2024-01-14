use super::super::Collate;
use super::TorchCollate;
use ndarray::{stack, Array, ArrayBase, ArrayView, Axis, Dimension, RemoveAxis};
use tch::Tensor;

impl<A, D> Collate<Array<A, D>> for TorchCollate
where
    A: Clone + tch::kind::Element,
    D: Dimension,
    D::Larger: RemoveAxis,
{
    type Output = Tensor;
    fn collate(&self, batch: Vec<Array<A, D>>) -> Self::Output {
        // Convert it to a `Vec` of view.
        let vec_of_view: Vec<ArrayView<'_, A, D>> = batch.iter().map(ArrayBase::view).collect();
        // TODO: maybe use tensor stack here
        let array = stack(Axis(0), vec_of_view.as_slice())
            .expect("Make sure you're items from the dataset have the same shape.");

        let tensor = Tensor::from_slice(array.as_slice().unwrap());
        #[allow(clippy::cast_possible_wrap)]
        let shape = array
            .shape()
            .iter()
            .map(|dim| *dim as i64)
            .collect::<Vec<_>>();
        tensor.reshape(shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn keep_dimension() {
        let batch = TorchCollate.collate(vec![array![1, 2], array![3, 4]]);
        assert_eq!(batch.dim(), 2);
        batch.print();
    }

    // #[test]
    // fn foo() {
    //     println!("has_cuda: {}", tch::utils::has_cuda());

    //     let array = vec![0; 1_000_000];
    //     let array = Array::from_vec(array);
    //     for i in 1..1_000_000 {
    //         let t = TorchCollate::default().collate(vec![&array]);
    //         println!("{} {:?}", i, t.size())
    //     }
    //     assert!(false);
    // }

    #[test]
    fn nested() {
        // If a type is an ndarray it get converted into tensor. But what if this tensor needs to be collated again?.
        // Look at the supported type if this case can happen.
    }
}
