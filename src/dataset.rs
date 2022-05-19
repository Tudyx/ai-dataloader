use ndarray::{Array, Dimension, NdIndex};

use crate::collate::Collect;
use crate::sampler::HasLength;

// TODO: remove the clone trait
// pub trait Dataset: HasLength + Index<usize> + Clone {type SampleType;}

// version du trait générique ou T est le type d'élement retourner par
// l'indexation du dataset. Par exemple T peut-être un tuple (str, Vec<i32>)

// require associated type bound :
// pub trait Dataset: HasLength + Index<usize, Output : Sized> + Clone
// where
//     DefaultCollector: Collect<Vec<Self::Output>>
// {}

pub trait Dataset<T>: HasLength + GetItem<usize> + Clone
where
    T: Collect<Vec<Self::Output>>,
{
}

pub trait Dataset3<F>: HasLength + GetItem<usize> + Clone
where
    F: Fn(Vec<Self::Output>) -> Self::CollateOutput,
{
    type CollateOutput;
}

// alternative au trait index mais où l'output est forcé d'avoir
// une size connu au compile time
pub trait GetItem<Idx: Sized = usize> {
    type Output: Sized;
    // real one
    // fn get_item(&self, index: Idx) -> &Self::Output;
    fn get_item(&self, index: Idx) -> Self::Output;
    // where Collect<Vec<Self::Output>>;
}
pub trait VecCollectable {}
pub trait GetItem2<Idx: Sized = usize> {
    type Output: Sized;
    // real one
    // fn get_item(&self, index: Idx) -> &Self::Output;
    fn get_item(&self, index: Idx) -> Self::Output;
}

// j'essaye d'ajouter une contrainte sur l'associade type de Index
// pub trait DatasetPlus : Dataset + Index<usize, Index::Output: T>{}

struct CustomDataset {
    pub content: Vec<i32>,
}
impl<T> Dataset<T> for CustomDataset where T: Collect<Vec<Self::Output>> {}

// impl Index<usize> for CustomDataset {
//     type Output = i32;
//     fn index(&self, index: usize) -> &Self::Output {
//         &self.content[index]
//     }
// }
impl GetItem<usize> for CustomDataset {
    type Output = i32;
    fn get_item(&self, index: usize) -> Self::Output {
        self.content[index]
    }
}
impl HasLength for CustomDataset {
    fn len(&self) -> usize {
        return self.content.len();
    }
}
impl Clone for CustomDataset {
    fn clone(&self) -> Self {
        CustomDataset {
            content: self.content.clone(),
        }
    }
}
impl<T: Clone, U> Dataset<U> for Vec<T> where U: Collect<Vec<Self::Output>> {}

impl<T: Clone> GetItem<usize> for Vec<T> {
    type Output = T;
    fn get_item(&self, index: usize) -> Self::Output {
        self[index].clone()
    }
}

// class TensorDataset(Dataset[Tuple[Tensor, ...]]):
//     r"""Dataset wrapping tensors.

//     Each sample will be retrieved by indexing tensors along the first dimension.

//     Args:
//         *tensors (Tensor): tensors that have the same size of the first dimension.
//     """
//     tensors: Tuple[Tensor, ...]

//     def __init__(self, *tensors: Tensor) -> None:
//         assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
//         self.tensors = tensors

//     def __getitem__(self, index):
//         return tuple(tensor[index] for tensor in self.tensors)

//     def __len__(self):
//         return self.tensors[0].size(0)
pub struct NdarrayDataset<A1, A2, D1, D2>
where
    A1: Clone,
    A2: Clone,
    D1: Dimension,
    D2: Dimension,
{
    pub ndarrays: (Array<A1, D1>, Array<A2, D2>),
}
impl<A1, A2, D1, D2, T> Dataset<T> for NdarrayDataset<A1, A2, D1, D2>
where
    A1: Clone,
    A2: Clone,
    D1: Dimension,
    D2: Dimension,
    usize: NdIndex<D1>,
    usize: NdIndex<D2>,
    T: Collect<Vec<Self::Output>>,
{
}

impl<A1, A2, D1, D2> Clone for NdarrayDataset<A1, A2, D1, D2>
where
    A1: Clone,
    A2: Clone,
    D1: Dimension,
    D2: Dimension,
{
    fn clone(&self) -> Self {
        NdarrayDataset {
            ndarrays: self.ndarrays.clone(),
        }
    }
}

impl<A1, A2, D1, D2> HasLength for NdarrayDataset<A1, A2, D1, D2>
where
    A1: Clone,
    A2: Clone,
    D1: Dimension,
    D2: Dimension,
{
    fn len(&self) -> usize {
        return self.ndarrays.0.len();
    }
}
impl<A1, A2, D1, D2> GetItem<usize> for NdarrayDataset<A1, A2, D1, D2>
where
    usize: NdIndex<D1>,
    usize: NdIndex<D2>,
    A1: Clone,
    A2: Clone,
    D1: Dimension,
    D2: Dimension,
{
    type Output = (A1, A2);
    fn get_item(&self, index: usize) -> Self::Output {
        (
            self.ndarrays.0[index].clone(),
            self.ndarrays.1[index].clone(),
        )
    }
}

#[cfg(test)]
mod tests {

    use ndarray::array;

    use super::*;

    #[test]
    fn test_custom_dataset() {
        let dataset = CustomDataset {
            content: vec![1, 2, 3, 4, 5],
        };
        println!("{}", dataset.get_item(1));
        println!("{}", dataset.len());
    }
    #[test]
    fn ndarray_dataset() {
        let dataset = NdarrayDataset {
            ndarrays: (array![1, 2], array![3, 4]),
        };
        let sample = dataset.get_item(1);
        println!("{sample:?}");
    }
}
