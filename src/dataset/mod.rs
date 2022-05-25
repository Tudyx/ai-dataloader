pub mod ndarray_dataset;

use ndarray::{Array, ArrayBase, Axis, Dimension, NdIndex, RemoveAxis};

use crate::collate::Collate;
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
    T: Collate<Vec<Self::Output>>,
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
impl<T> Dataset<T> for CustomDataset where T: Collate<Vec<Self::Output>> {}

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
        self.content.len()
    }
}
impl Clone for CustomDataset {
    fn clone(&self) -> Self {
        CustomDataset {
            content: self.content.clone(),
        }
    }
}
impl<T: Clone, U> Dataset<U> for Vec<T> where U: Collate<Vec<Self::Output>> {}

impl<T: Clone> GetItem<usize> for Vec<T> {
    type Output = T;
    fn get_item(&self, index: usize) -> Self::Output {
        self[index].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_custom_dataset() {
        let dataset = CustomDataset {
            content: vec![1, 2, 3, 4, 5],
        };
        println!("{}", dataset.get_item(1));
        println!("{}", dataset.len());
    }
}
