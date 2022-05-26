pub mod ndarray_dataset;
use crate::collate::Collate;
use crate::sampler::HasLength;

/// A dataset is just something that has a length and is indexable
///
/// We use a custom [GetItem] trait instead of `std::ops::Index` because
/// it provides more flexibility.
/// Indeed we could have provide this implementation:
/// ```
/// use dataloader_rs::collate::Collate;
/// use dataloader_rs::sampler::HasLength;
///
/// pub trait Dataset<T>: HasLength + std::ops::Index<usize>
/// where
/// T: Collate<Vec<Self::Output>>,
/// Self::Output: Sized,
/// {
/// }
/// ```
/// But as `Index::Output` must refer as something exist, it will not cover most of our use cases.
/// For instance if the dataset is something like that:
/// ```
/// struct Dataset {
///     labels: Vec<i32>,
///     texts: Vec<String>,
/// }
/// ```
/// And we want to return a tuple (label, text) when indexing, it will no be possible with `std:ops::Index`
pub trait Dataset<T>: HasLength + GetItem
where
    T: Collate<Vec<Self::Output>>,
{
}

pub trait Dataset3<F>: HasLength + GetItem
where
    F: Fn(Vec<Self::Output>) -> Self::CollateOutput,
{
    type CollateOutput;
}

/// Return an item of the dataset
pub trait GetItem {
    /// Dataset sample type
    type Output: Sized;
    /// Return the dataset element corresponding to the index
    fn get_item(&self, index: usize) -> Self::Output;
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
impl GetItem for CustomDataset {
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

impl<T: Clone> GetItem for Vec<T> {
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
