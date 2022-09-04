use super::super::Collate;
use super::DefaultCollate;
use itertools::Itertools;
use ndarray::{array, Array1};

// vec of tuple
// In python:
// `default_collate([(1, 2.0), (3, 4.0)]) == [tensor([1, 3]), tensor([2., 4.], dtype=torch.float64)]`
// In dataloader_rs:
// `collate(vec![(1, 2.0), (3, 4.0)]) == (Array1[1, 3], Array1[2., 4.])`
// > Note: a python list is roughly equivalent to a Rust mutable tuple

// list of tuple of vector
// In python:
// default_collate(
//  [
//      (tensor(1, 2.0), tensor(3, 4.0)),
//      (tensor(5, 6.0), tensor(7, 8.0)),
//  ]
// )
// ==
// [tensor([[1., 2.],
//    [5., 6.]]),
// tensor([[3., 4.],
//            [7., 8.]])
// ]
//
// In dataloader_rs:
// `collate(vec![(1, 2.0), (3, 4.0)]) == (Array1[1, 3], Array1[2., 4.])`*

// tuple of tensor
// If the content of the tuple are Tensor, they are stacked (using torch stack)
// In python:
// `default_collate((tensor(1, 2.0), tensor(3, 4.0))) == tensor([[1., 2.], [3., 4.]])
// In dataloader_rs:
// TODO: this case is complicated because we would have to specialize

macro_rules! impl_default_collate_vec_tuple {
    ($($name:ident)+) => (
        impl<$($name),+> Collate<($($name,)+)> for DefaultCollate{
            type Output = ($(Array1<$name>,)+);
            #[allow(non_snake_case)]
            fn collate(batch: Vec<($($name,)+)>) -> Self::Output {
                // multiunzip is used to transpose the vec of tuple
                let ($($name,)+) = batch.into_iter().multiunzip();
                // we convert the tuple of vec into a tuple of array
                (
                    $(Array1::from_vec($name),)+
                )
            }
        }
    );
}

impl_default_collate_vec_tuple! { A }
impl_default_collate_vec_tuple! { A B }
impl_default_collate_vec_tuple! { A B C }
impl_default_collate_vec_tuple! { A B C D }
impl_default_collate_vec_tuple! { A B C D E }
impl_default_collate_vec_tuple! { A B C D E F }
impl_default_collate_vec_tuple! { A B C D E F G }
impl_default_collate_vec_tuple! { A B C D E F G H }
impl_default_collate_vec_tuple! { A B C D E F G H I }
impl_default_collate_vec_tuple! { A B C D E F G H I J }
impl_default_collate_vec_tuple! { A B C D E F G H I J K }
impl_default_collate_vec_tuple! { A B C D E F G H I J K L }
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn vec_of_tuple() {
        assert_eq!(
            DefaultCollate::collate(vec![(1, 2)]),
            (array![1], array![2])
        );
        assert_eq!(
            DefaultCollate::collate(vec![(1.0, 2.0), (3.0, 4.0)]),
            (array![1.0, 3.0], array![2.0, 4.0])
        );
        assert_eq!(
            DefaultCollate::collate(vec![(1, 2), (3, 4)]),
            (array![1, 3], array![2, 4])
        );
        assert_eq!(
            DefaultCollate::collate(vec![(-1, 2), (3, 4)]),
            (array![-1, 3], array![2, 4])
        );
        assert_eq!(
            DefaultCollate::collate(vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]),
            (array![1.0, 3.0, 5.0], array![2.0, 4.0, 6.0])
        );
    }
    #[test]
    fn vec_of_tuple_with_len_1() {
        assert_eq!(DefaultCollate::collate(vec![(1,)]), (array![1],));
    }

    #[test]
    fn vec_of_tuple_with_len_2() {
        assert_eq!(
            DefaultCollate::collate(vec![(1, 2.0)]),
            (array![1], array![2.0])
        );
        assert_eq!(
            DefaultCollate::collate(vec![(1, 2.0), (3, 4.0)]),
            (array![1, 3], array![2.0, 4.0])
        );
        assert_eq!(
            DefaultCollate::collate(vec![(-1, true), (-3, false)]),
            (array![-1, -3], array![true, false])
        );
        assert_eq!(
            DefaultCollate::collate(vec![(-1, true), (3, false)]),
            (array![-1, 3], array![true, false])
        );
        assert_eq!(
            DefaultCollate::collate(vec![(1, 2.0), (3, 4.0), (5, 6.0)]),
            (array![1, 3, 5], array![2.0, 4.0, 6.0])
        );
    }
    #[test]
    fn vec_of_tuple_with_len_3() {
        assert_eq!(
            DefaultCollate::collate(vec![(1, 2.0, true)]),
            (array![1], array![2.0], array![true])
        );
        assert_eq!(
            DefaultCollate::collate(vec![(1, 2.0, true), (3, 4.0, true)]),
            (array![1, 3], array![2.0, 4.0], array![true, true])
        );
        assert_eq!(
            DefaultCollate::collate(vec![(1, 2.0, true), (3, 4.0, false), (5, 6.0, true)]),
            (
                array![1, 3, 5],
                array![2.0, 4.0, 6.0],
                array![true, false, true]
            )
        );
    }

    /// For now it's find that strings like are puts inside [`ndarray`] but this will not be possible when
    /// with a real tensor library. A tensor can't support string (not tokenized).
    /// In that case they must be put inside [`Vec`] or tuple.
    #[test]
    fn specialized_vec_of_tuple() {
        assert_eq!(
            DefaultCollate::collate(vec![(1, "lol"), (1, "mdrr"), (0, "serious")]),
            (array![1, 1, 0], array!["lol", "mdrr", "serious"])
        );
    }
}
