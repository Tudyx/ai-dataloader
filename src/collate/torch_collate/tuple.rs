use super::super::Collate;
use super::TorchCollate;
use itertools::Itertools;

// Maybe an implementation passing the length and the index of elements to the macro could be more efficient than with the
// `Iterttols::multiunzip`.

/// `tuple` implementation, up to 16 elements.
macro_rules! tuple_impl {
    ($($name:ident)+) => {
        impl<$($name),+> Collate<($($name,)+)> for TorchCollate
        where
            $($name: Clone,)+
            $(TorchCollate: Collate<$name>,)+

        {
            type Output = ($(<TorchCollate as Collate<$name>>::Output,)+);

            #[allow(non_snake_case)]
            fn collate(batch: Vec<($($name,)+)>) -> Self::Output {
                let copy = batch.to_vec();
                let ($($name,)+) = copy.into_iter().multiunzip();
                (
                    $(TorchCollate::collate($name),)+
                )

            }
        }
    };
}

tuple_impl! { A }
tuple_impl! { A B }
tuple_impl! { A B C }
tuple_impl! { A B C D }
tuple_impl! { A B C D E }
tuple_impl! { A B C D E F }
tuple_impl! { A B C D E F G }
tuple_impl! { A B C D E F G H }
tuple_impl! { A B C D E F G H I }
tuple_impl! { A B C D E F G H I J }
tuple_impl! { A B C D E F G H I J K }
tuple_impl! { A B C D E F G H I J K L }

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn vec_of_tuple() {
        assert_eq!(TorchCollate::collate(vec![(1, 2)]), (array![1], array![2]));
        assert_eq!(
            TorchCollate::collate(vec![(1.0, 2.0), (3.0, 4.0)]),
            (array![1.0, 3.0], array![2.0, 4.0])
        );
        assert_eq!(
            TorchCollate::collate(vec![(1, 2), (3, 4)]),
            (array![1, 3], array![2, 4])
        );
        assert_eq!(
            TorchCollate::collate(vec![(-1, 2), (3, 4)]),
            (array![-1, 3], array![2, 4])
        );
        assert_eq!(
            TorchCollate::collate(vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]),
            (array![1.0, 3.0, 5.0], array![2.0, 4.0, 6.0])
        );
    }
    #[test]
    fn vec_of_tuple_with_len_1() {
        assert_eq!(TorchCollate::collate(vec![(1,)]), (array![1],));
    }

    #[test]
    fn vec_of_tuple_with_len_2() {
        assert_eq!(
            TorchCollate::collate(vec![(1, 2.0)]),
            (array![1], array![2.0])
        );
        assert_eq!(
            TorchCollate::collate(vec![(1, 2.0), (3, 4.0)]),
            (array![1, 3], array![2.0, 4.0])
        );
        assert_eq!(
            TorchCollate::collate(vec![(-1, true), (-3, false)]),
            (array![-1, -3], array![true, false])
        );
        assert_eq!(
            TorchCollate::collate(vec![(-1, true), (3, false)]),
            (array![-1, 3], array![true, false])
        );
        assert_eq!(
            TorchCollate::collate(vec![(1, 2.0), (3, 4.0), (5, 6.0)]),
            (array![1, 3, 5], array![2.0, 4.0, 6.0])
        );
    }
    #[test]
    fn vec_of_tuple_with_len_3() {
        assert_eq!(
            TorchCollate::collate(vec![(1, 2.0, true)]),
            (array![1], array![2.0], array![true])
        );
        assert_eq!(
            TorchCollate::collate(vec![(1, 2.0, true), (3, 4.0, true)]),
            (array![1, 3], array![2.0, 4.0], array![true, true])
        );
        assert_eq!(
            TorchCollate::collate(vec![(1, 2.0, true), (3, 4.0, false), (5, 6.0, true)]),
            (
                array![1, 3, 5],
                array![2.0, 4.0, 6.0],
                array![true, false, true]
            )
        );
    }
}
