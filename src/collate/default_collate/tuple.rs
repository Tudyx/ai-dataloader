use super::super::Collate;
use super::DefaultCollator;
use itertools::Itertools;
use ndarray::{array, Array1};

macro_rules! impl_default_collate_vec_tuple {
    ($($name:ident)+) => (
        impl<$($name),+> Collate<Vec<($($name,)+)>> for DefaultCollator{
            type Output = ($(Array1<$name>,)+);
            #[allow(non_snake_case)]
            fn collate(batch: Vec<($($name,)+)>) -> Self::Output {
                let ($($name,)+) = batch.into_iter().multiunzip();
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

macro_rules! impl_tuple_collect {
    ($($t:ty)*) => {
        $(
            impl Collate<($t,)> for DefaultCollator {
                type Output = Array1<$t>;
                fn collate(batch: ($t,)) -> Self::Output {
                    array![batch.0]
                }
            }
        )*
    };
}
impl_tuple_collect!(usize u8 u16 u32 u64 u128
            isize i8 i16 i32 i64 i128
            f32 f64
            bool char);

macro_rules! impl_tuple_collect2 {
    ($($t:ty)*) => {
        $(
            impl Collate<($t,$t)> for DefaultCollator {
                type Output = Array1<$t>;
                fn collate(batch: ($t,$t)) -> Self::Output {
                    array![batch.0, batch.1]
                }
            }
        )*
    };
}
impl_tuple_collect2!(usize u8 u16 u32 u64 u128
                isize i8 i16 i32 i64 i128
                f32 f64
                bool char);

macro_rules! impl_tuple_collect3 {
    ($($t:ty)*) => {
        $(
            impl Collate<($t,$t,$t)> for DefaultCollator {
                type Output = Array1<$t>;
                fn collate(batch: ($t,$t,$t)) -> Self::Output {
                    array![batch.0, batch.1, batch.2]
                }
            }
        )*
    };
}
impl_tuple_collect3!(usize u8 u16 u32 u64 u128
                    isize i8 i16 i32 i64 i128
                    f32 f64
                    bool char);

// TODO: concatenate the `ndarray` as it's done for tensor
// problem :  it won't give an Array1<T> like the other
impl_tuple_collect!(Array1<f64>);
impl_tuple_collect2!(Array1<f64>);
impl_tuple_collect3!(Array1<f64>);

impl Collate<(String, String)> for DefaultCollator {
    type Output = (String, String);
    fn collate(batch: (String, String)) -> Self::Output {
        batch
    }
}
impl Collate<(String, String, String)> for DefaultCollator {
    type Output = (String, String, String);
    fn collate(batch: (String, String, String)) -> (String, String, String) {
        batch
    }
}

// build seems to never ends when we uncomment this
// impl<T> Collate<((T, T), (T, T))> for DefaultCollator
// where
//     T: Clone,
//     DefaultCollator: Collate<(T, T), Output = Array<T, Ix1>>,
// {
//     type Output = Vec<Array<T, Ix1>>;
//     fn collate(batch: ((T, T), (T, T))) -> Self::Output {
//         // This tuple is homogeneous, convert it into an array to be iterable
//         let a = [batch.0 .0, batch.0 .1];
//         let b = [batch.1 .0, batch.1 .1];

//         let mut res = vec![];
//         for samples in izip!(a, b) {
//             res.push(DefaultCollator::collate(samples));
//         }
//         res
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn vec_of_tuple() {
        assert_eq!(
            DefaultCollator::collate(vec![(1, 2)]),
            (array![1], array![2])
        );
        assert_eq!(
            DefaultCollator::collate(vec![(1.0, 2.0), (3.0, 4.0)]),
            (array![1.0, 3.0], array![2.0, 4.0])
        );
        assert_eq!(
            DefaultCollator::collate(vec![(1, 2), (3, 4)]),
            (array![1, 3], array![2, 4])
        );
        assert_eq!(
            DefaultCollator::collate(vec![(-1, 2), (3, 4)]),
            (array![-1, 3], array![2, 4])
        );
        assert_eq!(
            DefaultCollator::collate(vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]),
            (array![1.0, 3.0, 5.0], array![2.0, 4.0, 6.0])
        );
    }
    #[test]
    fn vec_of_tuple_with_len_1() {
        assert_eq!(DefaultCollator::collate(vec![(1,)]), (array![1],));
    }
    // En python
    // default_collate([(1, 2.0), (3, 4.0)]) == [tensor([1, 3]), tensor([2., 4.], dtype=torch.float64)]
    // Mais une list peut avoir des élémenst de type différents en python
    // C'est pourquoi j'essaye de remplacer par un tuple
    #[test]
    fn vec_of_tuple_with_len_2() {
        assert_eq!(
            DefaultCollator::collate(vec![(1, 2.0)]),
            (array![1], array![2.0])
        );
        assert_eq!(
            DefaultCollator::collate(vec![(1, 2.0), (3, 4.0)]),
            (array![1, 3], array![2.0, 4.0])
        );
        assert_eq!(
            DefaultCollator::collate(vec![(-1, true), (-3, false)]),
            (array![-1, -3], array![true, false])
        );
        assert_eq!(
            DefaultCollator::collate(vec![(-1, true), (3, false)]),
            (array![-1, 3], array![true, false])
        );
        assert_eq!(
            DefaultCollator::collate(vec![(1, 2.0), (3, 4.0), (5, 6.0)]),
            (array![1, 3, 5], array![2.0, 4.0, 6.0])
        );
    }
    #[test]
    fn vec_of_tuple_with_len_3() {
        assert_eq!(
            DefaultCollator::collate(vec![(1, 2.0, true)]),
            (array![1], array![2.0], array![true])
        );
        assert_eq!(
            DefaultCollator::collate(vec![(1, 2.0, true), (3, 4.0, true)]),
            (array![1, 3], array![2.0, 4.0], array![true, true])
        );
        assert_eq!(
            DefaultCollator::collate(vec![(1, 2.0, true), (3, 4.0, false), (5, 6.0, true)]),
            (
                array![1, 3, 5],
                array![2.0, 4.0, 6.0],
                array![true, false, true]
            )
        );
    }

    #[test]
    fn tuple() {
        // TODO: macro testing
        // (T,)
        assert_eq!(DefaultCollator::collate((1,)), array![1]);
        assert_eq!(DefaultCollator::collate((-1,)), array![-1]);
        assert_eq!(DefaultCollator::collate((1.0,)), array![1.0]);
        assert_eq!(DefaultCollator::collate((true,)), array![true]);
        // (T, T)
        assert_eq!(DefaultCollator::collate((1, 2)), array![1, 2]);
        assert_eq!(DefaultCollator::collate((-1, 2)), array![-1, 2]);
        assert_eq!(DefaultCollator::collate((1.0, 2.0)), array![1.0, 2.0]);
        assert_eq!(DefaultCollator::collate((true, false)), array![true, false]);
        // (T, T, T)
        assert_eq!(DefaultCollator::collate((1, 2, 3)), array![1, 2, 3]);
        assert_eq!(DefaultCollator::collate((-1, 2, 3)), array![-1, 2, 3]);
        assert_eq!(
            DefaultCollator::collate((1.0, 2.0, 3.0)),
            array![1.0, 2.0, 3.0]
        );
        assert_eq!(
            DefaultCollator::collate((true, false, true)),
            array![true, false, true]
        );
    }
    #[test]
    fn tuple_of_array() {
        // Warning! in the python version this return a tensor de shape [2,2]
        assert_eq!(
            DefaultCollator::collate((array![1.0, 2.0], array![3.0, 4.0])),
            array![array![1.0, 2.0], array![3.0, 4.0]]
        );
    }
}
