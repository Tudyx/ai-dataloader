use super::Collate;

use itertools::izip;
use itertools::Itertools;
use ndarray::{array, Array, Array1, Dimension, Ix1};
use std::collections::HashMap;

/// Basic collator that mimic the default collate function from PyTorch
/// As they are no such lib whith the same functionnality as PyTorch tensor in rust,
/// data is collated inside ndarray. Ndarray is the rust equivalent of numpy.ndarray with
/// almost the same capabilities. Nevertheless, they can't run on the GPU.
/// This function is always call with a Vec of data but then can be called recursively
///
/// Basic transformation implemented for the default collator :
/// ```md
/// - Vec<Scalar> -> ndarray<scalar>
/// - Vec<tuple> -> tuple(ndarray)
/// - Vec<HashMap<Key, Value>> -> HasMap<Key, DefaultCollator::collate(Vec<Value>)
/// - Vec<Array> -> ?
/// - Vec<Vec> -> ?
/// ```
///
#[derive(Default, Debug)]
pub struct DefaultCollator;

macro_rules! impl_vec_collect {
    ($($t:ty)*) => {
        $(
        impl Collate<Vec<$t>> for DefaultCollator {
            type Output = Array<$t, Ix1>;
            fn collate(batch: Vec<$t>) -> Self::Output {
                Array::from_vec(batch)
            }
        }
    )*
    };
}
impl_vec_collect!(usize u8 u16 u32 u64 u128
    isize i8 i16 i32 i64 i128
    f32 f64
    bool char);

// TODO: check que cela s'apelle bien un array
macro_rules! impl_array_collect {
        ($($t:ty)*) => {
            $(
            impl<const N: usize> Collate<[$t; N]> for DefaultCollator {
                type Output = Array<$t, Ix1>;
                fn collate(batch: [$t; N]) -> Self::Output {
                    Array::from_vec(batch.to_vec())
                }
            }
        )*
        };
    }
impl_array_collect!(usize u8 u16 u32 u64 u128
        isize i8 i16 i32 i64 i128
        f32 f64
        bool char);

macro_rules! impl_tuple_collect {
            ($($t:ty)*) => {
                $(
                impl Collate<($t,)> for DefaultCollator {
                    type Output = Array<$t, Ix1>;
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
                        type Output = Array<$t, Ix1>;
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
                            type Output = Array<$t, Ix1>;
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

// Todo macro for unpacking the vec
// macro_rules! unpack_vec {
//     ($element:expr; $len:expr) => {

//     };
// }
impl Collate<Vec<Vec<i32>>> for DefaultCollator {
    type Output = Vec<Array<i32, Ix1>>;
    fn collate(batch: Vec<Vec<i32>>) -> Self::Output {
        let elem_size = batch.get(0).unwrap().len();
        if !batch.iter().all(|vec| vec.len() == elem_size) {
            panic!("each element in list of batch should be of equal size");
        }
        let mut res = Vec::new();
        // I don't find a way to unpack a vec of vec.
        // Maybe i can turn this into a macro
        if batch.len() == 1 {
            res.push(DefaultCollator::collate(batch[0].clone()));
        } else if batch.len() == 2 {
            for samples in izip!(batch[0].clone(), batch[1].clone()) {
                res.push(DefaultCollator::collate(samples));
            }
        } else if batch.len() == 3 {
            for samples in izip!(batch[0].clone(), batch[1].clone(), batch[2].clone()) {
                res.push(DefaultCollator::collate(samples));
            }
        }
        res
    }
}

// comme pour les string sont transformé en tuple pour les strings il peut y avoir plusieurs type de retour..
// Cela retourne un tuple de la taille du vec.
impl Collate<Vec<Vec<String>>> for DefaultCollator {
    type Output = Vec<(String, String)>;
    fn collate(batch: Vec<Vec<String>>) -> Self::Output {
        let elem_size = batch.get(0).unwrap().len();
        if !batch.iter().all(|vec| vec.len() == elem_size) {
            panic!("each element in list of batch should be of equal size");
        }
        let mut res = Vec::new();

        // I don't find a way to unpack a vec of vec.
        // Maybe i can turn this into a macro
        if batch.len() == 1 {
            // res.push(DefaultCollector::collect(batch[0].clone()));
        } else if batch.len() == 2 {
            for samples in izip!(batch[0].clone(), batch[1].clone()) {
                res.push(DefaultCollator::collate(samples));
            }
        } else if batch.len() == 3 {
            //for samples in izip!(batch[0].clone(), batch[1].clone(), batch[2].clone()) {
            // res.push(DefaultCollector::collect(samples));
            //}
        }
        res
    }
}
// impl<T> Collect<Vec<Vec<T>>> for DefaultCollector
// where
//     DefaultCollector: Collate<Vec<T>>,
// {
//     type Output = <DefaultCollector as Collect<Vec<T>>>::Output;
//     fn collect(batch: Vec<Vec<T>>) -> Self::Output {
//         let elem_size = batch.iter().next().unwrap().len();
//         if !batch.iter().all(|vec| vec.len() == elem_size) {
//             panic!("each element in list of batch should be of equal size");
//         }
//         let mut res = Vec::new();

//         // I don't find a way to unpack a vec of vec.
//         // Maybe i can turn this into a macro
//         if batch.len() == 1 {
//             res.push(DefaultCollector::collect(batch[0].clone()));
//         } else if batch.len() == 2 {
//             for samples in izip!(batch[0].clone(), batch[1].clone()) {
//                 res.push(DefaultCollector::collect(samples))
//             }
//         } else if batch.len() == 3 {
//             for samples in izip!(batch[0].clone(), batch[1].clone(), batch[2].clone()) {
//                 res.push(DefaultCollector::collect(samples));
//             }
//         }
//         res
//     }
// }
impl<const N: usize> Collate<Vec<[i32; N]>> for DefaultCollator {
    type Output = Vec<Array<i32, Ix1>>;
    fn collate(batch: Vec<[i32; N]>) -> Vec<Array<i32, Ix1>> {
        let mut it = batch.iter();
        let elem_size = it.next().unwrap().len();
        if !batch.iter().all(|vec| vec.len() == elem_size) {
            panic!("each element in list of batch should be of equal size");
        }
        let mut res = Vec::new();
        if batch.len() == 1 {
            res.push(DefaultCollator::collate(batch[0]));
        } else if batch.len() == 2 {
            for samples in izip!(batch[0], batch[1]) {
                res.push(DefaultCollator::collate(samples))
            }
        } else if batch.len() == 3 {
            for samples in izip!(batch[0], batch[1], batch[2]) {
                res.push(DefaultCollator::collate(samples));
            }
        }
        res
    }
}

macro_rules! impl_default_collate_vec_hash_map {
    ($($t:ty)*) => {
        $(
            /// HasMap implementation for default collate. We can only be generic over keys, see comment below.
            impl<K> Collate<Vec<HashMap<K, $t>>> for DefaultCollator
            where
                K: std::cmp::Eq + std::hash::Hash + Clone,
            {
                type Output = HashMap<K, <DefaultCollator as Collate<Vec<$t>>>::Output>;

                fn collate(batch: Vec<HashMap<K, $t>>) -> Self::Output {
                    let mut res = HashMap::with_capacity(batch[0].keys().len());
                    for key in batch[0].keys() {
                        let vec: Vec<_> = batch.iter().map(|hash_map| hash_map[key].clone()).collect();
                        res.insert(key.clone(), DefaultCollator::collate(vec));
                    }
                    res
                }
            }
        )*
    };
}

impl_default_collate_vec_hash_map!(usize u8 u16 u32 u64 u128
isize i8 i16 i32 i64 i128
f32 f64
bool char
String
);

/// String slice require a specific implementation because of lifetime
impl<'a, K> Collate<Vec<HashMap<K, &'a str>>> for DefaultCollator
where
    K: std::cmp::Eq + std::hash::Hash + Clone,
{
    type Output = HashMap<K, <DefaultCollator as Collate<Vec<&'a str>>>::Output>;

    fn collate(batch: Vec<HashMap<K, &'a str>>) -> Self::Output {
        let mut res = HashMap::with_capacity(batch[0].keys().len());
        for key in batch[0].keys() {
            let vec: Vec<_> = batch.iter().map(|hash_map| hash_map[key].clone()).collect();
            res.insert(key.clone(), DefaultCollator::collate(vec));
        }
        res
    }
}

///
/// The trait bound can't work because it's mean that if V is a HashMap we should also be able to
/// collate it but were are defining here how to collate an HasMap
/// ```ignore
/// use std::collections::HashMap;
/// use dataloader_rs::collate::Collate;
/// use dataloader_rs::collate::default_collate::DefaultCollator;
///
/// impl<K, V> Collate<Vec<HashMap<K, V>>> for DefaultCollator
/// where
///     DefaultCollator: Collate<Vec<V>>,
/// {
///     type Output = HashMap<K, <DefaultCollator as Collate<Vec<V>>>::Output>;
///
///     fn collate(batch: Vec<HashMap<K, V>>) -> Self::Output {
///         let mut res = HashMap::new();
///         for key in batch[0].keys() {
///             let mut vec = Vec::new();
///             for d in &batch {
///                 vec.push(d[key]);
///             }
///             res.insert(*key, DefaultCollator::collate(vec));
///         }
///         res
///     }
/// }
/// ```

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
// impl Collate<Vec<(String, String)>> for DefaultCollator {
//     type Output = Vec<(String, String)>;
//     fn collate(batch: Vec<(String, String)>) -> Vec<(String, String)> {
//         batch
//     }
// }

/////////////////////////// case that require specialization ///////////////////////////
//  byte and string are a no op for default collector
impl Collate<String> for DefaultCollator {
    type Output = String;
    fn collate(batch: String) -> Self::Output {
        batch
    }
}
impl Collate<Vec<String>> for DefaultCollator {
    type Output = Vec<String>;
    fn collate(batch: Vec<String>) -> Self::Output {
        batch
    }
}
impl<'a> Collate<Vec<&'a str>> for DefaultCollator {
    type Output = Vec<&'a str>;
    fn collate(batch: Vec<&'a str>) -> Self::Output {
        batch
    }
}

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

// implémentation pour n'importe quelle dimension on fait rien ndarray -> ndarray FAUX (ndarray of a ndarray)
impl<T, I: Dimension> Collate<Array<T, I>> for DefaultCollator {
    type Output = Array<T, I>;
    fn collate(batch: Array<T, I>) -> Array<T, I> {
        batch
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn primitive_type() {
        assert_eq!(
            DefaultCollator::collate(vec![0, 1, 2, 3, 4, 5]),
            array![0, 1, 2, 3, 4, 5]
        );
        assert_eq!(
            DefaultCollator::collate(vec![0., 1., 2., 3., 4., 5.]),
            array![0., 1., 2., 3., 4., 5.]
        );
        assert_eq!(DefaultCollator::collate([1, 2, 3]), array![1, 2, 3]);
        assert_eq!(DefaultCollator::collate([1, -2, 3]), array![1, -2, 3]);
        assert_eq!(DefaultCollator::collate([1., 2., 3.]), array![1., 2., 3.]);
        assert_eq!(
            DefaultCollator::collate([true, false, true]),
            array![true, false, true]
        );
    }
    #[test]
    fn ndarray() {
        // should be a no op for primitive type
        assert_eq!(
            DefaultCollator::collate(array![1, 2, 3, 4]),
            array![1, 2, 3, 4]
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
    fn tuple_of_tuple() {
        // assert_eq!(
        //     DefaultCollator::collate(((1, 2), (3, 4))),
        //     vec![array![1, 3], array![2, 4]]
        // );
        // assert_eq!(
        //     DefaultCollator::collate(((1.0, 2.0), (3.0, 4.0))),
        //     vec![array![1.0, 3.0], array![2.0, 4.0]]
        // );
    }
    #[test]
    fn tuple_of_array() {
        // Warning! in the python version this return a tensor de shape [2,2]
        assert_eq!(
            DefaultCollator::collate((array![1.0, 2.0], array![3.0, 4.0])),
            array![array![1.0, 2.0], array![3.0, 4.0]]
        );
    }
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
    fn vec_of_vec() {
        assert_eq!(DefaultCollator::collate(vec![vec![1]]), vec![array![1]]);
        assert_eq!(
            DefaultCollator::collate(vec![vec![1, 2], vec![3, 4]]),
            vec![array![1, 3], array![2, 4]]
        );

        assert_eq!(
            DefaultCollator::collate(vec![vec![1, 2, 3], vec![4, 5, 6]]),
            vec![array![1, 4], array![2, 5], array![3, 6]]
        );
        assert_eq!(
            DefaultCollator::collate(vec![vec![1, 2], vec![3, 4], vec![5, 6]]),
            vec![array![1, 3, 5], array![2, 4, 6]]
        );
    }
    #[test]
    fn vec_of_array() {
        assert_eq!(
            DefaultCollator::collate(vec![[1, 2], [3, 4], [5, 6]]),
            vec![array![1, 3, 5], array![2, 4, 6]]
        );
    }

    #[test]
    fn vec_of_hash_map() {
        let map1 = HashMap::from([("A", 0), ("B", 1)]);
        let map2 = HashMap::from([("A", 100), ("B", 100)]);
        let expected_result = HashMap::from([("A", array![0, 100]), ("B", array![1, 100])]);
        assert_eq!(DefaultCollator::collate(vec![map1, map2]), expected_result);

        // Same value type but different key
        let map1 = HashMap::from([(1, 0), (2, 1)]);
        let map2 = HashMap::from([(1, 100), (2, 100)]);
        let expected_result = HashMap::from([(1, array![0, 100]), (2, array![1, 100])]);
        assert_eq!(DefaultCollator::collate(vec![map1, map2]), expected_result);

        let map1 = HashMap::from([("A", 0.0), ("B", 1.0)]);
        let map2 = HashMap::from([("A", 100.0), ("B", 100.0)]);
        let expected_result = HashMap::from([("A", array![0.0, 100.0]), ("B", array![1.0, 100.0])]);
        assert_eq!(DefaultCollator::collate(vec![map1, map2]), expected_result);
    }
    #[test]
    fn ndarray_of_array() {
        assert_eq!(
            DefaultCollator::collate(array![[0., 2.], [1., 3.]]),
            array![[0., 2.], [1., 3.]]
        );
    }

    #[test]
    fn default_collect_specialized() {
        // case that require specialization (string and bytes):
        assert_eq!(
            DefaultCollator::collate(String::from("foo")),
            String::from("foo")
        );
        assert_eq!(
            DefaultCollator::collate(vec![String::from("a"), String::from("b")]),
            vec![String::from("a"), String::from("b")]
        );

        assert_eq!(DefaultCollator::collate(vec!["a", "b"]), vec!["a", "b"]);
        assert_eq!(
            DefaultCollator::collate((String::from("a"), String::from("b"))),
            (String::from("a"), String::from("b"))
        );

        // TODO
        assert_eq!(
            DefaultCollator::collate(vec![
                vec![String::from("a"), String::from("b")],
                vec![String::from("c"), String::from("d")]
            ]),
            vec![
                (String::from('a'), String::from('c')),
                (String::from('b'), String::from('d')),
            ]
        );

        let map1 = HashMap::from([("A", String::from("0")), ("B", String::from("1"))]);
        let map2 = HashMap::from([("A", String::from("100")), ("B", String::from("100"))]);
        let expected_result = HashMap::from([
            ("A", vec![String::from("0"), String::from("100")]),
            ("B", vec![String::from("1"), String::from("100")]),
        ]);
        assert_eq!(DefaultCollator::collate(vec![map1, map2]), expected_result);

        let map1 = HashMap::from([("A", "0"), ("B", "1")]);
        let map2 = HashMap::from([("A", "100"), ("B", "100")]);
        let expected_result = HashMap::from([("A", vec!["0", "100"]), ("B", vec!["1", "100"])]);
        assert_eq!(DefaultCollator::collate(vec![map1, map2]), expected_result);
    }
}
