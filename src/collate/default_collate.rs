use super::Collate;

use itertools::izip;
use ndarray::{array, Array, Array1, ArrayBase, Dim, Dimension, Ix1, OwnedRepr};
use std::collections::HashMap;

#[derive(Default)]
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
    fn collate(batch: Vec<String>) -> Vec<String> {
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

impl Collate<Vec<(String, String)>> for DefaultCollator {
    type Output = Vec<(String, String)>;
    fn collate(batch: Vec<(String, String)>) -> Vec<(String, String)> {
        todo!()
    }
}

// implémentation pour n'importe quelle dimension on fait rien ndarray -> ndarray
impl<T, I: Dimension> Collate<Array<T, I>> for DefaultCollator {
    type Output = Array<T, I>;
    fn collate(batch: Array<T, I>) -> Array<T, I> {
        batch
    }
}
impl Collate<(i32, i32, i32)> for DefaultCollator {
    type Output = Array<i32, Ix1>;
    fn collate(batch: (i32, i32, i32)) -> Array<i32, Ix1> {
        array![batch.0, batch.1, batch.2]
    }
}

impl Collate<(i32, i32)> for DefaultCollator {
    type Output = Array<i32, Ix1>;
    fn collate(batch: (i32, i32)) -> Array<i32, Ix1> {
        array![batch.0, batch.1]
    }
}
impl Collate<(f64, f64)> for DefaultCollator {
    type Output = Array<f64, Ix1>;
    fn collate(batch: (f64, f64)) -> Self::Output {
        array![batch.0, batch.1]
    }
}
impl Collate<(f64, f64, f64)> for DefaultCollator {
    type Output = Array<f64, Ix1>;
    fn collate(batch: (f64, f64, f64)) -> Self::Output {
        array![batch.0, batch.1, batch.2]
    }
}
impl Collate<Vec<(f64, f64)>> for DefaultCollator {
    type Output = Vec<Array<f64, Ix1>>;
    fn collate(batch: Vec<(f64, f64)>) -> Self::Output {
        let mut res = Vec::new();
        if batch.len() == 1 {
            res.push(array![batch[0].0.clone()]);
            res.push(array![batch[0].1.clone()]);
        } else if batch.len() == 2 {
            let tuple = (batch[0].0.clone(), batch[1].0.clone());
            res.push(DefaultCollator::collate(tuple));
            let tuple = (batch[0].1.clone(), batch[1].1.clone());
            res.push(DefaultCollator::collate(tuple));
        } else if batch.len() == 3 {
            let tuple = (batch[0].0.clone(), batch[1].0.clone(), batch[2].0.clone());
            res.push(DefaultCollator::collate(tuple));
            let tuple = (batch[0].1.clone(), batch[1].1.clone(), batch[2].1.clone());
            res.push(DefaultCollator::collate(tuple));
        }
        res
    }
}

// impl Collect<(f64, i32)> for DefaultCollector {
//     type Output = Array<i32, Ix1>;
//     fn collect(batch: (i32, i32)) -> Array<i32, Ix1> {
//         array![batch.0, batch.1]
//     }
// }

// Todo macro for unpacking the vec
// macro_rules! unpack_vec {
//     ($element:expr; $len:expr) => {

//     };
// }
impl Collate<Vec<Vec<i32>>> for DefaultCollator {
    type Output = Vec<Array<i32, Ix1>>;
    fn collate(batch: Vec<Vec<i32>>) -> Self::Output {
        let elem_size = batch.iter().next().unwrap().len();
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
                res.push(DefaultCollator::collate(samples))
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
        let elem_size = batch.iter().next().unwrap().len();
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
            for samples in izip!(batch[0].clone(), batch[1].clone(), batch[2].clone()) {
                // res.push(DefaultCollector::collect(samples));
            }
        }
        res
    }
}
// impl<T> Collect<Vec<Vec<T>>> for DefaultCollector
// where
//     DefaultCollector: Collect<Vec<T>>,
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
            res.push(DefaultCollator::collate(batch[0].clone()));
        } else if batch.len() == 2 {
            for samples in izip!(batch[0].clone(), batch[1].clone()) {
                res.push(DefaultCollator::collate(samples))
            }
        } else if batch.len() == 3 {
            for samples in izip!(batch[0].clone(), batch[1].clone(), batch[2].clone()) {
                res.push(DefaultCollator::collate(samples));
            }
        }
        res
    }
}

impl<'a> Collate<Vec<HashMap<&'a str, i32>>> for DefaultCollator {
    type Output = HashMap<&'a str, Array1<i32>>;
    fn collate(batch: Vec<HashMap<&'a str, i32>>) -> Self::Output {
        let mut res = HashMap::new();
        for key in batch[0].keys() {
            let mut vec = Vec::new();
            for d in &batch {
                vec.push(d[key]);
            }
            res.insert(*key, DefaultCollator::collate(vec));
        }
        res
    }
}

impl
    Collate<
        Vec<(
            ArrayBase<OwnedRepr<f64>, Dim<[usize; 3]>>,
            ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>,
        )>,
    > for DefaultCollator
{
    type Output = ArrayBase<
        OwnedRepr<(
            ArrayBase<OwnedRepr<f64>, Dim<[usize; 3]>>,
            ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>,
        )>,
        Dim<[usize; 1]>,
    >;

    fn collate(
        batch: Vec<(
            ArrayBase<OwnedRepr<f64>, Dim<[usize; 3]>>,
            ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>,
        )>,
    ) -> Self::Output {
        // let res = vec![];
        // for (data, label) in batch {}
        Array::from_vec(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_collate_primitive_type() {
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
        // should be a no op
        assert_eq!(
            DefaultCollator::collate(array![1, 2, 3, 4]),
            array![1, 2, 3, 4]
        );
    }

    #[test]
    fn default_collect_map() {
        let map1 = HashMap::from([("A", 0), ("B", 1)]);
        let map2 = HashMap::from([("A", 100), ("B", 100)]);
        let expected_result = HashMap::from([("A", array![0, 100]), ("B", array![1, 100])]);
        assert_eq!(DefaultCollator::collate(vec![map1, map2]), expected_result);
    }

    // [tensor([1, 3]), tensor([2, 4])]
    #[test]
    fn default_collate_tuple() {
        assert_eq!(DefaultCollator::collate((1.0, 2.0)), array![1.0, 2.0]);

        assert_eq!(
            DefaultCollator::collate(vec![(1.0, 2.0)]),
            vec![array![1.0], array![2.0]]
        );
        assert_eq!(
            DefaultCollator::collate(vec![(1.0, 2.0), (3.0, 4.0)]),
            vec![array![1.0, 3.0], array![2.0, 4.0]]
        );
        assert_eq!(
            DefaultCollator::collate(vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]),
            vec![array![1.0, 3.0, 5.0], array![2.0, 4.0, 6.0]]
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

        // TODO
        // assert_eq!(
        //     DefaultCollector::collect(vec![
        //         (String::from("a"), String::from("b")),
        //         (String::from("c"), String::from("d"))
        //     ]),
        //     vec![
        //         (String::from('a'), String::from('c')),
        //         (String::from('b'), String::from('d'))
        //     ]
        // );
    }
    //TODO ?yq
    // assert_eq!(
    //     DefaultCollector::collect(vec![0..2, 0..2]), vec![!array![0,0], array![1,1]]
    // );
    #[test]
    fn default_collate_multi_dimensional() {
        assert_eq!(
            DefaultCollator::collate(array![[0., 2.], [1., 3.]]),
            array![[0., 2.], [1., 3.]]
        );
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
        assert_eq!(
            DefaultCollator::collate(vec![[1, 2], [3, 4], [5, 6]]),
            vec![array![1, 3, 5], array![2, 4, 6]]
        );

        // assert_eq!(DefaultCollector::collect(HashMap::from([("A", 0), ("B",1)])))
        // assert_eq!(
        //     DefaultCollector::collect(vec![[1., 2.], [3., 4.], [5., 6.]]),
        //     vec![array![1., 3., 5.], array![2., 4., 6.]]
        // );
    }
}
