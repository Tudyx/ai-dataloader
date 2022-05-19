use super::Collect;

use itertools::izip;
use ndarray::{array, Array, Array1, ArrayBase, Dim, Dimension, Ix1, OwnedRepr};
use std::collections::HashMap;

#[derive(Default)]
pub struct DefaultCollector;

macro_rules! impl_vec_collect {
    ($($t:ty)*) => {
        $(
        impl Collect<Vec<$t>> for DefaultCollector {
            type Output = Array<$t, Ix1>;
            fn collect(batch: Vec<$t>) -> Self::Output {
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
            impl<const N: usize> Collect<[$t; N]> for DefaultCollector {
                type Output = Array<$t, Ix1>;
                fn collect(batch: [$t; N]) -> Self::Output {
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
impl Collect<String> for DefaultCollector {
    type Output = String;
    fn collect(batch: String) -> Self::Output {
        batch
    }
}
impl Collect<Vec<String>> for DefaultCollector {
    type Output = Vec<String>;
    fn collect(batch: Vec<String>) -> Vec<String> {
        batch
    }
}

impl Collect<(String, String)> for DefaultCollector {
    type Output = (String, String);
    fn collect(batch: (String, String)) -> Self::Output {
        batch
    }
}
impl Collect<(String, String, String)> for DefaultCollector {
    type Output = (String, String, String);
    fn collect(batch: (String, String, String)) -> (String, String, String) {
        batch
    }
}

impl Collect<Vec<(String, String)>> for DefaultCollector {
    type Output = Vec<(String, String)>;
    fn collect(batch: Vec<(String, String)>) -> Vec<(String, String)> {
        todo!()
    }
}

// implémentation pour n'importe quelle dimension on fait rien ndarray -> ndarray
impl<T, I: Dimension> Collect<Array<T, I>> for DefaultCollector {
    type Output = Array<T, I>;
    fn collect(batch: Array<T, I>) -> Array<T, I> {
        batch
    }
}
impl Collect<(i32, i32, i32)> for DefaultCollector {
    type Output = Array<i32, Ix1>;
    fn collect(batch: (i32, i32, i32)) -> Array<i32, Ix1> {
        array![batch.0, batch.1, batch.2]
    }
}

impl Collect<(i32, i32)> for DefaultCollector {
    type Output = Array<i32, Ix1>;
    fn collect(batch: (i32, i32)) -> Array<i32, Ix1> {
        array![batch.0, batch.1]
    }
}
impl Collect<(f64, f64)> for DefaultCollector {
    type Output = Array<f64, Ix1>;
    fn collect(batch: (f64, f64)) -> Self::Output {
        array![batch.0, batch.1]
    }
}
impl Collect<(f64, f64, f64)> for DefaultCollector {
    type Output = Array<f64, Ix1>;
    fn collect(batch: (f64, f64, f64)) -> Self::Output {
        array![batch.0, batch.1, batch.2]
    }
}
impl Collect<Vec<(f64, f64)>> for DefaultCollector {
    type Output = Vec<Array<f64, Ix1>>;
    fn collect(batch: Vec<(f64, f64)>) -> Self::Output {
        let mut res = Vec::new();
        if batch.len() == 1 {
            res.push(array![batch[0].0.clone()]);
            res.push(array![batch[0].1.clone()]);
        } else if batch.len() == 2 {
            let tuple = (batch[0].0.clone(), batch[1].0.clone());
            res.push(DefaultCollector::collect(tuple));
            let tuple = (batch[0].1.clone(), batch[1].1.clone());
            res.push(DefaultCollector::collect(tuple));
        } else if batch.len() == 3 {
            let tuple = (batch[0].0.clone(), batch[1].0.clone(), batch[2].0.clone());
            res.push(DefaultCollector::collect(tuple));
            let tuple = (batch[0].1.clone(), batch[1].1.clone(), batch[2].1.clone());
            res.push(DefaultCollector::collect(tuple));
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
impl Collect<Vec<Vec<i32>>> for DefaultCollector {
    type Output = Vec<Array<i32, Ix1>>;
    fn collect(batch: Vec<Vec<i32>>) -> Self::Output {
        let elem_size = batch.iter().next().unwrap().len();
        if !batch.iter().all(|vec| vec.len() == elem_size) {
            panic!("each element in list of batch should be of equal size");
        }
        let mut res = Vec::new();

        // I don't find a way to unpack a vec of vec.
        // Maybe i can turn this into a macro
        if batch.len() == 1 {
            res.push(DefaultCollector::collect(batch[0].clone()));
        } else if batch.len() == 2 {
            for samples in izip!(batch[0].clone(), batch[1].clone()) {
                res.push(DefaultCollector::collect(samples))
            }
        } else if batch.len() == 3 {
            for samples in izip!(batch[0].clone(), batch[1].clone(), batch[2].clone()) {
                res.push(DefaultCollector::collect(samples));
            }
        }
        res
    }
}

// comme pour les string sont transformé en tuple pour les strings il peut y avoir plusieurs type de retour..
// Cela retourne un tuple de la taille du vec.
impl Collect<Vec<Vec<String>>> for DefaultCollector {
    type Output = Vec<(String, String)>;
    fn collect(batch: Vec<Vec<String>>) -> Self::Output {
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
                res.push(DefaultCollector::collect(samples));
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
impl<const N: usize> Collect<Vec<[i32; N]>> for DefaultCollector {
    type Output = Vec<Array<i32, Ix1>>;
    fn collect(batch: Vec<[i32; N]>) -> Vec<Array<i32, Ix1>> {
        let mut it = batch.iter();
        let elem_size = it.next().unwrap().len();
        if !batch.iter().all(|vec| vec.len() == elem_size) {
            panic!("each element in list of batch should be of equal size");
        }
        let mut res = Vec::new();
        if batch.len() == 1 {
            res.push(DefaultCollector::collect(batch[0].clone()));
        } else if batch.len() == 2 {
            for samples in izip!(batch[0].clone(), batch[1].clone()) {
                res.push(DefaultCollector::collect(samples))
            }
        } else if batch.len() == 3 {
            for samples in izip!(batch[0].clone(), batch[1].clone(), batch[2].clone()) {
                res.push(DefaultCollector::collect(samples));
            }
        }
        res
    }
}

impl<'a> Collect<Vec<HashMap<&'a str, i32>>> for DefaultCollector {
    type Output = HashMap<&'a str, Array1<i32>>;
    fn collect(batch: Vec<HashMap<&'a str, i32>>) -> Self::Output {
        let mut res = HashMap::new();
        for key in batch[0].keys() {
            let mut vec = Vec::new();
            for d in &batch {
                vec.push(d[key]);
            }
            res.insert(*key, DefaultCollector::collect(vec));
        }
        res
    }
}

impl
    Collect<
        Vec<(
            ArrayBase<OwnedRepr<f64>, Dim<[usize; 3]>>,
            ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>,
        )>,
    > for DefaultCollector
{
    type Output = ArrayBase<
        OwnedRepr<(
            ArrayBase<OwnedRepr<f64>, Dim<[usize; 3]>>,
            ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>,
        )>,
        Dim<[usize; 1]>,
    >;

    fn collect(
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
            DefaultCollector::collect(vec![0, 1, 2, 3, 4, 5]),
            array![0, 1, 2, 3, 4, 5]
        );
        assert_eq!(
            DefaultCollector::collect(vec![0., 1., 2., 3., 4., 5.]),
            array![0., 1., 2., 3., 4., 5.]
        );
        assert_eq!(DefaultCollector::collect([1, 2, 3]), array![1, 2, 3]);
        assert_eq!(DefaultCollector::collect([1, -2, 3]), array![1, -2, 3]);
        assert_eq!(DefaultCollector::collect([1., 2., 3.]), array![1., 2., 3.]);
        assert_eq!(
            DefaultCollector::collect([true, false, true]),
            array![true, false, true]
        );
        // should be a no op
        assert_eq!(
            DefaultCollector::collect(array![1, 2, 3, 4]),
            array![1, 2, 3, 4]
        );
    }

    #[test]
    fn default_collect_map() {
        let map1 = HashMap::from([("A", 0), ("B", 1)]);
        let map2 = HashMap::from([("A", 100), ("B", 100)]);
        let expected_result = HashMap::from([("A", array![0, 100]), ("B", array![1, 100])]);
        assert_eq!(DefaultCollector::collect(vec![map1, map2]), expected_result);
    }

    // [tensor([1, 3]), tensor([2, 4])]
    #[test]
    fn default_collate_tuple() {
        assert_eq!(DefaultCollector::collect((1.0, 2.0)), array![1.0, 2.0]);

        assert_eq!(
            DefaultCollector::collect(vec![(1.0, 2.0)]),
            vec![array![1.0], array![2.0]]
        );
        assert_eq!(
            DefaultCollector::collect(vec![(1.0, 2.0), (3.0, 4.0)]),
            vec![array![1.0, 3.0], array![2.0, 4.0]]
        );
        assert_eq!(
            DefaultCollector::collect(vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]),
            vec![array![1.0, 3.0, 5.0], array![2.0, 4.0, 6.0]]
        );
    }

    #[test]
    fn default_collect_specialized() {
        // case that require specialization (string and bytes):
        assert_eq!(
            DefaultCollector::collect(String::from("foo")),
            String::from("foo")
        );
        assert_eq!(
            DefaultCollector::collect(vec![String::from("a"), String::from("b")]),
            vec![String::from("a"), String::from("b")]
        );
        assert_eq!(
            DefaultCollector::collect((String::from("a"), String::from("b"))),
            (String::from("a"), String::from("b"))
        );

        // TODO
        assert_eq!(
            DefaultCollector::collect(vec![
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
            DefaultCollector::collect(array![[0., 2.], [1., 3.]]),
            array![[0., 2.], [1., 3.]]
        );
        assert_eq!(
            DefaultCollector::collect(vec![vec![1, 2], vec![3, 4]]),
            vec![array![1, 3], array![2, 4]]
        );

        assert_eq!(
            DefaultCollector::collect(vec![vec![1, 2, 3], vec![4, 5, 6]]),
            vec![array![1, 4], array![2, 5], array![3, 6]]
        );
        assert_eq!(
            DefaultCollector::collect(vec![vec![1, 2], vec![3, 4], vec![5, 6]]),
            vec![array![1, 3, 5], array![2, 4, 6]]
        );
        assert_eq!(
            DefaultCollector::collect(vec![[1, 2], [3, 4], [5, 6]]),
            vec![array![1, 3, 5], array![2, 4, 6]]
        );

        // assert_eq!(DefaultCollector::collect(HashMap::from([("A", 0), ("B",1)])))
        // assert_eq!(
        //     DefaultCollector::collect(vec![[1., 2.], [3., 4.], [5., 6.]]),
        //     vec![array![1., 3., 5.], array![2., 4., 6.]]
        // );
    }
}
