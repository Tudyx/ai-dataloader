/// Implementation for when the vec of batch conatins a vec
use super::super::Collate;
use super::DefaultCollate;
use itertools::izip;

macro_rules! impl_vec_vec {
    ($($t:ty)*) => {
        $(
            /// Implementation for vector of vector
            // TODO: ret
            // # FIXME : the current implmentation make a lot of copy
            // I tried with Itertools::izip! without success
            impl Collate<Vec<$t>> for DefaultCollate {
                type Output = Vec<<DefaultCollate as Collate<$t>>::Output>;
                fn collate(batch: Vec<Vec<$t>>) -> Self::Output {
                    let elem_size = batch
                        .get(0)
                        .expect("Batch should contain at least one element")
                        .len();
                    if !batch.iter().all(|vec| vec.len() == elem_size) {
                        panic!("Each Vec in the batch should have equal size");
                    }
                    let mut res = Vec::with_capacity(batch.len());

                    for i in 0..batch[0].len() {
                        let vec: Vec<_> = batch.iter().map(|sample| sample[i]).collect();
                        res.push(DefaultCollate::collate(vec));
                    }
                    res
                }
            }
        )*
    };
}
impl_vec_vec!(usize u8 u16 u32 u64 u128
isize i8 i16 i32 i64 i128
f32 f64
bool char
);

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn vec_of_vec() {
        assert_eq!(DefaultCollate::collate(vec![vec![1]]), vec![array![1]]);
        assert_eq!(
            DefaultCollate::collate(vec![vec![1, 2], vec![3, 4]]),
            vec![array![1, 3], array![2, 4]]
        );
        // different type
        assert_eq!(
            DefaultCollate::collate(vec![vec![true, false], vec![true, false]]),
            vec![array![true, true], array![false, false]]
        );

        assert_eq!(
            DefaultCollate::collate(vec![vec![1, 2, 3], vec![4, 5, 6]]),
            vec![array![1, 4], array![2, 5], array![3, 6]]
        );
        // batch_size 3
        assert_eq!(
            DefaultCollate::collate(vec![vec![1, 2], vec![3, 4], vec![5, 6]]),
            vec![array![1, 3, 5], array![2, 4, 6]]
        );
        // batch_size 10
        assert_eq!(
            DefaultCollate::collate(vec![
                vec![1, 2],
                vec![3, 4],
                vec![5, 6],
                vec![7, 8],
                vec![9, 10],
                vec![11, 12],
                vec![13, 14],
                vec![15, 16],
                vec![17, 18],
                vec![19, 20]
            ]),
            vec![
                array![1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
                array![2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
            ]
        );
    }

    // #[test]
    // fn specialized() {
    //     assert_eq!(
    //         DefaultCollate::collate(vec![
    //             vec![String::from("a"), String::from("b")],
    //             vec![String::from("c"), String::from("d")]
    //         ]),
    //         vec![
    //             (String::from('a'), String::from('c')),
    //             (String::from('b'), String::from('d')),
    //         ]
    //     );
    // }
}
