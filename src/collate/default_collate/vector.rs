/// Implementation for when the vec of batch conatins a vec
use super::super::Collate;
use super::DefaultCollate;

impl<T> Collate<Vec<T>> for DefaultCollate
where
    DefaultCollate: Collate<T>,
    T: Clone,
{
    type Output = Vec<<DefaultCollate as Collate<T>>::Output>;
    fn collate(batch: Vec<Vec<T>>) -> Self::Output {
        let elem_size = batch
            .get(0)
            .expect("Batch should contain at least one element")
            .len();
        if !batch.iter().all(|vec| vec.len() == elem_size) {
            panic!("Each Vec in the batch should have equal size");
        }

        let mut res = Vec::with_capacity(batch.len());

        for i in 0..batch[0].len() {
            let vec: Vec<_> = batch.iter().map(|sample| sample[i].clone()).collect();
            res.push(DefaultCollate::collate(vec));
        }
        res
    }
}

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

    #[test]
    fn specialized() {
        assert_eq!(
            DefaultCollate::collate(vec![
                vec![String::from("a"), String::from("b")],
                vec![String::from("c"), String::from("d")]
            ]),
            vec![
                vec![String::from('a'), String::from('c')],
                vec![String::from('b'), String::from('d')],
            ]
        );
    }
}
