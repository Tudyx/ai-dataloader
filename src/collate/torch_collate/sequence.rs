/// Implementation for Sequence.
///
/// Currently `BinaryHeap`, `BTreeSet`, `HashSet` and `LinkedList` are not supported because the current implementation
/// require indexing for doing the transpose.
///
use super::super::Collate;
use super::TorchCollate;
use std::collections::VecDeque;

impl<T> Collate<Vec<T>> for TorchCollate
where
    Self: Collate<T>,
    T: Clone,
{
    type Output = Vec<<Self as Collate<T>>::Output>;
    fn collate(&self, batch: Vec<Vec<T>>) -> Self::Output {
        let elem_size = batch
            .get(0)
            .expect("Batch should contain at least one element")
            .len();

        assert!(
            batch.iter().all(|vec| vec.len() == elem_size),
            "Each Vec in the batch should have equal size"
        );

        let mut collated = Vec::with_capacity(batch.len());

        for i in 0..batch[0].len() {
            let vec: Vec<_> = batch.iter().map(|sample| sample[i].clone()).collect();
            collated.push(self.collate(vec));
        }
        collated
    }
}

impl<T> Collate<VecDeque<T>> for TorchCollate
where
    Self: Collate<T>,
    T: Clone,
{
    type Output = Vec<<Self as Collate<T>>::Output>;
    fn collate(&self, batch: Vec<VecDeque<T>>) -> Self::Output {
        let elem_size = batch
            .get(0)
            .expect("Batch should contain at least one element")
            .len();

        assert!(
            batch.iter().all(|vec| vec.len() == elem_size),
            "Each Vec in the batch should have equal size"
        );

        let mut collated = Vec::with_capacity(batch.len());

        for i in 0..batch[0].len() {
            let vec: Vec<_> = batch.iter().map(|sample| sample[i].clone()).collect();
            collated.push(self.collate(vec));
        }
        collated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use tch::Tensor;

    #[test]
    fn vec_of_vec() {
        assert_eq!(
            TorchCollate::default().collate(vec![vec![1]]),
            vec![Tensor::of_slice(&[1])]
        );
        assert_eq!(
            TorchCollate::default().collate(vec![vec![1, 2], vec![3, 4]]),
            vec![Tensor::of_slice(&[1, 3]), Tensor::of_slice(&[2, 4])]
        );
        // different type
        assert_eq!(
            TorchCollate::default().collate(vec![vec![true, false], vec![true, false]]),
            vec![
                Tensor::of_slice(&[true, true]),
                Tensor::of_slice(&[false, false])
            ]
        );

        assert_eq!(
            TorchCollate::default().collate(vec![vec![1, 2, 3], vec![4, 5, 6]]),
            vec![
                Tensor::of_slice(&[1, 4]),
                Tensor::of_slice(&[2, 5]),
                Tensor::of_slice(&[3, 6])
            ]
        );
        // batch_size 3
        assert_eq!(
            TorchCollate::default().collate(vec![vec![1, 2], vec![3, 4], vec![5, 6]]),
            vec![Tensor::of_slice(&[1, 3, 5]), Tensor::of_slice(&[2, 4, 6])]
        );
        // batch_size 10
        assert_eq!(
            TorchCollate::default().collate(vec![
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
                Tensor::of_slice(&[1, 3, 5, 7, 9, 11, 13, 15, 17, 19]),
                Tensor::of_slice(&[2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
            ]
        );
    }

    #[test]
    fn specialized() {
        assert_eq!(
            TorchCollate::default().collate(vec![
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
