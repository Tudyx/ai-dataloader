use super::super::Collate;
use super::TorchCollate;

impl<T, const N: usize> Collate<[T; N]> for TorchCollate
where
    Self: Collate<T>,
    T: Clone,
{
    type Output = Vec<<Self as Collate<T>>::Output>;
    fn collate(batch: Vec<[T; N]>) -> Self::Output {
        let mut collated = Vec::with_capacity(batch.len());
        for i in 0..batch[0].len() {
            let vec: Vec<_> = batch.iter().map(|sample| sample[i].clone()).collect();
            collated.push(Self::collate(vec));
        }
        collated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Tensor;

    #[test]
    fn vec_of_array() {
        assert_eq!(
            TorchCollate::collate(vec![[1, 2], [3, 4], [5, 6]]),
            vec![Tensor::of_slice(&[1, 3, 5]), Tensor::of_slice(&[2, 4, 6])]
        );
    }
}
