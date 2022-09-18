use super::super::Collate;
use super::DefaultCollate;
// use itertools::izip;
// use ndarray::Array1;

impl<T, const N: usize> Collate<[T; N]> for DefaultCollate
where
    DefaultCollate: Collate<T>,
    T: Clone,
{
    type Output = Vec<<DefaultCollate as Collate<T>>::Output>;
    fn collate(batch: Vec<[T; N]>) -> Self::Output {
        let mut collated = Vec::with_capacity(batch.len());
        for i in 0..batch[0].len() {
            let vec: Vec<_> = batch.iter().map(|sample| sample[i].clone()).collect();
            collated.push(DefaultCollate::collate(vec));
        }
        collated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn vec_of_array() {
        assert_eq!(
            DefaultCollate::collate(vec![[1, 2], [3, 4], [5, 6]]),
            vec![array![1, 3, 5], array![2, 4, 6]]
        );
    }
}
