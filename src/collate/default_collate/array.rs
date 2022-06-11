use super::super::Collate;
use super::DefaultCollator;
use itertools::izip;
use ndarray::Array1;

impl<const N: usize> Collate<Vec<[i32; N]>> for DefaultCollator {
    type Output = Vec<Array1<i32>>;
    fn collate(batch: Vec<[i32; N]>) -> Vec<Array1<i32>> {
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

macro_rules! impl_array_collect {
    ($($t:ty)*) => {
        $(
            impl<const N: usize> Collate<[$t; N]> for DefaultCollator {
                type Output = Array1<$t>;
                fn collate(batch: [$t; N]) -> Self::Output {
                    Array1::from_vec(batch.to_vec())
                }
            }
        )*
    };
}
impl_array_collect!(usize u8 u16 u32 u64 u128
        isize i8 i16 i32 i64 i128
        f32 f64
        bool char);

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn vec_of_array() {
        assert_eq!(
            DefaultCollator::collate(vec![[1, 2], [3, 4], [5, 6]]),
            vec![array![1, 3, 5], array![2, 4, 6]]
        );
    }

    fn scalar_type() {
        assert_eq!(DefaultCollator::collate([1, 2, 3]), array![1, 2, 3]);
        assert_eq!(DefaultCollator::collate([1, -2, 3]), array![1, -2, 3]);
        assert_eq!(DefaultCollator::collate([1., 2., 3.]), array![1., 2., 3.]);
        assert_eq!(
            DefaultCollator::collate([true, false, true]),
            array![true, false, true]
        );
    }
}
