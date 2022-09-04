use super::super::Collate;
use super::DefaultCollate;

use ndarray::{Array, Ix1};

macro_rules! impl_vec_collect {
    ($($t:ty)*) => {
        $(
            impl Collate<Vec<$t>> for DefaultCollate {
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn scalar_type() {
        assert_eq!(
            DefaultCollate::collate(vec![0, 1, 2, 3, 4, 5]),
            array![0, 1, 2, 3, 4, 5]
        );
        assert_eq!(
            DefaultCollate::collate(vec![0., 1., 2., 3., 4., 5.]),
            array![0., 1., 2., 3., 4., 5.]
        );
    }
}
