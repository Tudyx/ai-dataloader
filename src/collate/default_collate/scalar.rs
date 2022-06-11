use super::super::Collate;
use super::DefaultCollator;

use ndarray::{Array, Ix1};

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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn scalar_type() {
        assert_eq!(
            DefaultCollator::collate(vec![0, 1, 2, 3, 4, 5]),
            array![0, 1, 2, 3, 4, 5]
        );
        assert_eq!(
            DefaultCollator::collate(vec![0., 1., 2., 3., 4., 5.]),
            array![0., 1., 2., 3., 4., 5.]
        );
    }

    #[test]
    fn specialized() {
        assert_eq!(
            DefaultCollator::collate(vec![String::from("a"), String::from("b")]),
            vec![String::from("a"), String::from("b")]
        );

        assert_eq!(DefaultCollator::collate(vec!["a", "b"]), vec!["a", "b"]);
    }
}
