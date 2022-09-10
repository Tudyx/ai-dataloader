use super::super::Collate;
use super::DefaultCollate;

use ndarray::{Array, Array1};

macro_rules! primitive_impl {
    ($($t:ty)*) => {
        $(
            impl Collate<$t> for DefaultCollate {
                type Output = Array1<$t>;
                fn collate(batch: Vec<$t>) -> Self::Output {
                    Array::from_vec(batch)
                }
            }
        )*
    };
}
primitive_impl!(usize u16 u32 u64 u128
    isize i8 i16 i32 i64 i128
    f32 f64
    bool char);

/// NoOp for binairy, as pytorch `default_collate` function.
impl Collate<u8> for DefaultCollate {
    type Output = Vec<u8>;
    fn collate(batch: Vec<u8>) -> Self::Output {
        batch
    }
}

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
