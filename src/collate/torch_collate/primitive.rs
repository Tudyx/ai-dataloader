use super::super::Collate;
use super::TorchCollate;
use tch::Tensor;

macro_rules! primitive_impl {
    ($($t:ty)*) => {
        $(
            impl Collate<$t> for TorchCollate {
                type Output = Tensor;
                fn collate(&self, batch: Vec<$t>) -> Self::Output {
                    Tensor::from_slice(batch.as_slice())
                }
            }
        )*
    };
}
primitive_impl!(    
    i8 i16 i32 i64 
    f32 f64
    bool);

// char i128 isize usize u16 u32 u64 u128 are not compatible with `tch::Tensor`.

/// `NoOp` for binary, as pytorch `default_collate` function.
impl Collate<u8> for TorchCollate {
    type Output = Vec<u8>;
    fn collate(&self, batch: Vec<u8>) -> Self::Output {
        batch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_type() {
        assert_eq!(
            TorchCollate.collate(vec![0, 1, 2, 3, 4, 5]),
            Tensor::from_slice(&[0, 1, 2, 3, 4, 5])
        );
        assert_eq!(
            TorchCollate.collate(vec![0., 1., 2., 3., 4., 5.]),
            Tensor::from_slice(&[0., 1., 2., 3., 4., 5.])
        );
    }
}
