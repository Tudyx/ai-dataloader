use super::super::Collate;
use super::DefaultCollate;
use ndarray::{Array, Ix1};
use std::num::*;

macro_rules! primitive_impl {
    ($($t:ty)*) => {
        $(
            impl Collate<$t> for DefaultCollate {
                type Output = Array<$t, Ix1>;
                fn collate(batch: Vec<$t>) -> Self::Output {
                    Array::from_vec(batch)
                }
            }
        )*
    };
}
primitive_impl!(
    NonZeroUsize NonZeroU8 NonZeroU16 NonZeroU32 NonZeroU64 NonZeroU128
    NonZeroIsize NonZeroI8 NonZeroI16 NonZeroI32 NonZeroI64 NonZeroI128
);
