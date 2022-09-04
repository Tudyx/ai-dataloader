// use super::super::Collate;
// use super::DefaultCollate;
// use ndarray::Array1;

// macro_rules! impl_array_collect {
//     ($($t:ty)*) => {
//         $(
//             impl<const N: usize> Collate<[$t; N]> for DefaultCollate {
//                 type Output = Array1<$t>;
//                 fn collate(batch: Vec<[$t; N]>) -> Self::Output {
//                     Array1::from_vec(batch.to_vec())
//                 }
//             }
//         )*
//     };
// }
// impl_array_collect!(usize u8 u16 u32 u64 u128
//         isize i8 i16 i32 i64 i128
//         f32 f64
//         bool char);

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use ndarray::array;

//     #[test]
//     fn vec_of_array() {
//         assert_eq!(
//             DefaultCollate::collate(vec![[1, 2], [3, 4], [5, 6]]),
//             vec![array![1, 3, 5], array![2, 4, 6]]
//         );
//     }

//     #[test]
//     fn scalar_type() {
//         assert_eq!(DefaultCollate::collate([1, 2, 3]), array![1, 2, 3]);
//         assert_eq!(DefaultCollate::collate([1, -2, 3]), array![1, -2, 3]);
//         assert_eq!(DefaultCollate::collate([1., 2., 3.]), array![1., 2., 3.]);
//         assert_eq!(
//             DefaultCollate::collate([true, false, true]),
//             array![true, false, true]
//         );
//     }
// }
