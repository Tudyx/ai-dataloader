// use super::super::Collate;
// use super::DefaultCollate;
// use itertools::izip;
// use ndarray::Array1;

// impl<const N: usize> Collate<[i32; N]> for DefaultCollate {
//     type Output = Vec<Array1<i32>>;
//     fn collate(batch: Vec<[i32; N]>) -> Vec<Array1<i32>> {
//         let mut it = batch.iter();
//         let elem_size = it.next().unwrap().len();
//         if !batch.iter().all(|vec| vec.len() == elem_size) {
//             panic!("each element in list of batch should be of equal size");
//         }
//         let mut res = Vec::new();
//         if batch.len() == 1 {
//             res.push(DefaultCollate::collate(batch[0]));
//         } else if batch.len() == 2 {
//             for samples in izip!(batch[0], batch[1]) {
//                 res.push(DefaultCollate::collate(samples))
//             }
//         } else if batch.len() == 3 {
//             for samples in izip!(batch[0], batch[1], batch[2]) {
//                 res.push(DefaultCollate::collate(samples));
//             }
//         }
//         res
//     }
// }

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

// }
