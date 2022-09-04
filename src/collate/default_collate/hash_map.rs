use super::super::Collate;
use super::DefaultCollate;
use std::collections::HashMap;

macro_rules! impl_default_collate_vec_hash_map {
    ($($t:ty)*) => {
        $(
            /// HasMap implementation for default collate. We can only be generic over keys, see comment below.
            impl<K> Collate<Vec<HashMap<K, $t>>> for DefaultCollate
            where
                K: std::cmp::Eq + std::hash::Hash + Clone,
            {
                type Output = HashMap<K, <DefaultCollate as Collate<Vec<$t>>>::Output>;

                fn collate(batch: Vec<HashMap<K, $t>>) -> Self::Output {
                    let mut res = HashMap::with_capacity(batch[0].keys().len());
                    for key in batch[0].keys() {
                        let vec: Vec<_> = batch.iter().map(|hash_map| hash_map[key].clone()).collect();
                        res.insert(key.clone(), DefaultCollate::collate(vec));
                    }
                    res
                }
            }
        )*
    };
}

impl_default_collate_vec_hash_map!(usize u8 u16 u32 u64 u128
isize i8 i16 i32 i64 i128
f32 f64
bool char
String
);

/// String slice require a specific implementation because of lifetime
impl<'a, K> Collate<Vec<HashMap<K, &'a str>>> for DefaultCollate
where
    K: std::cmp::Eq + std::hash::Hash + Clone,
{
    type Output = HashMap<K, <DefaultCollate as Collate<Vec<&'a str>>>::Output>;

    fn collate(batch: Vec<HashMap<K, &'a str>>) -> Self::Output {
        let mut res = HashMap::with_capacity(batch[0].keys().len());
        for key in batch[0].keys() {
            let vec: Vec<_> = batch.iter().map(|hash_map| hash_map[key].clone()).collect();
            res.insert(key.clone(), DefaultCollate::collate(vec));
        }
        res
    }
}

/// A generic implementation for any type of key and value is not possible.
/// Indeed if V is a HashMap we should also be able to
/// collate it but were are defining here how to collate an HasMap
/// ```ignore
/// use std::collections::HashMap;
/// use dataloader_rs::collate::Collate;
/// use dataloader_rs::collate::default_collate::DefaultCollate;
///
/// impl<K, V> Collate<Vec<HashMap<K, V>>> for DefaultCollate
/// where
///     DefaultCollate: Collate<Vec<V>>,
/// {
///     type Output = HashMap<K, <DefaultCollate as Collate<Vec<V>>>::Output>;
///
///     fn collate(batch: Vec<HashMap<K, V>>) -> Self::Output {
///         let mut res = HashMap::new();
///         for key in batch[0].keys() {
///             let mut vec = Vec::new();
///             for d in &batch {
///                 vec.push(d[key]);
///             }
///             res.insert(*key, DefaultCollate::collate(vec));
///         }
///         res
///     }
/// }
/// ```
///

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn vec_of_hash_map() {
        let map1 = HashMap::from([("A", 0), ("B", 1)]);
        let map2 = HashMap::from([("A", 100), ("B", 100)]);
        let expected_result = HashMap::from([("A", array![0, 100]), ("B", array![1, 100])]);
        assert_eq!(DefaultCollate::collate(vec![map1, map2]), expected_result);

        // Same value type but different key
        let map1 = HashMap::from([(1, 0), (2, 1)]);
        let map2 = HashMap::from([(1, 100), (2, 100)]);
        let expected_result = HashMap::from([(1, array![0, 100]), (2, array![1, 100])]);
        assert_eq!(DefaultCollate::collate(vec![map1, map2]), expected_result);

        let map1 = HashMap::from([("A", 0.0), ("B", 1.0)]);
        let map2 = HashMap::from([("A", 100.0), ("B", 100.0)]);
        let expected_result = HashMap::from([("A", array![0.0, 100.0]), ("B", array![1.0, 100.0])]);
        assert_eq!(DefaultCollate::collate(vec![map1, map2]), expected_result);
    }

    #[test]
    fn specialized() {
        let map1 = HashMap::from([("A", String::from("0")), ("B", String::from("1"))]);
        let map2 = HashMap::from([("A", String::from("100")), ("B", String::from("100"))]);
        let expected_result = HashMap::from([
            ("A", vec![String::from("0"), String::from("100")]),
            ("B", vec![String::from("1"), String::from("100")]),
        ]);
        assert_eq!(DefaultCollate::collate(vec![map1, map2]), expected_result);

        let map1 = HashMap::from([("A", "0"), ("B", "1")]);
        let map2 = HashMap::from([("A", "100"), ("B", "100")]);
        let expected_result = HashMap::from([("A", vec!["0", "100"]), ("B", vec!["1", "100"])]);
        assert_eq!(DefaultCollate::collate(vec![map1, map2]), expected_result);
    }
}
