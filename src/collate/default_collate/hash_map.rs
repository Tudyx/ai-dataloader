use super::super::Collate;
use super::DefaultCollate;
use std::collections::HashMap;

impl<K, T> Collate<HashMap<K, T>> for DefaultCollate
where
    K: std::cmp::Eq + std::hash::Hash + Clone,
    T: Clone,
    DefaultCollate: Collate<T>,
{
    type Output = HashMap<K, <DefaultCollate as Collate<T>>::Output>;
    fn collate(batch: Vec<HashMap<K, T>>) -> Self::Output {
        let mut res = HashMap::with_capacity(batch[0].keys().len());
        for key in batch[0].keys() {
            let vec: Vec<_> = batch.iter().map(|hash_map| hash_map[key].clone()).collect();
            res.insert(key.clone(), DefaultCollate::collate(vec));
        }
        res
    }
}

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
