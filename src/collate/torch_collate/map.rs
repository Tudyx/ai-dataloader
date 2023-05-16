use super::super::Collate;
use super::TorchCollate;
use std::{
    cmp::Eq,
    collections::{BTreeMap, HashMap},
    hash::{BuildHasher, Hash},
};

impl<K, V, H> Collate<HashMap<K, V, H>> for TorchCollate
where
    K: Eq + Hash + Clone,
    V: Clone,
    Self: Collate<V>,
    H: BuildHasher,
{
    type Output = HashMap<K, <Self as Collate<V>>::Output>;
    fn collate(&self, batch: Vec<HashMap<K, V, H>>) -> Self::Output {
        let mut collated = HashMap::with_capacity(batch[0].keys().len());
        for key in batch[0].keys() {
            let vec: Vec<_> = batch.iter().map(|hash_map| hash_map[key].clone()).collect();
            collated.insert(key.clone(), self.collate(vec));
        }
        collated
    }
}
impl<K, V> Collate<BTreeMap<K, V>> for TorchCollate
where
    K: Ord + Clone,
    V: Clone,
    Self: Collate<V>,
{
    type Output = BTreeMap<K, <Self as Collate<V>>::Output>;
    fn collate(&self, batch: Vec<BTreeMap<K, V>>) -> Self::Output {
        let mut collated = BTreeMap::new();
        for key in batch[0].keys() {
            let vec: Vec<_> = batch.iter().map(|hash_map| hash_map[key].clone()).collect();
            collated.insert(key.clone(), self.collate(vec));
        }
        collated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Tensor;

    #[test]
    fn vec_of_hash_map() {
        let map1 = HashMap::from([("A", 0), ("B", 1)]);
        let map2 = HashMap::from([("A", 100), ("B", 100)]);
        let expected_result = HashMap::from([
            ("A", Tensor::of_slice(&[0, 100])),
            ("B", Tensor::of_slice(&[1, 100])),
        ]);
        assert_eq!(
            TorchCollate::default().collate(vec![map1, map2]),
            expected_result
        );

        // Same value type but different key
        let map1 = HashMap::from([(1, 0), (2, 1)]);
        let map2 = HashMap::from([(1, 100), (2, 100)]);
        let expected_result = HashMap::from([
            (1, Tensor::of_slice(&[0, 100])),
            (2, Tensor::of_slice(&[1, 100])),
        ]);
        assert_eq!(
            TorchCollate::default().collate(vec![map1, map2]),
            expected_result
        );

        let map1 = HashMap::from([("A", 0.0), ("B", 1.0)]);
        let map2 = HashMap::from([("A", 100.0), ("B", 100.0)]);
        let expected_result = HashMap::from([
            ("A", Tensor::of_slice(&[0.0, 100.0])),
            ("B", Tensor::of_slice(&[1.0, 100.0])),
        ]);
        assert_eq!(
            TorchCollate::default().collate(vec![map1, map2]),
            expected_result
        );
    }

    #[test]
    fn specialized() {
        let map1 = HashMap::from([("A", String::from("0")), ("B", String::from("1"))]);
        let map2 = HashMap::from([("A", String::from("100")), ("B", String::from("100"))]);
        let expected_result = HashMap::from([
            ("A", vec![String::from("0"), String::from("100")]),
            ("B", vec![String::from("1"), String::from("100")]),
        ]);
        assert_eq!(
            TorchCollate::default().collate(vec![map1, map2]),
            expected_result
        );

        let map1 = HashMap::from([("A", "0"), ("B", "1")]);
        let map2 = HashMap::from([("A", "100"), ("B", "100")]);
        let expected_result = HashMap::from([("A", vec!["0", "100"]), ("B", vec!["1", "100"])]);
        assert_eq!(
            TorchCollate::default().collate(vec![map1, map2]),
            expected_result
        );
    }
}
