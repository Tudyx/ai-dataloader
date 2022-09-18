use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet, LinkedList, VecDeque};

/// Basic trait for anything that could have a length.
/// Even if a lot of struct have a `len()` method in the standard library,
/// to my knowledge this function is not included into any standard trait.
pub trait Len {
    /// Returns the number of elements in the collection, also referred to
    /// as its length.
    fn len(&self) -> usize;
    /// Return `true` if the collection has no element.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// TODO; generic macro for all std::collection
// Add bound to generic?
impl<T> Len for Vec<T> {
    fn len(&self) -> usize {
        self.len()
    }
}

impl<T> Len for VecDeque<T> {
    fn len(&self) -> usize {
        self.len()
    }
}

impl<K, V, H> Len for HashMap<K, V, H> {
    fn len(&self) -> usize {
        self.len()
    }
}

impl<K, V> Len for BTreeMap<K, V> {
    fn len(&self) -> usize {
        self.len()
    }
}

impl<T> Len for LinkedList<T> {
    fn len(&self) -> usize {
        self.len()
    }
}

impl<T> Len for BTreeSet<T> {
    fn len(&self) -> usize {
        self.len()
    }
}

impl<T, H> Len for HashSet<T, H> {
    fn len(&self) -> usize {
        self.len()
    }
}

impl<T> Len for BinaryHeap<T> {
    fn len(&self) -> usize {
        self.len()
    }
}
