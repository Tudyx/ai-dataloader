/// Basic collator that mimic the default collate function from PyTorch
/// As they are no such lib whith the same functionnality as PyTorch tensor in rust,
/// data is collated inside ndarray. Ndarray is the rust equivalent of numpy.ndarray with
/// almost the same capabilities. Nevertheless, they can't run on the GPU.
/// This function is always call with a Vec of data but then can be called recursively
///
/// Basic transformation implemented for the default collator :
/// ```md
/// - Vec<Scalar> -> ndarray<scalar>
/// - Vec<tuple> -> tuple(ndarray)
/// - Vec<HashMap<Key, Value>> -> HasMap<Key, DefaultCollator::collate(Vec<Value>)
/// - Vec<Array> -> ?
/// - Vec<Vec> -> ?
///
/// Like for Pytorch version, String and u8 aren't changed by the collation (No Op)
/// - Vec<String> -> Vec<String>
/// - Vec<&str> -> Vec<&str>
/// - Vec<u8> -> todo
/// ```
///
#[derive(Default, Debug)]
pub struct DefaultCollator;

pub mod array;
pub mod hash_map;
pub mod scalar;
pub mod tuple;
pub mod vector;
