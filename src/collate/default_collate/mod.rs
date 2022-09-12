/// Basic collator that mimic the default collate function from PyTorch
/// As they are no such lib with the same functionnality as PyTorch tensor in Rust,
/// data is collated inside ndarray. Ndarray is the rust equivalent of `numpy.ndarray` with
/// almost the same capabilities. Nevertheless, they can't run on the GPU.
///
/// This function is always call with a Vec of data but can be also be called recursively
///
/// Basic transformation implemented for the default collator :
/// ```md
/// - Vec<Scalar> -> ndarray<scalar>
/// - Vec<tuple> -> tuple(ndarray)
/// - Vec<HashMap<Key, Value>> -> HasMap<Key, DefaultCollate::collate(Vec<Value>)
/// - Vec<Array> -> ?
/// - Vec[V1_i, V2_i, ...]` -> Vec[default_collate([V1_1, V1_2, ...]), default_collate([V2_1, V2_2, ...]), ...]
///
///
/// Like for Pytorch version, String and u8 aren't changed by the collation (No Op)
/// - Vec<String> -> Vec<String>
/// - Vec<&str> -> Vec<&str>
/// - Vec<u8> -> todo
/// ```
///
#[derive(Default, Debug)]
pub struct DefaultCollate;

mod array;
mod map;
mod ndarray;
mod nonzero;
mod primitive;
mod sequence;
mod string;
mod tuple;
