/// Default Collate function that mimic the [`default_collate` function](https://pytorch.org/docs/stable/data.html#automatic-batching-default) from ``PyTorch``.
///
/// As they are no such lib with the same functionnality as `PyTorch` tensor in Rust,
/// data is collated inside `ndarray`. Ndarray is the rust equivalent of `numpy.ndarray` with
/// almost the same capabilities. Nevertheless, they can't run on the GPU.
///
///
/// Basic transformation implemented for the default Collate :
///
/// - `Vec<Scalar>` -> `ndarray<scalar>`
/// - `Vec<tuple>` -> `tuple(ndarray)`
/// - `Vec<HashMap<Key, Value>>` -> `HasMap<Key, DefaultCollate::collate(Vec<Value>)`
/// - `Vec<Array>` -> `Vec<Stack Array>`
/// - `Vec[V1_i, V2_i, ...]` -> `Vec[DefaultCollate::collate([V1_1, V1_2, ...]), DefaultCollate::collate([V2_1, V2_2, ...]), ...]`
///
///
/// Like for `PyTorch` version, `String` and `u8` aren't changed by the collation (No Op).
///
/// - `Vec<String>` -> `Vec<String>`
/// - `Vec<&str>` -> `Vec<&str>`
/// - `Vec<u8>` -> `Vec<u8>`
///
///
#[derive(Default, Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct DefaultCollate;

mod array;
mod map;
mod ndarray;
mod nonzero;
mod primitive;
mod reference;
mod sequence;
mod string;
mod tuple;
