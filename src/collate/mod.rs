pub mod default_collate;
pub mod generic_default_collate;
pub mod multiple_output_collate;
pub mod tuple_collate;
// collate rassemble les éléments du batch ensemble
// un numpy array <=> ndarray est just converti en tensor

pub trait Collate<T>: Default {
    type Output;
    fn collate(batch: T) -> Self::Output;
}
#[derive(Default)]
pub struct NoOpCollator;

impl<T> Collate<T> for NoOpCollator {
    type Output = T;
    fn collate(batch: T) -> Self::Output {
        batch
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn no_op_collate() {
        assert_eq!(NoOpCollator::collate(array![1, 2]), array![1, 2]);
    }
}
