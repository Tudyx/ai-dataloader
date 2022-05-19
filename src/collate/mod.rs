pub mod default_collate;

// collate rassemble les éléments du batch ensemble
// un numpy array <=> ndarray est just converti en tensor

pub trait Collect<T>: Default {
    type Output;
    fn collect(batch: T) -> Self::Output;
}
#[derive(Default)]
pub struct NoOpCollector;

impl<T> Collect<T> for NoOpCollector {
    type Output = T;
    fn collect(batch: T) -> Self::Output {
        batch
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn no_op_collate() {
        assert_eq!(NoOpCollector::collect(array![1, 2]), array![1, 2]);
    }
}
