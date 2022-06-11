use super::super::Collate;
use super::DefaultCollator;

impl Collate<Vec<String>> for DefaultCollator {
    type Output = Vec<String>;
    fn collate(batch: Vec<String>) -> Self::Output {
        batch
    }
}
impl<'a> Collate<Vec<&'a str>> for DefaultCollator {
    type Output = Vec<&'a str>;
    fn collate(batch: Vec<&'a str>) -> Self::Output {
        batch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_op() {
        assert_eq!(
            DefaultCollator::collate(vec![String::from("a"), String::from("b")]),
            vec![String::from("a"), String::from("b")]
        );

        assert_eq!(DefaultCollator::collate(vec!["a", "b"]), vec!["a", "b"]);
    }
}
