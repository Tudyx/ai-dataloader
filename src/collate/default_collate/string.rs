use super::super::Collate;
use super::DefaultCollate;

impl Collate<String> for DefaultCollate {
    type Output = Vec<String>;
    fn collate(batch: Vec<String>) -> Self::Output {
        batch
    }
}
impl<'a> Collate<&'a str> for DefaultCollate {
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
            DefaultCollate::collate(vec![String::from("a"), String::from("b")]),
            vec![String::from("a"), String::from("b")]
        );

        assert_eq!(DefaultCollate::collate(vec!["a", "b"]), vec!["a", "b"]);
    }
}
