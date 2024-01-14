use super::super::Collate;
use super::DefaultCollate;
use std::ffi::{CStr, CString, OsString};

impl Collate<String> for DefaultCollate {
    type Output = Vec<String>;
    fn collate(&self, batch: Vec<String>) -> Self::Output {
        batch
    }
}

impl<'a> Collate<&'a str> for DefaultCollate {
    type Output = Vec<&'a str>;
    fn collate(&self, batch: Vec<&'a str>) -> Self::Output {
        batch
    }
}

impl Collate<CString> for DefaultCollate {
    type Output = Vec<CString>;
    fn collate(&self, batch: Vec<CString>) -> Self::Output {
        batch
    }
}

impl<'a> Collate<&'a CStr> for DefaultCollate {
    type Output = Vec<&'a CStr>;
    fn collate(&self, batch: Vec<&'a CStr>) -> Self::Output {
        batch
    }
}

impl Collate<OsString> for DefaultCollate {
    type Output = Vec<OsString>;
    fn collate(&self, batch: Vec<OsString>) -> Self::Output {
        batch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_op() {
        assert_eq!(
            DefaultCollate.collate(vec![String::from("a"), String::from("b")]),
            vec![String::from("a"), String::from("b")]
        );

        assert_eq!(DefaultCollate.collate(vec!["a", "b"]), vec!["a", "b"]);
    }
}
