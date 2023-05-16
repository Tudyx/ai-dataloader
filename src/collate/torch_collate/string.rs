use super::super::Collate;
use super::TorchCollate;
use std::ffi::{CStr, CString, OsString};

impl Collate<String> for TorchCollate {
    type Output = Vec<String>;
    fn collate(&self, batch: Vec<String>) -> Self::Output {
        batch
    }
}

impl<'a> Collate<&'a str> for TorchCollate {
    type Output = Vec<&'a str>;
    fn collate(&self, batch: Vec<&'a str>) -> Self::Output {
        batch
    }
}

impl Collate<CString> for TorchCollate {
    type Output = Vec<CString>;
    fn collate(&self, batch: Vec<CString>) -> Self::Output {
        batch
    }
}

impl<'a> Collate<&'a CStr> for TorchCollate {
    type Output = Vec<&'a CStr>;
    fn collate(&self, batch: Vec<&'a CStr>) -> Self::Output {
        batch
    }
}

impl Collate<OsString> for TorchCollate {
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
            TorchCollate::default().collate(vec![String::from("a"), String::from("b")]),
            vec![String::from("a"), String::from("b")]
        );

        assert_eq!(
            TorchCollate::default().collate(vec!["a", "b"]),
            vec!["a", "b"]
        );
    }
}
