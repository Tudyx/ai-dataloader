pub mod collate;
pub mod dataloader;
pub mod dataset;
pub mod fetch;
pub mod sampler;
#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
