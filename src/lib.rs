//! The `dataloader_rs` crate provides a Rust implentation to the [PyTorch](https://pytorch.org/) dataloader
//!
//!
//! ## Highlights
//!
//! - Single threaded Dataloader
//! - Shuffle or Sequential Dataloader
//! - Customisable Sampler and collate function
//!

pub mod collate;
pub mod dataloader;
pub mod dataset;
pub mod fetch;
pub mod sampler;

pub use crate::dataloader::{builder::DataLoaderBuilder, DataLoader};
