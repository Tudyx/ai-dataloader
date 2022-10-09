#![deny(
    clippy::all,
    clippy::cargo,
    rustdoc::all,
    missing_docs,
    missing_debug_implementations,
    rust_2018_idioms,
    unreachable_pub
)]
// I've a false positive on this one.
#![allow(clippy::derive_partial_eq_without_eq)]

//! The `ai-dataloader` crate provides a Rust implementation to the [PyTorch] `DataLoader`.
//!
//!
//! Unlike the python version where almost everything happens in runtime, `ai-dataloader` is built on Rust's powerful trait system.
//!
//!
//! ## Highlights
//!
//! - Shuffle or Sequential Sampler.
//! - Customizable sampler
//! - Default collate function that covers most of the type of the standard library, supporting nested type.
//! - Customizable  collate function
//!
//! ## Examples
//!
//! Examples can be found in the [examples] folder.
//!
//! ## `PyTorch` `DataLoader` function equivalents
//!
//! ### `DataLoader` creation
//!
//! `PyTorch` | `ai-dataloader` | Notes
//! --------|-----------------|-------
//! `DataLoader(dataset)` | `DataLoader::builder(dataset).build()` | Create a `DataLoader` with default parameters
//! `DataLoader(dataset, batch_size=2)` | `DataLoader::builder(dataset).batch_size(2).build()` | Setup the batch size
//! `DataLoader(dataset, shuffle=True)` | `DataLoader::builder(dataset).shuffle().build()` | Shuffle the data
//! `DataLoader(dataset, sampler=CustomSampler)` | `DataLoader::builder(dataset).sampler::<CustomSampler>().build()` | Provide a custom sampler
//!
//! ### Combined options
//!
//! `PyTorch` | `ai-dataloader`
//! --------|-----------------
//! `DataLoader(dataset, shuffle=True, batch_size=2, drop_last=True, collate_fn=CustomCollate)` | `DataLoaderBuilder::new(dataset).shuffle().batch_size(2).drop_last().collate_fn(CustomCollate).build()`
//!
//! ### `DataLoader` iteration
//!
//! `PyTorch` | `ai-dataloader` | Notes
//! --------|-----------------|-------
//! `for text, label in data_loader:` | `for (text, label) in data_loader.iter()` | Simple iteration
//!
//!
//! [PyTorch]: https://pytorch.org/
//! [examples]: https://github.com/Tudyx/ai-dataloader/tree/main/examples
//!

mod dataloader;
mod dataset;
mod fetch;

pub mod collate;
pub mod sampler;

pub use dataloader::{Builder, DataLoader};
pub use dataset::{Dataset, GetSample, Len, NdarrayDataset};
