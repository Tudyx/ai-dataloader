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
//! - Iterable or indexable (Map style) `DataLoader`.
//! - Customizable `Sampler`, `BatchSampler` and `collate_fn`.
//! - Default collate function that will cover most of the uses cases, supporting nested type.
//! - Shuffling for iterable and indexable `DataLoader`.
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
//!
//!
//! ## Choosing between Iterable or Indexable dataloader
//!
//! You can choose Iterable `DataLoader` for instance if your dataset arrived from a stream and you don't have random access into it.
//! It's also useful for large dataset to only load a small part at the time in the RAM. When the order mater, for instance in Reinforcement Learning, Iterable
//! `DataLoader` is also a good fit.
//!
//! Otherwise Indexable Dataloader (Map style in [PyTorch] doc) maybe be a good fit.
//!
//! Both support shuffling the sample.
//!
//! To choose iterable:
//!
//! ```
//! use ai_dataloader::iterable::DataLoader;
//! ```
//!
//! To choose indexable:
//!
//! ```
//! use ai_dataloader::indexable::DataLoader;
//! ```
//!
//!
//!
//! [PyTorch]: https://pytorch.org/
//! [examples]: https://github.com/Tudyx/ai-dataloader/tree/main/examples

pub mod indexable;
pub mod iterable;

pub mod collate;

pub use indexable::{sampler, Dataset, GetSample, Len, NdarrayDataset};
