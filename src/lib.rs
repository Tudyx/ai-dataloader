// Force commenting everything
// #![deny(missing_docs)]
// #![deny(clippy::all)]
// #![deny(clippy::missing_docs_in_private_items)]

//! The `dataloader_rs` crate provides a Rust implementation to the [PyTorch] DataLoader.
//!
//!
//! Unlike the python version where almost everithing happen in runtime, `dataloader_rs` is built on Rust's powerful trait system.
//!
//!
//! ## Highlights
//!
//! - Shuffle or Sequential Dataloader.
//! - Customizable Sampler and collate function.
//! - Default collate function that cover most of the type of the `std`, supporting nested type.
//!
//! ## Examples
//!
//! Examples can be found in the [examples] folder.
//!
//! ## PyTorch DataLoader function equivalents
//!
//! ### DataLoader creation
//!
//! PyTorch | `dataloader_rs` | Notes
//! --------|-----------------|-------
//! `DataLoader(dataset)` | `DataLoader::builder(dataset).build()` | Create a DataLoader with default parameter
//! `DataLoader(dataset, batch_size=2)` | `DataLoader::builder(dataset).with_batch_size(2).build()` | Setup the batch size
//! `DataLoader(dataset, shuffle=True)` | `let loader: DataLoader<_, RandomSampler> = DataLoader::builder(dataset).build()` | Shuffle the data
//!
//! ### DataLoader iteration
//!
//! PyTorch | `dataloader_rs` | Notes
//! --------|-----------------|-------
//! `for text, label in data_loader:` | `for (text, label) in data_loader.iter()` | Simple iteration
//!
//! ## To be done
//!
//! - Parallel DataLoader
//! - Iterable dataset (dataset than are not indexable and have possibly no length)
//! - Integrate with a Tensor library on GPU when one will be mature enough (Open for suggestion).
//!
//! [PyTorch]: https://pytorch.org/
//! [examples]: https://github.com/Tudyx/dataloader_rs/tree/main/examples
//!

mod dataloader;
mod dataset;
mod fetch;

pub mod collate;
pub mod sampler;

pub use dataloader::{builder::DataLoaderBuilder, DataLoader};
pub use dataset::{Dataset, GetSample, Len};
