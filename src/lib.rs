// Force commenting everything
// #![deny(missing_docs)]
// #![deny(clippy::all)]
// #![deny(clippy::missing_docs_in_private_items)]

//! The `dataloader_rs` crate provides a Rust implementation to the [PyTorch](https://pytorch.org/) DataLoader
//!
//!
//! ## Highlights
//!
//! - Shuffle or Sequential Dataloader
//! - Customizable Sampler and collate function
//!
//! ## PyTorch DataLoader function equivalents
//!
//! ### DataLoader creation
//!
//! PyTorch | `dataloader_rs` | Notes
//! --------|-----------------|-------
//! `DataLoader(dataset)` | `DataLoaderBuilder::new(dataset).build()` | Create a DataLoader with default parameter
//! `DataLoader(dataset, batch_size=2)` | `DataLoaderBuilder::new(dataset).with_batch_size(2).build()` | Setup the batch size
//! `DataLoader(dataset, shuffle=True)` | `let loader: DataLoader<_, RandomSampler> = DataLoaderBuilder::new(dataset).build()` | Shuffle the data
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
//! - Integrate with a Tensor library on GPU when one will be mature enough
//!

pub mod collate;
pub mod dataloader;
pub mod dataset;
pub mod fetch;
pub mod sampler;

pub use crate::dataloader::{builder::DataLoaderBuilder, DataLoader};
