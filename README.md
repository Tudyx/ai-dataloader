[![CI](https://github.com/Tudyx/ai-dataloader/actions/workflows/ci.yml/badge.svg)](https://github.com/Tudyx/ai-dataloader/actions/workflows/ci.yml) 
[![Crates.io](https://img.shields.io/crates/v/ai-dataloader.svg)](https://crates.io/crates/ai-dataloader)
[![Documentation](https://docs.rs/ai-dataloader/badge.svg)](https://docs.rs/ai-dataloader/)

# ai-dataloader

A rust port of [`pytorch`](https://pytorch.org/) `dataloader` library.

## Highlights

- Iterable or indexable (Map style) `DataLoader`.
- Customizable `Sampler`, `BatchSampler` and `collate_fn`.
- Parallel dataloader using [`rayon`] for indexable dataloader (experimental).
- Integration with [`ndarray`](https://docs.rs/ndarray/latest/ndarray/) and [`tch-rs`](https://github.com/LaurentMazare/tch-rs), CPU and GPU support.
- Default collate function that will automatically collate most of your type (supporting nesting).
- Shuffling for iterable and indexable `DataLoader`.

More info in the [documentation](https://docs.rs/ai-dataloader/).

## Examples

Examples can be found in the [examples](examples/) folder but here there is a simple one

```rust 
use ai_dataloader::DataLoader;
let loader = DataLoader::builder(vec![(0, "hola"), (1, "hello"), (2, "hallo"), (3, "bonjour")]).batch_size(2).shuffle().build();

for (label, text) in &loader {     
    println!("Label {label:?}");
    println!("Text {text:?}");
}
```

## [`tch-rs`](https://github.com/LaurentMazare/tch-rs) integration

In order to collate your data into torch tensor that can run on the GPU, you must activate the `tch` feature.

This feature relies on the tch crate for bindings to the C++ `libTorch` API. The `libtorch` library is required can be downloaded either automatically or manually. The following provides a reference on how to set up your environment to use these bindings, please refer to the [tch](https://github.com/LaurentMazare/tch-rs) for detailed information or support.

### Next Features

This features could be added in the future:

- `RandomSampler` with replacement
- parallel `dataloader` for iterable dataset
- distributed `dataloader`


### MSRV

The current MSRV is 1.63.

[`rayon`]: https://docs.rs/rayon/latest/rayon/
