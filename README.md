[![CI](https://github.com/Tudyx/ai-dataloader/actions/workflows/ci.yml/badge.svg)](https://github.com/Tudyx/ai-dataloader/actions/workflows/ci.yml) 
[![Coverage](https://github.com/Tudyx/ai-dataloader/actions/workflows/codecov.yml/badge.svg)](https://github.com/Tudyx/ai-dataloader/actions/workflows/codecov.yml)

# ai-dataloader

A rust port of [`pytorch`](https://pytorch.org/) `dataloader` library.

> Note: This project is still heavily in development and is at an early stage.

## Highlights

- Iterable or indexable (Map style) `DataLoader`.
- Customizable `Sampler`, `BatchSampler` and `collate_fn`.
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

In order to collate your data into torch tensor that can run on the GPU, you must activate the torch feature.

This feature relies on the tch crate for bindings to the C++ `libTorch` API. The `libtorch` library is required can be downloaded either automatically or manually. The following provides a reference on how to set up your environment to use these bindings, please refer to the [tch](https://github.com/LaurentMazare/tch-rs) for detailed information or support.

We advise doing the manual installation, as [doctest don't pass with the automatic one](https://github.com/LaurentMazare/tch-rs).
### Next Features

This features could be added in the future:

- customizable `BatchSampler` (by using a `trait`)
- collect function as a closure 
- `RandomSampler` with replacement
- parallel `dataloader` (using [rayon](https://docs.rs/rayon/latest/rayon/)?)
- distributed `dataloader`

