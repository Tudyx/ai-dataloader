[![CI](https://github.com/Tudyx/ai-dataloader/actions/workflows/ci.yml/badge.svg)](https://github.com/Tudyx/ai-dataloader/actions/workflows/ci.yml) 
[![Crates.io](https://img.shields.io/crates/v/ai-dataloader.svg)](https://crates.io/crates/ai-dataloader)
[![Documentation](https://docs.rs/ai-dataloader/badge.svg)](https://docs.rs/ai-dataloader/)

# ai-dataloader

A rust port of [`pytorch`](https://pytorch.org/) `dataloader` library.

> Note: This project is still heavily in development and is at an early stage.

## Highlights

- Iterable or indexable (Map style) `DataLoader`.
- Customizable `Sampler`, `BatchSampler` and `collate_fn`.
- Default collate function that will cover most of the uses cases, supporting nested type.
- Shuffling for iterable and indexable `DataLoader`.

Feel free to [read the doc](https://docs.rs/ai-dataloader/) that contains tutorials for [`pytorch`](https://pytorch.org/) user.

## Examples

Examples can be found in the [examples](examples/) folder but here there is a simple one

```rust use ai-dataloader::DataLoader;

let loader = DataLoader::builder(vec![(0, "hola"), (1, "hello"), (2, "hallo"), (3, "bonjour")]).batch_size(2).shuffle().build();

for (label, text) in &loader {     
    println!("Label {label:?}");
    println!("Text {text:?}");
}
```

## GPU

[`ndarray`](https://docs.rs/ndarray/latest/ndarray/) can't [currently run on the GPU](https://github.com/rust-ndarray/ndarray/issues/840).

But if your tensor library can be created from a [`ndarray`](https://docs.rs/ndarray/latest/ndarray/), it could be easily integrated.

I've planned to integrate different tensor libraries using [features](https://doc.rust-lang.org/cargo/reference/features.html), feel free to add an issue if you want to submit one.

### Next Features

This features could be added in the future:

- customizable `BatchSampler` (by using a `trait`)
- collect function as a closure 
- `RandomSampler` with replacement
- parallel `dataloader` (using [rayon](https://docs.rs/rayon/latest/rayon/)?)
- distributed `dataloader`

