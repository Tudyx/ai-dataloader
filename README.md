[![CI](https://github.com/Tudyx/dataloader_rs/actions/workflows/ci.yml/badge.svg)](https://github.com/Tudyx/dataloader_rs/actions/workflows/ci.yml)
[![Coverage](https://github.com/Tudyx/dataloader_rs/actions/workflows/codecov.yml/badge.svg)](https://github.com/Tudyx/dataloader_rs/actions/workflows/codecov.yml)

# dataloader_rs

A rust port of pytorch dataloader library.

> Note: This project is still heavily in development and is at an early stage.

## Highlights

- Random or sequential `Sampler`.
- Customisable `Sampler`
- Default collate function that cover most of the type of the `std`, supporting nested type.
- Cutomizable collate function
  
## Examples

Examples can be found in the [examples](examples/) folder but here there is a simple one

```rust
use dataloader_rs::DataLoader;

let loader = DataLoader::builder(vec![(0, "hola"), (1, "hello"), (2, "hallo"), (3, "bonjour")]).batch_size(2).shuffle().build();

for (label, text) in &loader {
    println!("Label {label:?}");
    println!("Text {text:?}");
}
```

## GPU

[`ndarray`](https://docs.rs/ndarray/latest/ndarray/) can't [currently run on the GPU](https://github.com/rust-ndarray/ndarray/issues/840).

But if you're tensor library can be created from an [`ndarray`](https://docs.rs/ndarray/latest/ndarray/), it could be easily integrated.

I've plan to integrate different tensor libraries using [features](https://doc.rust-lang.org/cargo/reference/features.html), file free to add an issue if you want to submit one.-

### Next Features

This features that could be added in the future:

- customizable batchSampler (by using a `trait`)
- collect function as a closure 
- `RandomSampler` with replacement
- paralllel dataloader (using [rayon](https://docs.rs/rayon/latest/rayon/)?)
- distributed dataloader
