use ai_dataloader::iterable::DataLoader;

#[cfg(not(feature = "torch"))]
fn main() {
    let dataset = vec![
        (0, vec![1, 23, 4, 0]),
        (1, vec![4, 0, 0, 0]),
        (1, vec![8, 23, 12, 3]),
        (0, vec![2, 45, 4, 0]),
    ];
    // `Vec` implements `IntoIterator` and the `Item` yield are supported by the default collate function,
    // so no further work is required.
    let loader = DataLoader::builder(dataset).batch_size(2).shuffle().build();

    for (label, text) in &loader {
        dbg!(label);
        dbg!(text);
    }
}

#[cfg(feature = "torch")]
use ai_dataloader::collate::TorchCollate;

#[cfg(feature = "torch")]
fn main() {
    // Lets say we have tokenized the input.
    let dataset = vec![
        (0, vec![1, 23, 4, 0]),
        (1, vec![4, 0, 0, 0]),
        (1, vec![8, 23, 12, 3]),
        (0, vec![2, 45, 4, 0]),
    ];
    // `Vec` implements `IntoIterator` and the `Item` yield are supported by the default collate function,
    // so no further work is required.
    let loader = DataLoader::builder(dataset)
        .batch_size(2)
        .shuffle()
        .collate_fn(TorchCollate)
        .build();

    for (label, text) in &loader {
        dbg!(label);
        dbg!(text);
    }
}
