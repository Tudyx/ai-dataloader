use ai_dataloader::iterable::DataLoader;

fn main() {
    let dataset = vec![
        (0, "I'm happy"),
        (1, "I'm sad"),
        (0, "It feel goo"),
        (0, "Let's go!"),
    ];
    // `Vec` implements `IntoIterator` and the `Item` yield are supported by the default collate function,
    // so no further work is required.
    let loader = DataLoader::builder(dataset).batch_size(2).shuffle().build();

    for (label, text) in &loader {
        dbg!(label);
        dbg!(text);
    }
}
