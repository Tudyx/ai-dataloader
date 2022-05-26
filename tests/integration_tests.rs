use dataloader_rs::{DataLoader, DataLoaderBuilder};
use ndarray::array;

#[test]
fn text_classification() {
    let dataset = vec![
        (0, "I'm happy"),
        (1, "I'm sad"),
        (0, "It feel goo"),
        (0, "Let's go!"),
    ];
    let loader: DataLoader<_> = DataLoaderBuilder::new(dataset).build();
    let mut loader = loader.iter();
    assert_eq!(loader.next(), Some((array![0], array!["I'm happy"])));
    assert_eq!(loader.next(), Some((array![1], array!["I'm sad"])));
    assert_eq!(loader.next(), Some((array![0], array!["It feel goo"])));
    assert_eq!(loader.next(), Some((array![0], array!["Let's go!"])));
    assert_eq!(loader.next(), None);
}

#[test]
fn text_classification_batch() {
    let dataset = vec![
        (0, "I'm happy"),
        (1, "I'm sad"),
        (0, "It feel goo"),
        (0, "Let's go!"),
    ];
    let loader: DataLoader<_> = DataLoaderBuilder::new(dataset).with_batch_size(2).build();
    let mut loader = loader.iter();
    assert_eq!(
        loader.next(),
        Some((array![0, 1], array!["I'm happy", "I'm sad"]))
    );
    assert_eq!(
        loader.next(),
        Some((array![0, 0], array!["It feel goo", "Let's go!"]))
    );
}
