use dataloader_rs::DataLoader;
use ndarray::array;

#[test]
fn text_classification() {
    let dataset = vec![
        (0, "I'm happy"),
        (1, "I'm sad"),
        (0, "It feel goo"),
        (0, "Let's go!"),
    ];
    let loader = DataLoader::builder(dataset).build();
    let mut loader = loader.iter();
    assert_eq!(loader.next(), Some((array![0], vec!["I'm happy"])));
    assert_eq!(loader.next(), Some((array![1], vec!["I'm sad"])));
    assert_eq!(loader.next(), Some((array![0], vec!["It feel goo"])));
    assert_eq!(loader.next(), Some((array![0], vec!["Let's go!"])));
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
    let loader = DataLoader::builder(dataset).batch_size(2).build();
    let mut loader = loader.iter();
    assert_eq!(
        loader.next(),
        Some((array![0, 1], vec!["I'm happy", "I'm sad"]))
    );
    assert_eq!(
        loader.next(),
        Some((array![0, 0], vec!["It feel goo", "Let's go!"]))
    );
}
