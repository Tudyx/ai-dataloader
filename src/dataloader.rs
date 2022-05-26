use crate::collate::default_collate::DefaultCollator;
use crate::collate::Collate;
use crate::dataset::{Dataset, GetItem};
use crate::fetch::{Fetcher, MapDatasetFetcher};
use crate::sampler::batch_sampler::{BatchIterator, BatchSampler};
use crate::sampler::DefaultSampler;
use crate::sampler::HasLength;
use crate::sampler::Sampler;
pub struct DataLoader<D, S = DefaultSampler, C = DefaultCollator>
where
    D: Dataset<C>,
    S: Sampler,
    C: Collate<Vec<<D as GetItem<usize>>::Output>>,
{
    dataset: D,
    // how many sample to launch per batch
    batch_size: usize,
    sampler: S,
    batch_sampler: BatchSampler<S>,
    num_worker: u32,
    drop_last: bool,
    collate_fn: C,
}

impl<D, S, C> DataLoader<D, S, C>
where
    D: Dataset<C>,
    S: Sampler,
    C: Collate<Vec<<D as GetItem<usize>>::Output>>,
{
    fn len(&self) -> usize {
        self.batch_sampler.len()
    }
}
// combinaison de _BaseDataLoaderIter et _SingleProcessDataLoaderIter
pub struct SingleProcessDataLoaderIter<'dataset, D, S = DefaultSampler, C = DefaultCollator>
where
    D: Dataset<C>,
    S: Sampler,
    C: Collate<Vec<<D as GetItem<usize>>::Output>>,
{
    sampler_iter: BatchIterator<S::IntoIter>,
    num_yielded: u64,
    data_fetcher: MapDatasetFetcher<'dataset, D, C>,
}

impl<'dataset, D, S, C> SingleProcessDataLoaderIter<'dataset, D, S, C>
where
    D: Dataset<C>,
    S: Sampler,
    C: Collate<Vec<<D as GetItem<usize>>::Output>>,
{
    fn new(loader: &DataLoader<D, S, C>) -> SingleProcessDataLoaderIter<D, S, C> {
        SingleProcessDataLoaderIter {
            sampler_iter: loader.batch_sampler.iter(),
            num_yielded: 0,
            data_fetcher: MapDatasetFetcher {
                dataset: &loader.dataset,
                collate_fn: C::default(),
            },
        }
    }
    fn next_index(&mut self) -> Option<Vec<usize>> {
        self.sampler_iter.next()
    }
    fn next_data(&mut self) -> Option<C::Output> {
        let index = self.next_index();
        if let Some(index) = index {
            let data = self.data_fetcher.fetch(index);
            return Some(data);
        }
        None
    }
    fn reset(&mut self) {
        self.num_yielded = 0;
        // TODO: store the batch sampler to be able to make a new iter
        // self.sampler_iter = loader.batch_sampler.iter();
    }
}
impl<'dataset, D, S, C> Iterator for SingleProcessDataLoaderIter<'dataset, D, S, C>
where
    D: Dataset<C>,
    S: Sampler,
    C: Collate<Vec<<D as GetItem<usize>>::Output>>,
{
    type Item = C::Output;
    // type Item = Vec<Collect<Vec<<D as GetItem<usize>>::Output>>>;
    fn next(&mut self) -> Option<Self::Item> {
        let data = self.next_data();

        if let Some(data) = data {
            self.num_yielded += 1;
            return Some(data);
        }
        None
    }
}
impl<'dataset, D, S, C> IntoIterator for &'dataset DataLoader<D, S, C>
where
    D: Dataset<C>,
    S: Sampler,
    C: Collate<Vec<<D as GetItem<usize>>::Output>>,
{
    type Item = C::Output;

    // type Item = <DefaultCollector as Collect<Vec<D::Output>>>::Output;
    type IntoIter = SingleProcessDataLoaderIter<'dataset, D, S, C>;
    fn into_iter(self) -> Self::IntoIter {
        SingleProcessDataLoaderIter::new(self)
    }
}

impl<D, S, C> DataLoader<D, S, C>
where
    D: Dataset<C>,
    S: Sampler,
    C: Collate<Vec<<D as GetItem<usize>>::Output>>,
{
    pub fn iter(&self) -> SingleProcessDataLoaderIter<D, S, C> {
        SingleProcessDataLoaderIter::new(self)
    }
}

// builder pour construire des dataloader. Doit rester dans le même mod car accède au membre privée.
pub struct DataLoaderBuilder<D, S = DefaultSampler, C = DefaultCollator>
where
    D: Dataset<C>,
    S: Sampler,
    C: Collate<Vec<<D as GetItem<usize>>::Output>>,
{
    dataset: D,
    batch_size: usize,
    sampler: Option<S>,
    batch_sampler: Option<BatchSampler<S>>,
    num_worker: u32,
    drop_last: bool,
    collate_fn: Option<C>,
}
impl<D, S, C> DataLoaderBuilder<D, S, C>
where
    D: Dataset<C>,
    S: Sampler,
    C: Collate<Vec<<D as GetItem<usize>>::Output>>,
{
    pub fn new(dataset: D) -> Self {
        Self {
            dataset,
            batch_size: 1,
            sampler: None,
            batch_sampler: None,
            num_worker: 0,
            drop_last: false,
            collate_fn: None,
        }
    }
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
    pub fn with_num_worker(mut self, num_worker: u32) -> Self {
        self.num_worker = num_worker;
        self
    }
    pub fn with_collate_fn(mut self, collate_fn: C) -> Self {
        self.collate_fn = Some(collate_fn);
        self
    }
    pub fn with_sampler(mut self, sampler: S) -> Self {
        self.sampler = Some(sampler);
        self
    }

    pub fn build(mut self) -> DataLoader<D, S, C> {
        if self.batch_sampler.is_some() && self.batch_size != 0
            || self.sampler.is_some()
            || self.drop_last
        {
            panic!("batch_sampler option is mutually exclusive with batch_size,  sampler, and drop_last'")
        }

        let sampler = self.sampler.unwrap_or_else(|| S::new(self.dataset.len()));

        if self.batch_sampler.is_none() {
            self.batch_sampler = Some(BatchSampler {
                sampler: sampler.clone(),
                batch_size: self.batch_size,
                drop_last: self.drop_last,
            })
        }
        DataLoader {
            dataset: self.dataset,
            batch_size: self.batch_size,
            sampler,
            batch_sampler: self.batch_sampler.unwrap(),
            num_worker: self.num_worker,
            drop_last: self.drop_last,
            collate_fn: self.collate_fn.unwrap_or_default(),
        }
    }
}
// fn create_hasmap_dataset() {
//     let mut dataset = Vec::new();
//     let mut sample = HashMap::new();
//     sample.insert("label".to_string(), "1".to_string());
//     sample.insert("text".to_string(), "first sample content".to_string());
//     dataset.push(sample);
//     let mut sample = HashMap::new();
//     sample.insert("label".to_string(), "2".to_string());
//     sample.insert("text".to_string(), "second sample content".to_string());
//     dataset.push(sample);
//     dataset.len();
//     println!("{:?}", dataset);
// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collate::NoOpCollator;
    use crate::dataset::ndarray_dataset::NdarrayDataset;
    use crate::sampler::random_sampler::RandomSampler;
    use crate::sampler::sequential_sampler::SequentialSampler;
    use crate::sampler::HasLength;
    use ndarray::{array, Array, Array1, Array4, Axis, Ix1, Ix4, Slice};
    use ndarray_rand::rand_distr::{Normal, Uniform};
    use ndarray_rand::RandomExt;
    use std::collections::HashMap;

    #[test]
    fn one_dimension_basic() {
        let dataset = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        // why type annotation is required even with default genric type ?
        let dataloader: DataLoader<_> = DataLoaderBuilder::new(dataset).with_batch_size(2).build();

        let mut iter = dataloader.iter();
        assert_eq!(iter.next(), Some(array![1, 2]));
        assert_eq!(iter.next(), Some(array![3, 4]));
        assert_eq!(iter.next(), Some(array![5, 6]));
        assert_eq!(iter.next(), Some(array![7, 8]));
        assert_eq!(iter.next(), Some(array![9, 10]));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn two_iteration() {
        let dataset = vec![1, 2, 3, 4];
        let dataloader: DataLoader<_> = DataLoaderBuilder::new(dataset).with_batch_size(2).build();

        let mut iter = dataloader.iter();
        assert_eq!(iter.next(), Some(array![1, 2]));
        assert_eq!(iter.next(), Some(array![3, 4]));
        assert_eq!(iter.next(), None);
        let mut iter = dataloader.iter();
        assert_eq!(iter.next(), Some(array![1, 2]));
        assert_eq!(iter.next(), Some(array![3, 4]));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn one_dimension_basic_string() {
        let dataset = vec![String::from("a"), String::from("b")];
        let dataloader: DataLoader<_> = DataLoaderBuilder::new(dataset).with_batch_size(1).build();

        let mut iter = dataloader.iter();
        assert_eq!(iter.next(), Some(vec![String::from("a")]));
        assert_eq!(iter.next(), Some(vec![String::from("b")]));
        assert_eq!(iter.next(), None);
    }
    #[test]
    fn test_collator() {
        let dataset = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let dataloader: DataLoader<_, SequentialSampler, NoOpCollator> =
            DataLoaderBuilder::new(dataset)
                .with_collate_fn(NoOpCollator)
                .with_batch_size(2)
                .build();
        let mut iter = dataloader.iter();

        assert_eq!(iter.next(), Some(vec![1, 2]));
        assert_eq!(iter.next(), Some(vec![3, 4]));
        assert_eq!(iter.next(), Some(vec![5, 6]));
        assert_eq!(iter.next(), Some(vec![7, 8]));
        assert_eq!(iter.next(), Some(vec![9, 10]));
        assert_eq!(iter.next(), None);
    }
    struct TestDataLoader<S: Sampler> {
        loader: DataLoader<NdarrayDataset<f64, f64, Ix4, Ix1>, S>,
        data: Array4<f64>,
        labels: Array1<f64>,
        dataset: NdarrayDataset<f64, f64, Ix4, Ix1>,
    }
    enum TestDataLoaderData {
        Sequential(TestDataLoader<SequentialSampler>),
        Random(TestDataLoader<RandomSampler>),
    }
    fn get_loader_with_dummy_data(batch_size: usize, shuffle: bool) -> TestDataLoaderData {
        let normal: Normal<f64> = Normal::new(0.0, 1.0).unwrap();
        let data = Array::random((100, 2, 3, 5), normal);
        let labels = Array::random(100, Uniform::<f64>::new(0., 50.));
        let dataset = NdarrayDataset {
            ndarrays: (data.clone(), labels.clone()),
        };

        if shuffle {
            let loader: DataLoader<_, RandomSampler> = DataLoaderBuilder::new(dataset.clone())
                .with_batch_size(batch_size)
                .build();
            TestDataLoaderData::Random(TestDataLoader {
                loader,
                data,
                labels,
                dataset,
            })
        } else {
            let loader: DataLoader<_, SequentialSampler> = DataLoaderBuilder::new(dataset.clone())
                .with_batch_size(batch_size)
                .build();
            TestDataLoaderData::Sequential(TestDataLoader {
                loader,
                data,
                labels,
                dataset,
            })
        }
    }

    #[test]
    fn sequential_non_batch() {
        let test_dataloader_data = tests::get_loader_with_dummy_data(1, false);
        let test_data;
        if let TestDataLoaderData::Sequential(test_dataloader_data) = test_dataloader_data {
            test_data = test_dataloader_data;
        } else {
            panic!("Excpected a Sequential loader")
        }
        let mut current_idx = 0;

        for (idx, (sample, label)) in test_data.loader.iter().enumerate() {
            assert_eq!(sample[0], test_data.data.index_axis(Axis(0), idx));
            assert_eq!(label[0], test_data.labels.index_axis(Axis(0), idx));
            current_idx = idx;
        }
        assert_eq!(current_idx, test_data.dataset.len() - 1);
    }

    #[test]
    fn sequential_batch() {
        let batch_size = 2;
        let test_dataloader_data = tests::get_loader_with_dummy_data(2, false);
        let test_data;
        if let TestDataLoaderData::Sequential(test_dataloader_data) = test_dataloader_data {
            test_data = test_dataloader_data;
        } else {
            panic!("Expected a sequential loader")
        }

        let mut current_i = 0;

        for (i, (sample, label)) in test_data.loader.iter().enumerate() {
            let idx = i * batch_size;
            // Even if the display on the console are the same we can compare them due to mismatch type (Array0<f64> and f64), hence the convertion
            // (work out of the box with numpy)
            let label: Array<_, _> = label.iter().map(|x| x.clone().into_scalar()).collect();
            assert_eq!(
                label,
                test_data
                    .labels
                    .slice_axis(Axis(0), Slice::from(idx..idx + batch_size))
            );
            // It seems to be that tensor/numpy do a elementwise comparison, producing an array of bool and ndarray produce the equivalent of (a == b).all()
            // elementwise comparison (which is not the default unlike numpy where elementwise is invoked when "a == b")
            let vec: Vec<_> = sample
                .iter()
                .flatten()
                .zip(
                    test_data
                        .data
                        .slice_axis(Axis(0), Slice::from(idx..idx + batch_size)),
                )
                .map(|x| x.0 == x.1)
                .collect();
            assert!(vec.iter().all(|x| *x == true));
            current_i = i;
        }
        assert_eq!(current_i, (test_data.dataset.len() - 1) / batch_size);
    }

    #[test]
    fn shuffle_non_batch() {
        let test_dataloader_data = tests::get_loader_with_dummy_data(1, true);
        let test_data;
        if let TestDataLoaderData::Random(test_dataloader_data) = test_dataloader_data {
            test_data = test_dataloader_data;
        } else {
            panic!("Expected a random loader")
        }
        let mut found_data: HashMap<_, _> = (0..test_data.data.len())
            .zip(vec![0; test_data.data.len()])
            .collect();
        let mut found_labels: HashMap<_, _> = (0..test_data.labels.len())
            .zip(vec![0; test_data.labels.len()])
            .collect();
        let mut current_i = 0;
        for (i, (sample, label)) in test_data.loader.iter().enumerate() {
            current_i = i;
            let mut current_data_point_idx = 0;
            for (data_point_idx, data_point) in test_data.data.outer_iter().enumerate() {
                current_data_point_idx = data_point_idx;
                if data_point == sample[0] {
                    assert_eq!(found_data[&data_point_idx], 0);
                    *found_data.get_mut(&data_point_idx).unwrap() += 1;
                    break;
                }
            }

            assert_eq!(
                label[0],
                test_data.labels.index_axis(Axis(0), current_data_point_idx)
            );
            *found_labels.get_mut(&current_data_point_idx).unwrap() += 1;
            assert_eq!(found_data.values().sum::<usize>(), i + 1);
            assert_eq!(found_labels.values().sum::<usize>(), i + 1);
        }
        assert_eq!(current_i, test_data.dataset.len() - 1)
    }

    #[test]
    fn shuffle_batch() {
        let batch_size = 2;
        let test_dataloader_data = tests::get_loader_with_dummy_data(batch_size, true);
        let test_data;
        if let TestDataLoaderData::Random(test_dataloader_data) = test_dataloader_data {
            test_data = test_dataloader_data;
        } else {
            panic!("Expected a random loader")
        }
        let mut found_data: HashMap<_, _> = (0..test_data.data.len())
            .zip(vec![0; test_data.data.len()])
            .collect();
        let mut found_labels: HashMap<_, _> = (0..test_data.labels.len())
            .zip(vec![0; test_data.labels.len()])
            .collect();
        let mut current_i = 0;
        for (i, (batch_samples, batch_targets)) in test_data.loader.iter().enumerate() {
            current_i = i;
            for (sample, label) in batch_samples.iter().zip(batch_targets) {
                let mut current_data_point_idx = 0;
                for (data_point_idx, data_point) in test_data.data.outer_iter().enumerate() {
                    current_data_point_idx = data_point_idx;
                    if data_point == sample {
                        assert_eq!(found_data[&data_point_idx], 0);
                        *found_data.get_mut(&data_point_idx).unwrap() += 1;
                        break;
                    }
                }
                assert_eq!(
                    label,
                    test_data.labels.index_axis(Axis(0), current_data_point_idx)
                );
                *found_labels.get_mut(&current_data_point_idx).unwrap() += 1;
            }
            assert_eq!(found_data.values().sum::<usize>(), (i + 1) * batch_size);
            assert_eq!(found_labels.values().sum::<usize>(), (i + 1) * batch_size);
        }
        assert_eq!(current_i, (test_data.dataset.len() - 1) / batch_size)
    }

    #[test]
    fn text_classification() {
        let dataset = vec![
            (0, "I'm happy"),
            (1, "i'm sad"),
            (0, "it feel goo"),
            (0, "Let's go!"),
        ];
        let loader: DataLoader<_> = DataLoaderBuilder::new(dataset).build();
        for (label, text) in loader.iter() {
            println!("label {label}");
            println!("text {text}");
        }
    }

    #[test]
    fn text_classification_batch() {
        let dataset = vec![
            (0, "I'm happy"),
            (1, "i'm sad"),
            (0, "it feel goo"),
            (0, "Let's go!"),
        ];
        let loader: DataLoader<_> = DataLoaderBuilder::new(dataset).with_batch_size(2).build();
        for (label, text) in loader.iter() {
            println!("label {label}");
            println!("text {text}");
        }
    }
}
