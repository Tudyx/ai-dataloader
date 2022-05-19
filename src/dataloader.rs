use crate::collate::default_collate::DefaultCollector;
use crate::collate::Collate;
use crate::dataset::{Dataset, GetItem};
use crate::fetch::{Fetcher, MapDatasetFetcher};
use crate::sampler::batch_sampler::{BatchIterator, BatchSampler};
use crate::sampler::DefaultSampler;
use crate::sampler::Sampler;

pub struct DataLoader<D, S = DefaultSampler, C = DefaultCollector>
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
    // Change it to private to see the changes
    current_index: usize,
    collate_fn: C,
}
// combinaison de _BaseDataLoaderIter et _SingleProcessDataLoaderIter
pub struct SingleProcessDataLoaderIter<'a, D, S = DefaultSampler, C = DefaultCollector>
where
    D: Dataset<C>,
    S: Sampler,
    C: Collate<Vec<<D as GetItem<usize>>::Output>>,
{
    dataset: &'a D,
    sampler_iter: BatchIterator<S::IntoIter>,
    num_yielded: u64,
    data_fetcher: MapDatasetFetcher<D, C>,
}

impl<'a, D, S, C> SingleProcessDataLoaderIter<'a, D, S, C>
where
    D: Dataset<C>,
    S: Sampler,
    C: Collate<Vec<<D as GetItem<usize>>::Output>>,
{
    fn new(loader: &'a DataLoader<D, S, C>) -> SingleProcessDataLoaderIter<'a, D, S, C> {
        SingleProcessDataLoaderIter {
            dataset: &loader.dataset,
            sampler_iter: loader.batch_sampler.iter(),
            num_yielded: 0,
            data_fetcher: MapDatasetFetcher {
                dataset: loader.dataset.clone(),
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
impl<'a, D, S, C> Iterator for SingleProcessDataLoaderIter<'a, D, S, C>
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
impl<'a, D, S, C> IntoIterator for &'a DataLoader<D, S, C>
where
    D: Dataset<C>,
    S: Sampler,
    C: Collate<Vec<<D as GetItem<usize>>::Output>>,
{
    type Item = C::Output;

    // type Item = <DefaultCollector as Collect<Vec<D::Output>>>::Output;
    type IntoIter = SingleProcessDataLoaderIter<'a, D, S, C>;
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
pub struct DataLoaderBuilder<D, S = DefaultSampler, C = DefaultCollector>
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
            current_index: 0,
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
    use crate::collate::NoOpCollector;
    use crate::dataset::NdarrayDataset;
    use crate::sampler::random_sampler::RandomSampler;
    use crate::sampler::sequential_sampler::SequentialSampler;
    use ndarray::{array, Array, Array2, ArrayBase};
    use ndarray_rand::rand_distr::{Normal, Uniform};
    use ndarray_rand::RandomExt;

    use super::*;

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
    fn test_collector() {
        let dataset = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let dataloader: DataLoader<_, SequentialSampler, NoOpCollector> =
            DataLoaderBuilder::new(dataset)
                .with_collate_fn(NoOpCollector)
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
    // def _test_sequential(self, loader):
    // batch_size = loader.batch_size
    // if batch_size is None:
    //     for idx, (sample, target) in enumerate(loader):
    //         self.assertEqual(sample, self.data[idx])
    //         self.assertEqual(target, self.labels[idx])
    //     self.assertEqual(idx, len(self.dataset) - 1)
    // else:
    //     for i, (sample, target) in enumerate(loader):
    //         idx = i * batch_size
    //         self.assertEqual(sample, self.data[idx:idx + batch_size])
    //         self.assertEqual(target, self.labels[idx:idx + batch_size])
    //     self.assertEqual(i, math.floor((len(self.dataset) - 1) / batch_size))

    #[test]
    fn test_sequential() {
        let normal: Normal<f64> = Normal::new(0.0, 1.0).unwrap();
        let data = Array::random((100, 2, 3, 5), normal);
        let label = Array::random((1, 100), Uniform::<f64>::new(0., 50.));
        let dataset = NdarrayDataset {
            ndarrays: (data, label),
        };
        let loader: DataLoader<_> = DataLoaderBuilder::new(dataset).build();
        // for (idx, (sample, target)) in loader.iter().enumerate() {}
        for el in loader.iter().take(1) {
            println!("{el:?}");
        }
    }
    #[test]
    fn test_random() {
        let dataset = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let loader: DataLoader<_, RandomSampler> =
            DataLoaderBuilder::new(dataset).with_batch_size(3).build();

        for samples in loader.iter() {
            println!("{samples}");
        }

        // This seems to be equivalent (and less verbose)
        // let loader: DataLoader<_, DefaultCollector, RandomSampler> =
        //     DataLoaderBuilder::new(dataset).build();
    }

    // helper that will be used by other test
    // fn test_sequential<D: Dataset>(mut loader: DataLoader<D>)
    // where
    //     DefaultCollector: Collect<Vec<<D as GetItem<usize>>::Output>>,
    // {
    //     let batch_size = loader.batch_size;
    //     for (i, (sample, target)) in &loader.enumerate(){
    //         let mut idx = i * batch_size;
    //     }

    // }
}
