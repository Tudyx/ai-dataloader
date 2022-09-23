///! # Example with a dataset containing images.
///
/// This example is a simplified version of [this pytorch tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html).
///
use dataloader_rs::{DataLoader, Dataset, GetSample, Len};
use ndarray::{Array1, Array2, Array3};
use nshare::ToNdarray3;
use std::{env, path::PathBuf};

struct FaceLandmarksDataset {
    root_dir: PathBuf,
    landmarks_frame: Vec<csv::StringRecord>,
}

impl FaceLandmarksDataset {
    fn new(csv_file: &str, root_dir: PathBuf) -> FaceLandmarksDataset {
        let mut landmarks_frame = csv::Reader::from_path(csv_file).unwrap();

        // We parse the reader beacause so we can easily manipulate the data
        let landmarks_frame: Vec<csv::StringRecord> = landmarks_frame
            .records()
            .map(|record| record.unwrap())
            .collect();
        FaceLandmarksDataset {
            root_dir: root_dir.to_owned(),
            landmarks_frame,
        }
    }
}

impl Dataset for FaceLandmarksDataset {}

impl Len for FaceLandmarksDataset {
    /// Returns the number of elements in the collection, also referred to
    /// as its 'length'.
    fn len(&self) -> usize {
        self.landmarks_frame.len()
    }
}
impl GetSample for FaceLandmarksDataset {
    type Sample = (Array3<u8>, Array2<f64>);

    /// Return the dataset sample corresponding to the index
    fn get_sample(&self, index: usize) -> Self::Sample {
        let record = &self.landmarks_frame[index];

        // We parse the landmark from the CSV into and ndarray
        let landmark: Array1<f64> = record
            .iter()
            .skip(1)
            .map(|elem| elem.parse().unwrap())
            .collect();
        let landmark = landmark.into_shape((68, 2)).expect("Incompatible shape");

        // We parse the image into an ndarray
        let mut image_path = self.root_dir.clone();
        let filename = &record[0];
        image_path.push(filename);
        let image = image::open(image_path).expect("Something went wrong while reading the image");
        // We can apply transformation to the image if we want here, like rescaling, rotating, etc..
        let image = image.resize_exact(250, 250, image::imageops::FilterType::Nearest);
        // The axes/dimensions created follow the pytorch convention : Color x Height x Width
        let image = image.into_rgb8().into_ndarray3();
        (image, landmark)
    }
}

fn main() {
    let dataset = FaceLandmarksDataset::new(
        "examples/image/dataset/face_landmarks.csv",
        env::current_dir().unwrap().join("examples/image/dataset/"),
    );
    let loader = DataLoader::builder(dataset).batch_size(4).build();

    for (batch_id, (image, landmarks)) in loader.iter().enumerate() {
        println!(
            "Batch {}: image shape {:?}, landmark shape {:?}",
            batch_id,
            image.shape(),
            landmarks.shape()
        );
    }
}
