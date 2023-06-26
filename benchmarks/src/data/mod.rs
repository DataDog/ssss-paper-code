/// Utilities to load from files or generate data to be used as an input to the sketches.
use std::{
    fmt,
    fs::{read_dir, File},
    io::{BufRead, BufReader},
    iter::Iterator,
    path::{Path, PathBuf},
};

use flate2::read::GzDecoder;

pub mod synth;

pub trait Dataset: fmt::Display {
    type Label;
    type Item;

    fn iter(&self) -> Box<dyn Iterator<Item = (Self::Label, Self::Item)>>;
}

#[derive(Clone, Debug)]
pub struct FolderDataset {
    path: PathBuf,
    max_per_file: usize,
}

impl FolderDataset {
    pub fn new(path: impl AsRef<Path>, max_per_file: usize) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            max_per_file,
        }
    }
}

impl fmt::Display for FolderDataset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.path.file_name().unwrap().to_str().unwrap())
    }
}

impl Dataset for FolderDataset {
    type Label = String;
    type Item = String;

    fn iter(&self) -> Box<dyn Iterator<Item = (Self::Label, Self::Item)>> {
        let max_per_file = self.max_per_file;
        Box::new(
            read_dir(&self.path)
                .unwrap()
                .map(|path| path.unwrap())
                .map(|dir_entry| dir_entry.path())
                .flat_map(move |file_path| {
                    BufReader::new(GzDecoder::new(File::open(file_path).unwrap()))
                        .lines()
                        .take(max_per_file)
                        .map(|line| {
                            let line_str = line.unwrap();
                            let fields = line_str.split(',').take(2).collect::<Vec<&str>>();
                            (fields[1].to_string(), fields[0].to_string())
                        })
                }),
        )
    }
}

#[derive(Clone, Debug)]
pub struct FileDataset {
    path: PathBuf,
    max_per_file: usize,
}

impl FileDataset {
    pub fn new(path: impl AsRef<Path>, max_per_file: usize) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            max_per_file,
        }
    }
}

impl fmt::Display for FileDataset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.path.file_name().unwrap().to_str().unwrap())
    }
}

impl Dataset for FileDataset {
    type Label = String;
    type Item = String;

    fn iter(&self) -> Box<dyn Iterator<Item = (Self::Label, Self::Item)>> {
        Box::new(
            BufReader::new(GzDecoder::new(File::open(&self.path).unwrap()))
                .lines()
                .take(self.max_per_file)
                .map(|line| {
                    let line_str = line.unwrap();
                    let fields = line_str.split(',').take(2).collect::<Vec<&str>>();
                    (fields[1].to_string(), fields[0].to_string())
                }),
        )
    }
}
