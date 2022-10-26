use std::{fs, borrow::Borrow};

static FOLDER_PREFIX: &str = "./src/mnist/data/";


pub fn read_mnist_labels() {
    let mut file_path = FOLDER_PREFIX.clone().to_owned();
    file_path.push_str("t10k-labels-idx1-ubyte");

    let labels = fs::read(&file_path);
    match labels {
        Ok(r) => println!("{}", r[120]),
        Err(e) => println!("{}", e),
    }
}