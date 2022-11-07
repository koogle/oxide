use crate::engine::Engine;
use crate::engine::Value;
use crate::nn::MLP;
use byteorder::{BigEndian, ByteOrder};
use std::fs;

static FOLDER_PREFIX: &str = "./src/mnist/data/";

#[derive(Clone)]
pub struct ImageDimensions {
    height: usize,
    width: usize,
}

pub struct MnistImage {
    dimensions: ImageDimensions,
    label: u8,
    pixels: Vec<u8>,
    index: u32,
}

pub fn read_mnist_labels() -> Vec<MnistImage> {
    let mut label_file_path = FOLDER_PREFIX.clone().to_owned();
    label_file_path.push_str("train-labels-idx1-ubyte");

    let raw_bytes = fs::read(&label_file_path);
    let labels: Vec<u8>;
    match raw_bytes {
        Ok(r) => labels = read_lables(r),
        Err(e) => {
            println!("{}", e);
            return vec![];
        }
    }

    let mut images_file_path = FOLDER_PREFIX.clone().to_owned();
    images_file_path.push_str("train-images-idx3-ubyte");
    let raw_images = fs::read(&images_file_path);
    let images: Vec<MnistImage>;
    match raw_images {
        Ok(r) => images = read_images(r, labels),
        Err(e) => {
            // break
            println!("{}", e);
            return vec![];
        }
    }
    return images;
}

fn read_lables(raw_bytes: Vec<u8>) -> Vec<u8> {
    let magic_number = BigEndian::read_u32(&raw_bytes[0..4]);
    assert_eq!(2049, magic_number);
    let number_of_labels = BigEndian::read_u32(&raw_bytes[4..8]);
    assert_eq!(60000, number_of_labels);

    let offset: usize = 8;
    let mut output: Vec<u8> = vec![0; number_of_labels as usize];
    output.copy_from_slice(&raw_bytes[offset..(offset + number_of_labels as usize)]);
    return output;
}

fn read_images(raw_bytes: Vec<u8>, labels: Vec<u8>) -> Vec<MnistImage> {
    let magic_number = BigEndian::read_u32(&raw_bytes[0..4]);
    assert_eq!(2051, magic_number);
    let number_of_images = BigEndian::read_u32(&raw_bytes[4..8]);
    assert_eq!(60000, number_of_images);
    let width = BigEndian::read_u32(&raw_bytes[8..12]) as usize;
    assert_eq!(28, width);
    let height = BigEndian::read_u32(&raw_bytes[12..16]) as usize;
    assert_eq!(28, height);
    let mut offset: usize = 16;

    let mut images: Vec<MnistImage> = Vec::with_capacity(number_of_images as usize);
    for index in 0..number_of_images {
        let mut pixels: Vec<u8> = vec![0; 28 * 28];
        pixels.copy_from_slice(&raw_bytes[offset..(offset + 28 * 28)]);

        let image = MnistImage {
            dimensions: ImageDimensions { width, height },
            index,
            label: labels[index as usize],
            pixels,
        };
        images.push(image);
        offset += 28 * 28;
    }

    return images;
}

fn print_image(image: &MnistImage) {
    for row in 0..image.dimensions.height {
        let mut row_str = String::with_capacity(image.dimensions.width as usize);
        for column in 0..image.dimensions.width {
            let index = (row * image.dimensions.width + column) as usize;
            if image.pixels[index] > 123 {
                row_str.push('0');
            } else {
                row_str.push(' ');
            }
        }
        println!("{}", row_str);
    }
    println!("Label {}", image.label);
}

fn to_f64(pixels: Vec<u8>) -> Vec<f64> {
    let mut out = Vec::with_capacity(pixels.len());
    for pixel in pixels.iter() {
        out.push(*pixel as f64);
    }
    return out;
}

fn one_hot_encoding(label: u8) -> Vec<f64> {
    let mut out = vec![0.0; 10];
    assert_eq!(label < 10, true);
    out[label as usize] = 1.0;
    return out;
}


#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_read_mnist_labels() {
        let images = read_mnist_labels();
        assert_ne!(images.len(), 0);
        let dimension = images[0].dimensions.clone();
        let mut mlp = MLP::new(
            vec![dimension.width * dimension.height, 10, 20, 10],
            vec![false, false, false, true],
            dimension.width * dimension.height,
        );
        let alpha = 0.01;
        let outputs = mlp.outputs();
        
        let image_index = 0;

        mlp.set(to_f64(images[image_index].pixels.clone()));
        let y_hat = one_hot_encoding(images[image_index].label);
        
        let mut loss = Value::from(0.0);
        for (index, output) in outputs.iter().enumerate() {
            let tmp = Engine::pow(&Engine::add(
                output,
                &Engine::inv(&Value::from(y_hat[index])),
            ));
    
            loss = Engine::add(&loss, &tmp);
        }
        loss = Engine::mul(&loss, &Value::from(1.0 / y_hat.len() as f64));
        loss.borrow_mut().forward();
        loss.borrow_mut().grad = 1.0;

        for _ in 0..1000 {
            //mlp.zero_grad();
            //loss.borrow_mut().grad = 1.0;
            loss.borrow_mut().backward();
            //mlp.update(alpha);
            loss.borrow_mut().forward();
            println!("{}", loss.borrow().value);
        }
    }
}
