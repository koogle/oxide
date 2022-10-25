use crate::engine::ValueRef;
use crate::engine::Value;
use crate::nn::Layer;

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(sizes: Vec<usize>, input_size: usize) -> MLP {
        let mut layers: Vec<Layer> = vec![];
        layers.reserve(sizes.len());
        let inputs: Vec<ValueRef> = (0..input_size).map(|_x| Value::from(0.0)).collect();
        let mut output: &Vec<ValueRef> = &inputs;
        for (index, size) in sizes.iter().enumerate() {
            let layer = Layer::new(*size, output);
            layers.push(layer);
            output = &layers[index].outputs;
        }

        return MLP { layers };
    }

    pub fn set(&mut self, inputs: Vec<f64>) {
        self.layers[0].set(inputs);
    }

    pub fn outputs(&self) -> Vec<ValueRef> {
        let mut last_layer: Vec<ValueRef> = vec![];
        last_layer.reserve(self.layers[self.layers.len() - 1].outputs.len());
        for output in self.layers[self.layers.len() - 1].outputs.iter() {
            last_layer.push(output.clone());
        }

        return last_layer;
    }

    pub fn zero_grad(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.zero_grad();
        }
    }

    pub fn update(&self, alpha: f64) {
        for layer in self.layers.iter() {
            layer.update(alpha);
        }
    }
}