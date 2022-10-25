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

#[cfg(test)]
mod test {
    use crate::engine::Value;
    use crate::engine::VALUE_RANDOM_SEED;
    use crate::engine::Engine;
    use super::*;

    #[test]
    fn test_mlp() {
        VALUE_RANDOM_SEED.store(1, std::sync::atomic::Ordering::Relaxed);

        let mut net = MLP::new(vec![4, 6, 4], 4);
        let values = vec![1.0, 0.0,0.0,1.0];
        let alpha = 0.01;
        net.set(values.clone());
        let outputs = net.outputs();

        let mut loss = Value::from(0.0);
        for (index, output) in outputs.iter().enumerate() {
            let tmp = Engine::pow(&Engine::add(
                output,
                &Engine::inv(&Value::from(values[index])),
            ));

            loss = Engine::add(&loss, &tmp);
        }
        loss = Engine::mul(&loss, &Value::from(1.0 / values.len() as f64));
        loss.borrow_mut().forward();
        loss.borrow_mut().grad = 1.0;

        for _ in 0..1000 {
            net.zero_grad();
            loss.borrow_mut().grad = 1.0;
            loss.borrow_mut().backward();
            net.update(alpha);
            loss.borrow_mut().forward();
        }
        println!("{}", loss.borrow().value);
        assert!(loss.borrow().value < 1e-10);
    }

}