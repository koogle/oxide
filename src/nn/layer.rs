use crate::engine::ValueRef;
use crate::nn::Neuron;

pub struct Layer {
    neurons: Vec<Neuron>,
    pub outputs: Vec<ValueRef>,
}

impl Layer {
    pub fn new(n_neurons: usize, inputs: &Vec<ValueRef>) -> Layer {
        let neurons: Vec<Neuron> = (0..n_neurons)
            .map(|_| Neuron::new(&inputs.clone()))
            .collect();
        let mut outputs: Vec<ValueRef> = vec![];
        outputs.reserve(n_neurons);

        for neuron in neurons.iter() {
            outputs.push(neuron.output.clone());
        }

        return Layer { neurons, outputs };
    }

    pub fn zero_grad(&self) {
        for neuron in self.neurons.iter() {
            neuron.zero_grad()
        }
    }

    pub fn update(&self, alpha: f64) {
        for neuron in self.neurons.iter() {
            neuron.update(alpha)
        }
    }

    pub fn set(&self, inputs: Vec<f64>) {
        for neuron in self.neurons.iter() {
            neuron.set(inputs.clone())
        }
    }
}
