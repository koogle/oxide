use crate::engine::ValueRef;
use crate::engine::Value;
use crate::engine::Engine;
 

pub struct Neuron {
    parameters: Vec<ValueRef>,
    inputs: Vec<ValueRef>,
    pub output: ValueRef,
}

impl Neuron {
    pub fn new(inputs: &Vec<ValueRef>) -> Neuron {
        let size = inputs.len();
        // let mut inputs: Vec <ValueRef> = (1..size).map(|_x| Value::from(0.0)).collect();
        let parameters: Vec<ValueRef> = (0..(size + 1)).map(|_x| Value::random()).collect();
        let mut input_refs: Vec<ValueRef> = vec![];
        input_refs.reserve(inputs.len());

        for input in inputs {
            input_refs.push(input.clone());
        }

        for parameter in parameters.iter() {
            parameter.borrow_mut().needs_grad = true;
        }

        let mut output: ValueRef = parameters[0].clone();
        for index in 0..size {
            output = Engine::add(
                &output,
                &Engine::mul(&parameters[index + 1], &input_refs[index]),
            );
        }

        return Neuron {
            parameters,
            inputs: input_refs,
            output: Engine::relu(&output),
        };
    }

    pub fn zero_grad(&self) {
        for param in self.parameters.iter() {
            param.borrow_mut().grad = 0.0;
            param.borrow_mut().color = String::from("");
        }
    }

    pub fn update(&self, alpha: f64) {
        for param in self.parameters.iter() {
            if !param.borrow().needs_grad {
                continue;
            }

            let v = { param.borrow().grad };
            param.borrow_mut().value -= v * alpha;
        }
    }

    pub fn set(&self, inputs: Vec<f64>) {
        assert_eq!(inputs.len(), self.inputs.len());

        for (index, input) in inputs.iter().enumerate() {
            self.inputs[index].borrow_mut().value = *input;
        }
    }
}