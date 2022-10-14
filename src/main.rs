use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;
//use std::time::SystemTime;
use rand::Rng;
use std::collections::VecDeque;
use std::fmt;

// Value class

enum Operation {
    ADD,
    MUL,
    RELU,
    NONE,
}

impl Default for Operation {
    fn default() -> Operation {
        Operation::NONE
    }
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        return write!(
            f,
            "{}",
            match self {
                Operation::ADD => "Add",
                Operation::MUL => "Mul",
                Operation::RELU => "Relu",
                Operation::NONE => "None",
            }
        );
    }
}

#[derive(Eq, Hash, PartialEq, Copy, Clone)]
struct Identifier {
    value: u32,
}

static mut COUNTER: u32 = 0;

impl Default for Identifier {
    fn default() -> Identifier {
        unsafe {
            COUNTER += 1;
            return Identifier { value: COUNTER };
        }
    }
}

type ValueRef = Rc<RefCell<Value>>;

#[derive(Default)]
struct Value {
    value: f64,
    grad: f64,
    needs_grad: bool,
    id: Identifier,
    operation: Operation,
    previous_nodes: Vec<ValueRef>,
    color: String,
}

struct Engine {}

impl Engine {
    fn add(left: &ValueRef, right: &ValueRef) -> ValueRef {
        let v = Value {
            value: left.borrow().value + right.borrow().value,
            needs_grad: true,
            operation: Operation::ADD,
            grad: 0.0,
            previous_nodes: vec![Rc::clone(&left), Rc::clone(&right)],
            id: Identifier::default(),
            color: String::default(),
        };
        return Rc::new(RefCell::new(v));
    }

    fn mul(left: &ValueRef, right: &ValueRef) -> ValueRef {
        let v = Value {
            value: left.borrow().value * right.borrow().value,
            needs_grad: true,
            operation: Operation::MUL,
            grad: 0.0,
            previous_nodes: vec![Rc::clone(&left), Rc::clone(&right)],
            id: Identifier::default(),
            color: String::default(),
        };
        return Rc::new(RefCell::new(v));
    }

    fn relu(node: &ValueRef) -> ValueRef {
        let value = match node.borrow().value > 0.0 {
            true => node.borrow().value,
            false => 0.0,
        };

        let v = Value {
            value,
            needs_grad: true,
            operation: Operation::RELU,
            grad: 0.0,
            previous_nodes: vec![Rc::clone(&node)],
            id: Identifier::default(),
            color: String::default(),
        };
        return Rc::new(RefCell::new(v));
    }

    fn inv(node: &ValueRef) -> ValueRef {
        return Engine::mul(node, &Value::from(-1.0));
    }

    fn pow(node: &ValueRef) -> ValueRef {
        return Engine::mul(node, node);
    }
}

impl Value {
    fn random() -> ValueRef {
        return Value::from(rand::thread_rng().gen_range(0.0..1.0));
    }

    fn from(value: f64) -> ValueRef {
        let mut new_value = Value::default();
        new_value.value = value;
        return Rc::new(RefCell::new(new_value));
    }

    fn backward_recursive(
        &self,
        mut pointers: VecDeque<ValueRef>,
        mut visited: HashSet<Identifier>,
    ) -> (VecDeque<ValueRef>, HashSet<Identifier>) {
        if !visited.contains(&self.id) {
            visited.insert(self.id);
            for node in self.previous_nodes.iter() {
                (pointers, visited) = node.borrow_mut().backward_recursive(pointers, visited);
                pointers.push_back(Rc::clone(node));
            }
        }
        return (pointers, visited);
    }

    fn backward(&mut self) {
        // implement toplogical search
        let (mut pointers, _) = self.backward_recursive(VecDeque::new(), HashSet::new());
        self.update_previous();
        while pointers.len() > 0 {
            let node = pointers.pop_back();

            match node {
                Some(resolved) => {
                    resolved.borrow_mut().update_previous();
                }
                None => {}
            }
        }
    }

    fn forward_step(&mut self) {
        self.color = String::from("forward");
        // remember to reset grads on forward pass
        //if !self.needs_grad {
        self.grad = 0.0;
        //}

        match self.operation {
            Operation::ADD => {
                self.value =
                    self.previous_nodes[0].borrow().value + self.previous_nodes[1].borrow().value;
            }
            Operation::MUL => {
                self.value =
                    self.previous_nodes[0].borrow().value * self.previous_nodes[1].borrow().value;
            }
            Operation::RELU => {
                self.value = match self.previous_nodes[0].borrow().value > 0.0 {
                    true => self.previous_nodes[0].borrow().value,
                    false => 0.0,
                };
            }
            Operation::NONE => {}
        }
    }

    fn forward(&mut self) {
        let (mut pointers, _) = self.backward_recursive(VecDeque::new(), HashSet::new());
        while pointers.len() > 0 {
            let node = pointers.pop_front();

            match node {
                Some(resolved) => {
                    resolved.borrow_mut().forward_step();
                }
                None => {}
            }
        }
        self.forward_step();
    }

    fn update_previous(&mut self) {
        self.color = String::from("backward");

        match self.operation {
            Operation::ADD => {
                if self.needs_grad(0) {
                    self.update_previous_node(0, self.grad);
                }
                if self.needs_grad(1) {
                    self.update_previous_node(1, self.grad);
                }
            }
            Operation::MUL => {
                if self.needs_grad(0) {
                    self.update_previous_node(0, self.get_previous_value(1) * self.grad);
                }
                if self.needs_grad(1) {
                    self.update_previous_node(1, self.get_previous_value(0) * self.grad);
                }
            }
            Operation::RELU => {
                let mut first = self.previous_nodes[0].borrow_mut();
                if first.needs_grad {
                    first.grad += match self.value > 0.0 {
                        true => self.grad,
                        false => 0.0,
                    };
                }
            }
            Operation::NONE => {}
        }
    }

    fn needs_grad(&self, index: usize) -> bool {
        return self.previous_nodes[index].borrow().needs_grad;
    }

    fn update_previous_node(&self, index: usize, value: f64) {
        println!(
            "update node id: {} index: {}, value: {}",
            self.previous_nodes[index].borrow().id.value,
            index,
            value
        );
        self.previous_nodes[index].borrow_mut().grad += value;
    }

    fn get_previous_value(&self, index: usize) -> f64 {
        return self.previous_nodes[index].borrow().value;
    }
}

// Neuron class
struct Neuron {
    parameters: Vec<ValueRef>,
    inputs: Vec<ValueRef>,
    output: ValueRef,
}

impl Neuron {
    fn new(inputs: &Vec<ValueRef>) -> Neuron {
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

    fn zero_grad(&self) {
        for param in self.parameters.iter() {
            param.borrow_mut().grad = 0.0;
            param.borrow_mut().color = String::from("");
        }
    }

    fn update(&self, alpha: f64) {
        for param in self.parameters.iter() {
            if !param.borrow().needs_grad {
                continue;
            }

            let v = { param.borrow().grad };
            let c = { param.borrow().color.clone() };
            let id = { param.borrow().id };

            let old_v = { param.borrow().value };
            param.borrow_mut().value -= v * alpha;
            let new_v = { param.borrow().value };
            println!(
                "{} color: {} grad: {} old: {} new: {}",
                id.value, c, v, old_v, new_v
            );
        }
    }

    fn set(&self, inputs: Vec<f64>) {
        assert_eq!(inputs.len(), self.inputs.len());

        for (index, input) in inputs.iter().enumerate() {
            self.inputs[index].borrow_mut().value = *input;
        }
    }
}

// Layers
struct Layer {
    neurons: Vec<Neuron>,
    outputs: Vec<ValueRef>,
}

impl Layer {
    fn new(n_neurons: usize, inputs: &Vec<ValueRef>) -> Layer {
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

    fn zero_grad(&self) {
        for neuron in self.neurons.iter() {
            neuron.zero_grad()
        }
    }

    fn update(&self, alpha: f64) {
        for neuron in self.neurons.iter() {
            neuron.update(alpha)
        }
    }

    fn set(&self, inputs: Vec<f64>) {
        for neuron in self.neurons.iter() {
            neuron.set(inputs.clone())
        }
    }
}

struct MLP {
    layers: Vec<Layer>,
    inputs: Vec<ValueRef>,
}

impl MLP {
    fn new(sizes: Vec<usize>, input_size: usize) -> MLP {
        let mut layers: Vec<Layer> = vec![];
        layers.reserve(sizes.len());
        let inputs: Vec<ValueRef> = (0..input_size).map(|_x| Value::from(0.0)).collect();
        let mut output: &Vec<ValueRef> = &inputs;
        for (index, size) in sizes.iter().enumerate() {
            let layer = Layer::new(*size, output);
            layers.push(layer);
            output = &layers[index].outputs;
        }

        return MLP { layers, inputs };
    }

    fn set(&mut self, inputs: Vec<f64>) {
        self.layers[0].set(inputs);
    }

    fn outputs(&self) -> Vec<ValueRef> {
        let mut last_layer: Vec<ValueRef> = vec![];
        last_layer.reserve(self.layers[self.layers.len() - 1].outputs.len());
        for output in self.layers[self.layers.len() - 1].outputs.iter() {
            last_layer.push(output.clone());
        }

        return last_layer;
    }

    fn zero_grad(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.zero_grad();
        }
    }

    fn update(&self, alpha: f64) {
        for (index, layer) in self.layers.iter().enumerate() {
            println!("update layer {}", index);
            layer.update(alpha);
        }
    }
}

fn main() {
    let mut net = MLP::new(vec![4, 6, 4], 4);
    // let layer = net.layers[0];
    let values = vec![1.0, 0.0,0.0,1.0];
    let alpha = 0.1;
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
    for (index, o) in outputs.iter().enumerate() {
        println!("{}: {}", index, o.borrow().value);
    }
    println!("{}", loss.borrow().value);

    for i in 0..200 {
        net.zero_grad();
        loss.borrow_mut().grad = 1.0;
        loss.borrow_mut().backward();
        net.update(alpha);
        loss.borrow_mut().forward();
        

        let final_o = net.outputs();
        for (index, o) in final_o.iter().enumerate() {
            println!("{}: {}", index, o.borrow().value);
        }

        /*for v in loss.borrow().previous_nodes.iter() {
            println!("> {} {}", v.borrow().value, v.borrow().operation);
            for v2 in v.borrow().previous_nodes.iter() {
                println!(">> {} {}", v2.borrow().value, v2.borrow().operation);
            }
        }*/
        println!("Loss after epoch{}: {}", i, loss.borrow().value);
    }

    /*if loss.borrow().value == 0.5 {
        let final_o = net.outputs();
        for (index, o) in final_o.iter().enumerate() {
            println!("{}: {}", index, o.borrow().value);
        }
    }*/
    /*let a = Value::from(1.0);
    a.borrow_mut().needs_grad = true;
    let b = Value::from(1.3);
    let c = Engine::mul(&a, &b);
    let e = Engine::add(&c, &Value::from(2.0));
    e.borrow_mut().grad = 10.0;
    e.borrow_mut().backward();*/
    // c.borrow().backward();
    // let test = ref_a.borrow();
    // println!("Hello, world!");
}

// Outdated code to be removed later

/*fn relu(self) -> (Value, ValueRef) {
    let value = match self.value > 0.0 {
        true => self.value,
        false => 0.0,
    };
    let self_ref = Rc::new(RefCell::new(self));
    let output = Value {
        value: value,
        needs_grad: true,
        operation: Operation::RELU,
        grad: 0.0,
        previous_nodes: vec![Rc::clone(&self_ref)],
        id: Identifier::default()
    };
    return (output, self_ref);
}*/

/*fn combine_with_refs(left: Value, right: Value, new_value: f64, operation: Operation) -> (Value, ValueRef, ValueRef) {
    let l_ref = Rc::new(RefCell::new(left));
    let r_ref = Rc::new(RefCell::new(right));

    let output = Value {
        value: new_value,
        needs_grad: true,
        operation,
        grad: 0.0,
        previous_nodes: vec![Rc::clone(&l_ref), Rc::clone(&r_ref)],
        id: Identifier::default()
    };
    return (output, l_ref, r_ref);
}

impl Add for Value {
    type Output = (Value, ValueRef, ValueRef);

    fn add(self, other: Value) -> (Value, ValueRef, ValueRef) {
        let new_value = self.value + other.value;
        return combine_with_refs(self, other, new_value, Operation::ADD);
    }
}

impl Add<f64> for Value {
    type Output = (Value, ValueRef);

    fn add(self, other: f64) -> (Value, ValueRef) {
        let new_value = self.value + other;
        let (out, l_ref, _) =  combine_with_refs(self, Value::from(other), new_value, Operation::ADD);
        return (out, l_ref);
    }
}

impl Mul for Value {
    type Output = (Value, ValueRef, ValueRef);

    fn mul(self, other: Value) -> (Value, ValueRef, ValueRef) {
        let new_value = self.value * other.value;
        return combine_with_refs(self, other, new_value, Operation::MUL);
    }
}

impl Mul<f64> for Value {
    type Output = (Value, ValueRef);

    fn mul(self, other: f64) -> (Value, ValueRef) {
        let new_value = self.value * other;
        let (out, l_ref, _) =  combine_with_refs(self, Value::from(other), new_value, Operation::MUL);
        return (out, l_ref);
    }
}*/
