use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::cell::RefCell;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::atomic::AtomicU64;

use crate::engine::id::Identifier;
use crate::engine::operation::Operation;

pub type ValueRef = Rc<RefCell<Value>>;
pub static VALUE_RANDOM_SEED: AtomicU64 = AtomicU64::new(0);

#[derive(Default)]
pub struct Value {
    pub value: f64,
    pub grad: f64,
    pub needs_grad: bool,
    pub id: Identifier,
    pub operation: Operation,
    pub previous_nodes: Vec<ValueRef>,
    pub has_been_reset: bool,
    pub backward_graph: VecDeque<ValueRef>,
}

impl Value {
    pub fn random() -> ValueRef {
        if VALUE_RANDOM_SEED.load(std::sync::atomic::Ordering::Relaxed) == 0 {
            return Value::from(rand::thread_rng().gen_range(0.0..1.0));
        } else {
            return Value::from(
                ChaCha8Rng::seed_from_u64(
                    VALUE_RANDOM_SEED.load(std::sync::atomic::Ordering::Relaxed),
                )
                .gen_range(0.0..1.0),
            );
        }
    }

    pub fn from(value: f64) -> ValueRef {
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

    pub fn backward(&mut self) {
        // implement toplogical search
        if self.backward_graph.len() == 0 {
            (self.backward_graph, _) = self.backward_recursive(VecDeque::new(), HashSet::new());
        }

        // flush grads if necessary
        for pointer in self.backward_graph.iter_mut() {
            pointer.borrow_mut().propagate_zero_grad();
        }
        self.update_previous();

        for pointer in self.backward_graph.iter_mut().rev() {
            pointer.borrow_mut().has_been_reset = false;
            pointer.borrow_mut().update_previous();
        }
    }

    fn forward_step(&mut self) {
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

    pub fn forward(&mut self) {
        if self.backward_graph.len() == 0 {
            (self.backward_graph, _) = self.backward_recursive(VecDeque::new(), HashSet::new());
        }

        for pointer in self.backward_graph.iter_mut() {
            pointer.borrow_mut().forward_step();
        }
        self.forward_step();
    }

    fn update_previous(&mut self) {
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
        self.previous_nodes[index].borrow_mut().grad += value;
    }

    fn get_previous_value(&self, index: usize) -> f64 {
        return self.previous_nodes[index].borrow().value;
    }

    pub fn zero_grad(&mut self) {
        self.grad = 0.0;
        self.has_been_reset = true;
    }

    fn propagate_zero_grad(&mut self) {
        let mut needs_reset = false;
        for previous in self.previous_nodes.iter() {
            if previous.borrow().has_been_reset {
                needs_reset = true;
            }
        }
        if needs_reset {
            self.grad = 0.0;
            self.has_been_reset = true;
        }
    }
}
