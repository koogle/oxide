use std::cell::RefCell;
use std::rc::Rc;
use std::collections::VecDeque;
use std::collections::HashSet;
use rand::Rng;

use crate::engine::operation::Operation;
use crate::engine::id::Identifier;


pub type ValueRef = Rc<RefCell<Value>>;

#[derive(Default)]
pub struct Value {
    pub value: f64,
    pub grad: f64,
    pub needs_grad: bool,
    pub id: Identifier,
    pub operation: Operation,
    pub previous_nodes: Vec<ValueRef>,
    pub color: String,
}

impl Value {
    pub fn random() -> ValueRef {
        return Value::from(rand::thread_rng().gen_range(0.0..1.0));
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

    pub fn forward(&mut self) {
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
        self.previous_nodes[index].borrow_mut().grad += value;
    }

    fn get_previous_value(&self, index: usize) -> f64 {
        return self.previous_nodes[index].borrow().value;
    }
}