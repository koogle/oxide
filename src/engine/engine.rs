use std::cell::RefCell;
use std::rc::Rc;

use crate::engine::value::ValueRef;
use crate::engine::value::Value;
use crate::engine::id::Identifier;
use crate::engine::operation::Operation;

pub struct Engine {}

impl Engine {
    pub fn add(left: &ValueRef, right: &ValueRef) -> ValueRef {
        let v = Value {
            value: left.borrow().value + right.borrow().value,
            needs_grad: true,
            operation: Operation::ADD,
            grad: 0.0,
            previous_nodes: vec![Rc::clone(&left), Rc::clone(&right)],
            id: Identifier::default(),
            color: String::default(),
            has_been_reset: false,
        };
        return Rc::new(RefCell::new(v));
    }

    pub fn mul(left: &ValueRef, right: &ValueRef) -> ValueRef {
        let v = Value {
            value: left.borrow().value * right.borrow().value,
            needs_grad: true,
            operation: Operation::MUL,
            grad: 0.0,
            previous_nodes: vec![Rc::clone(&left), Rc::clone(&right)],
            id: Identifier::default(),
            color: String::default(),
            has_been_reset: false,
        };
        return Rc::new(RefCell::new(v));
    }

    pub fn relu(node: &ValueRef) -> ValueRef {
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
            has_been_reset: false,
        };
        return Rc::new(RefCell::new(v));
    }

    pub fn inv(node: &ValueRef) -> ValueRef {
        return Engine::mul(node, &Value::from(-1.0));
    }

    pub fn pow(node: &ValueRef) -> ValueRef {
        return Engine::mul(node, node);
    }
}
