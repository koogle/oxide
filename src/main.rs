mod engine;
mod nn;

use engine::Engine;
use engine::Value;
use nn::MLP;

fn main() {
    let mut net = MLP::new(vec![4, 6, 4], 4);
    // let layer = net.layers[0];
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

    for i in 0..1000 {
        net.zero_grad();
        loss.borrow_mut().grad = 1.0;
        loss.borrow_mut().backward();
        net.update(alpha);
        loss.borrow_mut().forward();
        

        let final_o = net.outputs();
        println!("Loss after epoch{}: {}", i, loss.borrow().value);
    }
}