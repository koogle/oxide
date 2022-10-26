# Minimal neural net in Rust

![image](https://user-images.githubusercontent.com/6087389/198004793-3a02eaa0-4b18-4666-a56e-5624000a7e76.png)

Because what if we want to run a net on a very small form factor


### Inspiration

- https://github.com/karpathy/micrograd
- https://github.com/geohot/tinygrad

### MLP Example

```rust
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
```
