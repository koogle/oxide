use std::sync::atomic::AtomicU32;

#[derive(Eq, Hash, PartialEq, Copy, Clone)]
pub struct Identifier {
    pub value: u32,
}

static COUNTER: AtomicU32 = AtomicU32::new(0);

impl Default for Identifier {
    fn default() -> Identifier {
        COUNTER.store(COUNTER.load(std::sync::atomic::Ordering::Relaxed) + 1, std::sync::atomic::Ordering::Relaxed);

        return Identifier { value: COUNTER.load(std::sync::atomic::Ordering::Relaxed) };
    }
}
