#[derive(Eq, Hash, PartialEq, Copy, Clone)]
pub struct Identifier {
    pub value: u32,
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
