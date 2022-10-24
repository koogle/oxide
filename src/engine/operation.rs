use std::fmt;

pub enum Operation {
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
