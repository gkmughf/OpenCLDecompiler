from src.ir.instructions.generic import GenericInstruction
from src.ir.registers.reg import PredReg, Val


class Label(GenericInstruction):
    def __init__(self, name: Val, is_scalar: bool = True):
        super().__init__("label", name, is_scalar=is_scalar)
        self.name = name

    def to_text(self) -> str:
        return f"label {self.name.to_text()}"

    def writes_first_operand(self) -> bool:
        return False

    def is_control_flow(self) -> bool:
        return True


class Branch(GenericInstruction):
    def __init__(self, target: Val, is_scalar: bool = True, predicate: PredReg | None = None):
        super().__init__("bra", target, is_scalar=is_scalar, predicate=predicate)
        self.target = target

    def writes_first_operand(self) -> bool:
        return False

    def is_control_flow(self) -> bool:
        return True

class BranchNot(GenericInstruction):
    def __init__(self, target: Val, is_scalar: bool = True, predicate: PredReg | None = None):
        super().__init__("bra_n", target, is_scalar=is_scalar, predicate=predicate)
        self.target = target

    def writes_first_operand(self) -> bool:
        return False

    def is_control_flow(self) -> bool:
        return True

class Jump(Branch):
    pass
