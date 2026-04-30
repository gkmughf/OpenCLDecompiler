from __future__ import annotations

from src.ir.instructions.generic import GenericInstruction
from src.ir.registers.reg import PredReg


class InitPredicate(GenericInstruction):
    def __init__(self, predicate: PredReg, is_scalar: bool = True):
        super().__init__("init_pred", predicate, is_scalar=is_scalar)
        self.predicate = predicate
    
    def _get_normalize_opcode(self):
        return "s_ipred"

    def get_written_registers(self) -> tuple[PredReg, ...]:
        return ()

    def writes_first_operand(self) -> bool:
        return False

    def has_side_effects(self) -> bool:
        return True

