from __future__ import annotations

from src.ir.instructions.generic import GenericInstruction
from src.ir.registers.reg import PredReg


class ChangeMask(GenericInstruction):
    def __init__(self, predicate: PredReg, is_scalar: bool = True):
        super().__init__("change_mask", predicate, is_scalar=is_scalar)
        self.predicate = predicate

    def get_parts(self) -> list[list[str]]:
        return [[self._get_normalize_opcode(), "exec", self.predicate.name]]
        
    def _get_normalize_opcode(self):
        return "s_mov_b64"
    
    def get_written_registers(self) -> tuple[PredReg, ...]:
        return ()

    def writes_first_operand(self) -> bool:
        return False

    def has_side_effects(self) -> bool:
        return True

