from __future__ import annotations

import copy

from src.ir.instructions.generic import GenericInstruction
from src.ir.registers.reg import PredReg
from src.ir.instructions.lowering import NodeLoweringContext
from src.instructions.IRspecial.ChangeMask import UseMask


class ChangeMask(GenericInstruction):
    def __init__(self, predicate: PredReg, is_scalar: bool = True):
        super().__init__("change_mask", predicate, is_scalar=is_scalar)
        self.predicate = predicate

    def get_parts(self) -> list[list[str]]:
        return []
        
    def _get_normalize_opcode(self):
        return "change_mask"

    def to_fill_node(self, state, parents):
        return NodeLoweringContext(state, parents).emit_backend(
            UseMask,
            self._get_normalize_opcode(),
            self.operands,
        )
    def writes_first_operand(self) -> bool:
        return False

    def has_side_effects(self) -> bool:
        return True
