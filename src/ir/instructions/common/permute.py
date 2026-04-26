from src.ir.instructions.generic import GenericInstruction
from src.ir.registers.reg import RegOrVal_ty, Reg_ty


class Permute32(GenericInstruction):
    def __init__(
        self,
        destination: Reg_ty,
        operand1: RegOrVal_ty,
        operand2: RegOrVal_ty,
        selector: RegOrVal_ty,
        is_scalar: bool = False,
    ):
        super().__init__("permute32", destination, operand1, operand2, selector, is_scalar=is_scalar)

    def _get_normalize_opcode(self) -> str:
        return "v_perm_b32"
