from src.ir.instructions.generic import GenericInstruction
from src.ir.registers.reg import Reg_ty, RegOrVal_ty, Reg64, get_reg_rang, expand_register_names

from src.ir.instructions.lowering import NodeLoweringContext
from src.instructions.vop2.v_sub import VSub

class Sub(GenericInstruction):
    def __init__(self, destination: Reg_ty, operand1: RegOrVal_ty, operand2: RegOrVal_ty, is_scalar: bool = False):
        super().__init__("sub", destination, operand1, operand2, is_scalar=is_scalar)
        self.destination = destination
        self.operand1 = operand1
        self.operand2 = operand2
        
    def _is_64bit(self) -> bool:
        return self.destination.bit_width == 64
    
    def _get_normalize_opcode(self, is_subb: bool = False) -> str:
        is_scalar = self.is_scalar()
        if is_subb:
            return "v_subb_u32"
        return "v_sub_u32"

    def get_parts(self) -> list[list[str]]:
        result = []
    
        if not self._is_64bit():
            opcode = self._get_normalize_opcode()
            dest_str = self.destination.name
            op1_str = self.operand1.name
            op2_str = self.operand2.name

            result.append([opcode, dest_str, "vcc", op1_str, op2_str])
        else:
            dest_lo, dest_hi = expand_register_names(self.destination)
            op1_lo, op1_hi = expand_register_names(self.operand1)
            op2_lo, op2_hi = expand_register_names(self.operand2)

            sub_opcode = self._get_normalize_opcode(is_subb=False)
            subb_opcode = self._get_normalize_opcode(is_subb=True)

            line1 = [sub_opcode, dest_lo, "vcc", op1_lo, op2_lo]
            line2 = [subb_opcode, dest_hi, "vcc", op1_hi, op2_hi, "vcc"]
            result.extend([line1, line2])

        return result
    
    def to_fill_node(self, state, parents):
        ctx = NodeLoweringContext(state, parents)
        if not self._is_64bit():
            return ctx.emit_backend(
                VSub,
                self._get_normalize_opcode(),
                self.operands,
                "u32",
            )

        dest_lo, dest_hi = get_reg_rang(self.destination)
        op1_lo, op1_hi = get_reg_rang(self.operand1)
        op2_lo, op2_hi = get_reg_rang(self.operand2)

        ctx.emit_backend(
            VSub,
            "v_sub_u32",
            [dest_lo, op1_lo, op2_lo],
            "u32",
        )
        return ctx.emit_backend(
            VSub,
            "v_subb_u32",
            [dest_hi, op1_hi, op2_hi],
            "u32",
        )

class SubRev(Sub):
    def __init__(self, destination: Reg_ty, operand1: RegOrVal_ty, operand2: RegOrVal_ty, is_scalar):
        super().__init__(destination, operand2, operand1, is_scalar=is_scalar)
