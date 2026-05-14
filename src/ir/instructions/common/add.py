from src.ir.instructions.generic import GenericInstruction
from src.ir.registers.reg import Reg_ty, RegOrVal_ty, get_reg_rang, Val

from src.ir.instructions.lowering import NodeLoweringContext
from src.instructions.vop2.v_add import VAdd
from src.instructions.vop2.v_addc import VAddc


class Add(GenericInstruction):
    def __init__(self, destination: Reg_ty, operand1: RegOrVal_ty, operand2: RegOrVal_ty, is_scalar: bool = False):
        super().__init__("add", destination, operand1, operand2, is_scalar=is_scalar)
        self.destination = destination
        self.operand1 = operand1
        self.operand2 = operand2
        
    def _is_64bit(self) -> bool:
        return self.destination.bit_width == 64
    
    def _get_normalize_opcode(self, is_addc: bool = False) -> str:
        if is_addc:
            return "v_addc_u32"
        return "v_add_u32"
    

    def to_fill_node(self, state, parents):
        ctx = NodeLoweringContext(state, parents)
        if not self._is_64bit():
            return ctx.emit_backend(
                VAdd,
                self._get_normalize_opcode(),
                self.operands,
                "u32",
            )

        return ctx.emit_backend(
            VAdd,
            self._get_normalize_opcode(),
            [self.destination, self.operand1, self.operand2],
            "u64",
        )



class AddC(GenericInstruction):
    def __init__(self, destination: Reg_ty, operand1: RegOrVal_ty, operand2: RegOrVal_ty, is_scalar: bool = False):
        super().__init__("addc", destination, operand1, operand2, is_scalar=is_scalar)
        self.destination = destination
        self.operand1 = operand1
        self.operand2 = operand2
        

    def _get_normalize_opcode(self) -> str:
        return "v_addc_u32"
    
    def to_fill_node(self, state, parents):
        return NodeLoweringContext(state, parents).emit_backend(
            VAddc,
            self._get_normalize_opcode(),
            self.operands,
            "u32",
        )


class AddF(GenericInstruction):
    def __init__(self, destination: Reg_ty, operand1: RegOrVal_ty, operand2: RegOrVal_ty, is_scalar: bool = False):
        super().__init__("addc", destination, operand1, operand2, is_scalar=is_scalar)
        self.destination = destination
        self.operand1 = operand1
        self.operand2 = operand2
        

    def _get_normalize_opcode(self) -> str:
        return "v_add_f64"
    
    def to_fill_node(self, state, parents):
        return NodeLoweringContext(state, parents).emit_backend(
            VAdd,
            self._get_normalize_opcode(),
            self.operands,
            "f64",
        )