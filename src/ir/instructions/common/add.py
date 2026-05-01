from src.ir.instructions.generic import GenericInstruction
from src.ir.registers.reg import Reg_ty, RegOrVal_ty, Reg64, Val, Reg32, expand_register_names, get_reg_rang
from typing import Optional
from src.ir.TemporaryVariableAllocator import tva

from src.ir.instructions.lowering import NodeLoweringContext
from src.instructions.vop2.v_add import VAdd
from src.instructions.vop2.v_addc import VAddc


class Add(GenericInstruction):
    def __init__(self, destination: Reg_ty, operand1: RegOrVal_ty, operand2: RegOrVal_ty, is_scalar: bool = False):
        self.operand2_val: Optional[Val] = None
        self.operand2_tmp_reg: Optional[Reg_ty] = None
        
        if isinstance(operand2, Val):
            if destination.bit_width == 64:
                self.operand2_tmp_reg = Reg64(tva.generate("add"))
            else:
                self.operand2_tmp_reg = Reg32(tva.generate("add"))
            
            self.operand2_val = operand2
        
        
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

    def get_operands(self):
        if self.operand2_tmp_reg is not None:
            return super().get_operands() + (self.operand2_tmp_reg,)
        return super().get_operands()

    def get_parts(self) -> list[list[str]]:
        result = []
        operand2 = self.operand2

        if not self._is_64bit():
            opcode = self._get_normalize_opcode()
            dest_str = self.destination.name
            op1_str = self.operand1.name
            op2_str = operand2.name
            
            result.append([opcode, dest_str, "vcc", op1_str, op2_str])
        else:
            dest_lo, dest_hi = expand_register_names(self.destination)
            op1_lo, op1_hi = expand_register_names(self.operand1)
            op2_lo, op2_hi = expand_register_names(operand2)

            add_opcode = self._get_normalize_opcode()
            addc_opcode =  self._get_normalize_opcode(is_addc=True)

            line1 = [add_opcode, dest_lo, "vcc", op1_lo, op2_lo]
            line2 = [addc_opcode, dest_hi, "vcc", op1_hi, op2_hi, "vcc"]

            result.extend([line1, line2])


        return result
    

    def to_fill_node(self, state, parents):
        ctx = NodeLoweringContext(state, parents)
        if not self._is_64bit():
            return ctx.emit_backend(
                VAdd,
                self._get_normalize_opcode(),
                self.operands,
                "u32",
            )

        dest_lo, dest_hi = get_reg_rang(self.destination)
        op1_lo, op1_hi = get_reg_rang(self.operand1)
        op2_lo, op2_hi = get_reg_rang(self.operand2)

        ctx.emit_backend(
            VAdd,
            self._get_normalize_opcode(),
            [dest_lo, op1_lo, op2_lo],
            "u32",
        )
        return ctx.emit_backend(
            VAddc,
            self._get_normalize_opcode(is_addc=True),
            [dest_hi, op1_hi, op2_hi],
            "u32",
        )



class AddC(GenericInstruction):
    def __init__(self, destination: Reg_ty, operand1: RegOrVal_ty, operand2: RegOrVal_ty, is_scalar: bool = False):
        super().__init__("addc", destination, operand1, operand2, is_scalar=is_scalar)
        self.destination = destination
        self.operand1 = operand1
        self.operand2 = operand2
        

    def _get_normalize_opcode(self) -> str:
        return "v_addc_u32"

    def get_parts(self) -> list[list[str]]:
        opcode = self._get_normalize_opcode()
        dest_str = self.destination.name
        op1_str = self.operand1.name
        op2_str = self.operand2.name
        
        return [[opcode, dest_str, "vcc", op1_str, op2_str, "vcc"]]
    
    def to_fill_node(self, state, parents):
        return NodeLoweringContext(state, parents).emit_backend(
            VAddc,
            self._get_normalize_opcode(),
            self.operands,
            "u32",
        )

        