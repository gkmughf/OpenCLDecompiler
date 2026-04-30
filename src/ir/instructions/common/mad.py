from src.ir.registers.reg import RegOrVal_ty, Reg32, CompositeReg, Val, Reg64, expand_register_names
from src.ir.instructions.generic import GenericInstruction
from src.ir.TemporaryVariableAllocator import tva
from typing import Optional


class Mad(GenericInstruction):
    def __init__(self, destination: RegOrVal_ty, operand1: RegOrVal_ty, operand2: RegOrVal_ty, operand3: RegOrVal_ty, signed=True,  is_scalar=False):
        self.operand3_val: Optional[Val] = None
        self.operand3_tmp_reg: Optional[Reg64] = None
        
        if isinstance(operand3, Val):
            tmp_reg_name = tva.generate("mad")
            self.operand3_tmp_reg = Reg64(tmp_reg_name)
            self.operand3_val = operand3

        if destination.bit_width == 32:
            dest_name = tva.generate("mad_dest")
            dest_hi = Reg32(tva.generate("dest_hi"))
            destination = CompositeReg(dest_name, [destination, dest_hi])

        name =  "mad_s" if signed else "mad_u"   
        super().__init__(name, destination, operand1, operand2, operand3, is_scalar=is_scalar)


        self.destination = destination
        self.operand1 = operand1
        self.operand2 = operand2
        self.operand3 = operand3
        self.signed = signed

    def _get_normalize_opcode(self) -> str:
        return "v_mad_i64_i32"  if self.signed else "v_mad_u64_u32" 
    
    def get_operands(self):
        if self.operand3_tmp_reg is not None:
            return super().get_operands() + (self.operand3_tmp_reg,)
        return super().get_operands()
    

    def get_parts(self) -> list[list[str]]:
        result = []

        opcode = self._get_normalize_opcode()
        dest_str = self.destination.name
        op1_str = self.operand1.name
        op2_str = self.operand2.name
        op3_str = self.operand3.name

        if self.operand3_val is not None:
            tmp_reg_str = self.operand3_tmp_reg.name
            val_str = self.operand3_val.name
            tmp_lo, tmp_hi = expand_register_names(self.operand3_tmp_reg)
            result.append(["v_mov_b32", tmp_lo, val_str])
            result.append(["s_mov_b32", tmp_hi, '0'])
            result.append([opcode, dest_str, '0', op1_str, op2_str, tmp_reg_str])   
            return result

        result.append([opcode, dest_str, '0', op1_str, op2_str, op3_str])   

        return result