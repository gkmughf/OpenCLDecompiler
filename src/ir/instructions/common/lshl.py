from src.ir.registers.reg import Reg_ty, RegOrVal_ty
from src.ir.instructions.generic import GenericInstruction


class ShiftInstruction(GenericInstruction):
    def __init__(self, name: str, destination: Reg_ty, operand1: Reg_ty, operand2: RegOrVal_ty, is_scalar):
        super().__init__(name, destination, operand1, operand2, is_scalar=is_scalar)
        self.destination = destination
        self.operand1 = operand1
        self.operand2 = operand2

    def _is_64bit(self) -> bool:
        return self.destination.bit_width == 64

    def _get_normalize_opcode(self) -> str:
        raise NotImplementedError
    
    def get_parts(self) -> list[list[str]]:
        result = []
        opcode = self._get_normalize_opcode()
        dest_str = self.destination.name
        op1_str = self.operand1.name
        op2_str =self.operand2.name
        
        if self.is_scalar():
            result.append([opcode, dest_str, op1_str, op2_str])
        else:
            result.append([opcode, dest_str, op2_str, op1_str])
        return result



class LShl(ShiftInstruction):
    def __init__(self, destination: Reg_ty, operand1: Reg_ty, operand2: RegOrVal_ty, is_scalar):
        super().__init__("lshl", destination, operand1, operand2, is_scalar=is_scalar)

    def _get_normalize_opcode(self) -> str:
        if self._is_64bit():
            return "s_lshl_b64" if self.is_scalar() else "v_lshlrev_b64"
        return "s_lshl_b32" if self.is_scalar() else "v_lshlrev_b32"
    

class LShr(ShiftInstruction):
    def __init__(self, destination: Reg_ty, operand1: Reg_ty, operand2: RegOrVal_ty, is_scalar):
        super().__init__("rshl", destination, operand1, operand2, is_scalar=is_scalar)


    def _get_normalize_opcode(self) -> str:
        if self._is_64bit():
            return "s_lshr_b64" if self.is_scalar() else "v_lshrrev_b64"
        return "s_lshr_b32" if self.is_scalar() else "v_lshrrev_b32"
        

class AShr(ShiftInstruction):
    def __init__(self, destination: Reg_ty, operand1: Reg_ty, operand2: RegOrVal_ty, is_scalar):
        super().__init__("rshl", destination, operand1, operand2, is_scalar=is_scalar)

    def _get_normalize_opcode(self) -> str:
        if self._is_64bit():
            return "s_ashr_i64" if self.is_scalar() else "v_ashrrev_i64"
        else:
            return "s_ashr_i32" if self.is_scalar() else "v_ashrrev_i32"
 

class LShl_Rev(LShl):
    def __init__(self, destination: Reg_ty, operand1: Reg_ty, operand2: RegOrVal_ty, is_scalar):
        super().__init__(destination, operand2, operand1, is_scalar=is_scalar)

class LShr_Rev(LShr):
    def __init__(self, destination: Reg_ty, operand1: Reg_ty, operand2: RegOrVal_ty, is_scalar):
        super().__init__(destination, operand2, operand1, is_scalar=is_scalar)

class AShr_Rev(AShr):
    def __init__(self, destination: Reg_ty, operand1: Reg_ty, operand2: RegOrVal_ty, is_scalar):
        super().__init__(destination, operand2, operand1, is_scalar=is_scalar)