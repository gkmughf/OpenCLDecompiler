from src.ir.registers.reg import Reg_ty, RegOrVal_ty, Val
from src.ir.instructions.generic import GenericInstruction

class And(GenericInstruction):
    operation = "and"
    def __init__(self, destination: Reg_ty, operand1: Reg_ty, operand2: RegOrVal_ty, is_scalar):
        super().__init__(self.operation, destination, operand1, operand2, is_scalar=is_scalar)
        self.destination = destination
        if isinstance(operand2, Val) and not is_scalar:
            self.operand1 = operand2
            self.operand2 = operand1
        else:
            self.operand1 = operand1
            self.operand2 = operand2
        

    def _is_64bit(self) -> bool:
        return self.destination.bit_width == 64

    def _get_normalize_opcode(self) -> str:
        if self._is_64bit():
            return f's_{self.operation}_b64' if self.is_scalar() else f'v_{self.operation}_b64'
        return f's_{self.operation}_b32' if self.is_scalar() else f'v_{self.operation}_b32'
        
    def _is_numeric_val(self, val: Val) -> bool:
        if not isinstance(val, Val):
            return False
        
        value = val.value.strip()
        if not value:
            return False
        
        if value.startswith(('0x', '0X')):
            try:
                int(value, 16)
                return True
            except ValueError:
                return False
            
        try:
            int(value, 10)
            return True
        except ValueError:
            return False

    def _split_val_to_64bit(self, val: Val) -> tuple[str, str]:
        num_value = int(val.value, 0)

        lo = num_value & 0xFFFFFFFF
        hi = (num_value >> 32) & 0xFFFFFFFF
        
        return (f"{lo}", f"{hi}")


    def get_parts(self) -> list[list[str]]:
        result = []
        opcode = self._get_normalize_opcode()
        
        if not self._is_64bit():
            dest_str = self.destination.name
            op1_str = self.operand1.name
            op2_str = self.operand2.name
            if self._is_numeric_val(self.operand1):
                op1_str = self.operand1.value
            result.append([opcode, dest_str, op1_str, op2_str])
        else:
            dest_lo = self.destination.name
            op2_lo = self.operand2.name
            
            op1_str = self.operand1.name
            
            result.append([opcode, dest_lo, op1_str, op2_lo])
        
        return result