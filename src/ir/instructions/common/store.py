from src.ir.registers.reg import Reg64, RegOrVal_ty
from src.ir.instructions.generic import GenericInstruction

class Store(GenericInstruction):
    def __init__(self, address: Reg64, value: RegOrVal_ty, is_scalar, size):
        super().__init__("store", address, value, is_scalar=is_scalar)
        self.address = address
        self.value = value
        self.size = size

    def _get_normalize_opcode(self) -> str:
        prefix = "s_store" if self.is_scalar() else "flat_store"
        if self.size == 8:
            return f"{prefix}_byte"
        if self.size == 16:
            return f"{prefix}_short"
        elif self.size == 32:
            return f"{prefix}_dword"
        elif self.size == 64:
            return f"{prefix}_dwordx2"
        else:
            return f"{prefix}_dwordx4"
        
    def writes_first_operand(self) -> bool:
        return False  

class Store8(Store):
    def __init__(self, address: Reg64, value: RegOrVal_ty, is_scalar):
        super().__init__(address, value, is_scalar=True, size=8)

class Store16(Store):
    def __init__(self, address: Reg64, value: RegOrVal_ty, is_scalar):
        super().__init__(address, value, is_scalar=True, size=16)

class Store32(Store):
    def __init__(self, address: Reg64, value: RegOrVal_ty, is_scalar):
        super().__init__(address, value, is_scalar=True, size=32)

class Store64(Store):
    def __init__(self, address: Reg64, value: RegOrVal_ty, is_scalar):
        super().__init__(address, value, is_scalar=True, size=64)

class Store128(Store):
    def __init__(self, address: Reg64, value: RegOrVal_ty, is_scalar):
        super().__init__(address, value, is_scalar=True, size=128)


class FStore8(Store):
    def __init__(self, address: Reg64, value: RegOrVal_ty, is_scalar):
        super().__init__(address, value, is_scalar=False, size=8)

class FStore16(Store):
    def __init__(self, address: Reg64, value: RegOrVal_ty, is_scalar):
        super().__init__(address, value, is_scalar=False, size=16)

class FStore32(Store):
    def __init__(self, address: Reg64, value: RegOrVal_ty, is_scalar):
        super().__init__(address, value, is_scalar=False, size=32)

class FStore64(Store):
    def __init__(self, address: Reg64, value: RegOrVal_ty, is_scalar):
        super().__init__(address, value, is_scalar=False, size=64)

class FStore128(Store):
    def __init__(self, address: Reg64, value: RegOrVal_ty, is_scalar):
        super().__init__(address, value, is_scalar=False, size=128)