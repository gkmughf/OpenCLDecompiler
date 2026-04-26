from src.ir.instructions.generic import GenericInstruction
from src.ir.registers.reg import Reg64, Reg_ty, Val

class MemoryAllocation(GenericInstruction):
    def __init__(self, destination: Reg64, is_scalar: bool = True):
        super().__init__("alloc", destination, is_scalar=is_scalar)
    def _get_normalize_opcode(self) -> str:
        return "s_alloc"

class Store(GenericInstruction):
    def __init__(self, destination: Reg_ty, arg_name: Val, arg_type: Val, offset: Val, is_scalar: bool = True):
        super().__init__("store", destination, arg_type, arg_name, offset, is_scalar=is_scalar)
        self.destination = destination
        self.arg_name = arg_name
        self.arg_type = arg_type
        self.offset = offset

    def _get_normalize_opcode(self) -> str:
        return "s_store"
    
    def set_arg_type(self, type_name: str) -> None:
        self.arg_type = Val(type_name)
        self.operands = (self.destination, self.arg_type, self.arg_name, self.offset)
