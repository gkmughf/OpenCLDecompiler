from src.ir.instructions.lowering import NodeLoweringContext
from src.instructions.smem.s_load import SLoad
from src.instructions.flat.flat_load import FlatLoad

from src.ir.registers.reg import Reg64, Reg_ty, Val
from src.ir.instructions.generic import GenericInstruction

_SUFFIX_BY_SIZE = {
    32: "dword",
    64: "dwordx2",
    128: "dwordx4",
}

class GenericLoad(GenericInstruction):
    operation: str
    backend_instruction: type
    
    def __init__(self, destination: Reg_ty, address: Reg64, offset: Val, size):
        self.destination = destination
        self.address = address
        self.offset = offset if offset != None else Val("0")
        super().__init__("load", self.destination, self.address, self.offset)
        self.size = size

    def _get_normalize_opcode(self) -> str:
        return f"{self.operation}_{self.get_suffix()}"

        
    def _get_opcode(self):
        return self.backend_instruction

    def get_suffix(self):
        return _SUFFIX_BY_SIZE.get(self.size)


    def to_fill_node(self, state, parents):
        ctx = NodeLoweringContext(state, parents)
        return ctx.emit_backend(
            self._get_opcode(),
            self._get_normalize_opcode(),
            self.operands,
            self.get_suffix(),
        )

class Load(GenericLoad):
    operation: str = "s_load"
    backend_instruction: type = SLoad

class Load32(Load):
    def __init__(self, destination: Reg_ty, address: Reg64, offset: Val | None = None):
        super().__init__(destination, address, offset, size=32)

class Load64(Load):
    def __init__(self, destination: Reg_ty, address: Reg64, offset: Val | None = None):
        super().__init__(destination, address, offset, size=64)

class Load128(Load):
    def __init__(self, destination: Reg_ty, address: Reg64, offset: Val | None = None):
        super().__init__(destination, address, offset, size=128)


class FLoad(GenericLoad):
    operation: str = "flat_load"
    backend_instruction: type = FlatLoad

class FLoad32(FLoad):
    def __init__(self, destination: Reg_ty, address: Reg64, offset: Val | None = None):
        super().__init__(destination, address, offset, size=32)

class FLoad64(FLoad):
    def __init__(self, destination: Reg_ty, address: Reg64, offset: Val | None = None):
        super().__init__(destination, address, offset, size=64)

class FLoad128(FLoad):
    def __init__(self, destination: Reg_ty, address: Reg64, offset: Val | None = None):
        super().__init__(destination, address, offset, size=128)
