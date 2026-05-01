from src.ir.instructions.lowering import NodeLoweringContext
from src.instructions.smem.s_load import SLoad
from src.instructions.flat.flat_load import FlatLoad

from src.ir.registers.reg import Reg64, Reg_ty, Val, expand_register_names
from src.ir.instructions.generic import GenericInstruction

class Load(GenericInstruction):
    def __init__(self, destination: Reg_ty, address: Reg64, offset: Val, is_scalar, size):
        super().__init__("load", destination, address, offset, is_scalar=is_scalar)
        self.destination = destination
        self.address = address
        self.offset = offset if offset != None else Val("0")
        self.size = size

    def _get_normalize_opcode(self) -> str:
        prefix = "s_load" if self.is_scalar() else "flat_load"

        if self.size<= 32:
            return f"{prefix}_dword"
        elif self.size <= 64:
            return f"{prefix}_dwordx2"
        else:
            return f"{prefix}_dwordx4"

    def get_parts(self) -> list[list[str]]:
        result = []
        opcode = self._get_normalize_opcode()
        dest_str = self.destination.name
        addr_str = self.address.name
        offset_str =  self.offset.name

        # all_parts = expand_register_names(self.destination)
        # total_parts = len(all_parts)
        
        # parts_to_load = (self.size) // 32
        
        # if total_parts > parts_to_load:
        #     for i in range(parts_to_load, total_parts):
        #         result.append(["v_mov_b32", all_parts[i], "0"])
        
        result.append([opcode, dest_str, addr_str, offset_str])
        return result
    
    def get_instruction(self):
        pass

    def get_suffix(self):
        if self.size<= 32:
            return "dword"
        elif self.size <= 64:
            return "dwordx2"
        else:
            return "dwordx4"

    def to_fill_node(self, state, parents):
        ctx = NodeLoweringContext(state, parents)
        if self.is_scalar():
            return ctx.emit_backend(
                SLoad,
                self._get_normalize_opcode(),
                self.operands,
                self.get_suffix(),
            )
        return ctx.emit_backend(
            FlatLoad,
            self._get_normalize_opcode(),
            self.operands,
            self.get_suffix(),
        )
                
class Load32(Load):
    def __init__(self, destination: Reg_ty, address: Reg64, offset: Val | None = None, is_scalar: bool = False):
        super().__init__(destination, address, offset, is_scalar=True, size=32)

class Load64(Load):
    def __init__(self, destination: Reg_ty, address: Reg64, offset: Val | None = None, is_scalar: bool = False):
        super().__init__(destination, address, offset, is_scalar=True, size=64)

class Load128(Load):
    def __init__(self, destination: Reg_ty, address: Reg64, offset: Val | None = None, is_scalar: bool = False):
        super().__init__(destination, address, offset, is_scalar=True, size=128)


class FLoad32(Load):
    def __init__(self, destination: Reg_ty, address: Reg64, offset: Val | None = None, is_scalar: bool = True):
        super().__init__(destination, address, offset, is_scalar=False, size=32)

class FLoad64(Load):
    def __init__(self, destination: Reg_ty, address: Reg64, offset: Val | None = None, is_scalar: bool = True):
        super().__init__(destination, address, offset, is_scalar=False, size=64)

class FLoad128(Load):
    def __init__(self, destination: Reg_ty, address: Reg64, offset: Val | None = None, is_scalar: bool = True):
        super().__init__(destination, address, offset, is_scalar=False, size=128)