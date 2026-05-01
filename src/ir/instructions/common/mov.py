from src.ir.registers.reg import Reg_ty, RegOrVal_ty, Val, expand_register_names
from src.ir.instructions.generic import GenericInstruction

from src.ir.instructions.lowering import NodeLoweringContext
from src.instructions.sop1.s_mov import SMov


class Mov(GenericInstruction):
    def __init__(self, destination: Reg_ty, source: RegOrVal_ty, is_scalar):
        super().__init__("mov", destination, source, is_scalar=is_scalar)
        self.destination = destination
        self.source = source

    def _is_64bit(self) -> bool:
        return self.destination.bit_width == 64

    def _get_normalize_opcode(self) -> str:
        return "s_mov_b64" if self._is_64bit() else "s_mov_b32"

    def get_parts(self) -> list[list[str]]:
        result = []


        if self._is_64bit() and isinstance(self.source, Val):
            dest_lo, dest_hi = expand_register_names(self.destination)
            val_str = self.source.name
            result.append(["v_mov_b32", dest_lo, val_str])
            result.append(["v_mov_b32", dest_hi, "0"])
            
            return result
        
        # NOTE забавный факт, если делать так то тесты sub показывают более правильный результат(вместо "a + -1" будет "a - 1")
        # elif self._is_64bit():
        #     dest_lo, dest_hi = split_range(manager.map(self.destination))
        #     src_lo, src_hi = split_range(manager.map(self.source))
        #     result.append(["v_mov_b32", dest_lo, src_lo])
        #     result.append(["v_mov_b32", dest_hi, src_hi])
            
        #     return result
        
        
        opcode = self._get_normalize_opcode()
        dest_str = self.destination.name
        src_str = self.source.name

        result.append([opcode, dest_str, src_str])
        return result
        
    def get_suffix(self):
        if self._is_64bit():
            return "b64"
        return "b32"

    def to_fill_node(self, state, parents):
        return NodeLoweringContext(state, parents).emit_backend(
            SMov,
            self._get_normalize_opcode(),
            self.operands,
            self.get_suffix(),
        )

class MovK(GenericInstruction):
    def __init__(self, destination: Reg_ty, source: RegOrVal_ty, is_scalar):
        super().__init__("mov", destination, source, is_scalar=is_scalar)
        self.destination = destination
        self.source = source

    def _is_64bit(self) -> bool:
        return False

    def _get_normalize_opcode(self) -> str:
        return "s_movk_i32"
