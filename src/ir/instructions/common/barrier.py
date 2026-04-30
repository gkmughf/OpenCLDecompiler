from src.ir.instructions.generic import GenericInstruction

class Barrier(GenericInstruction):
    def __init__(self, is_scalar: bool):
        super().__init__("bar", is_scalar=is_scalar)
   
    def _get_normalize_opcode(self) -> str:
        return "s_waitcnt"
    
    def get_parts(self) -> list[list[str]]:
        return [[self._get_normalize_opcode(), "lgkmcnt(0)"]]
