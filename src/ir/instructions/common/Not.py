from src.ir.instructions.generic import GenericInstruction
from src.ir.instructions.lowering import NodeLoweringContext
from src.ir.registers.reg import PredReg
from src.instructions.sop1.s_not import SNot


class Not(GenericInstruction):
    def __init__(self, destination: PredReg, source: PredReg):
        super().__init__("not", destination, source)
        self.destination = destination
        self.source = source

    def _get_normalize_opcode(self) -> str:
        return "s_not_b64"

    def to_fill_node(self, state, parents):
        return NodeLoweringContext(state, parents).emit_backend(
            SNot,
            self._get_normalize_opcode(),
            self.operands,
            "b64",
        )
