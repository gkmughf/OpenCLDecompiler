from src.ir.instructions.generic import GenericInstruction
from src.ir.instructions.lowering import NodeLoweringContext
from src.ir.registers.reg import Reg64, RegOrVal_ty
from src.instructions.flat.flat_store import FlatStore

_SUFFIX_BY_SIZE = {
    8: "byte",
    16: "short",
    32: "dword",
    64: "dwordx2",
    128: "dwordx4",
}


class GenericStore(GenericInstruction):
    operation: str
    backend_instruction: type

    def __init__(self, address: Reg64, value: RegOrVal_ty, is_scalar, size):
        super().__init__("store", address, value, is_scalar=is_scalar)
        self.address = address
        self.value = value
        self.size = size

    def _get_normalize_opcode(self) -> str:
        return f"{self.operation}_{self.get_suffix()}"

    def _get_opcode(self):
        return self.backend_instruction

    def writes_first_operand(self) -> bool:
        return False

    def get_suffix(self):
        return _SUFFIX_BY_SIZE.get(self.size)

    def to_fill_node(self, state, parents):
        return NodeLoweringContext(state, parents).emit_backend(
            self._get_opcode(),
            self._get_normalize_opcode(),
            self.operands,
            self.get_suffix(),
        )


class Store(GenericStore):
    operation: str = "flat_store"
    backend_instruction: type = FlatStore


class Store8(Store):
    def __init__(self, address: Reg64, value: RegOrVal_ty, is_scalar):
        del is_scalar
        super().__init__(address, value, is_scalar=True, size=8)


class Store16(Store):
    def __init__(self, address: Reg64, value: RegOrVal_ty, is_scalar):
        del is_scalar
        super().__init__(address, value, is_scalar=True, size=16)


class Store32(Store):
    def __init__(self, address: Reg64, value: RegOrVal_ty, is_scalar):
        del is_scalar
        super().__init__(address, value, is_scalar=True, size=32)


class Store64(Store):
    def __init__(self, address: Reg64, value: RegOrVal_ty, is_scalar):
        del is_scalar
        super().__init__(address, value, is_scalar=True, size=64)


class Store128(Store):
    def __init__(self, address: Reg64, value: RegOrVal_ty, is_scalar):
        del is_scalar
        super().__init__(address, value, is_scalar=True, size=128)


class FStore(GenericStore):
    operation: str = "flat_store"
    backend_instruction: type = FlatStore


class FStore8(FStore):
    def __init__(self, address: Reg64, value: RegOrVal_ty, is_scalar):
        del is_scalar
        super().__init__(address, value, is_scalar=False, size=8)


class FStore16(FStore):
    def __init__(self, address: Reg64, value: RegOrVal_ty, is_scalar):
        del is_scalar
        super().__init__(address, value, is_scalar=False, size=16)


class FStore32(FStore):
    def __init__(self, address: Reg64, value: RegOrVal_ty, is_scalar):
        del is_scalar
        super().__init__(address, value, is_scalar=False, size=32)


class FStore64(FStore):
    def __init__(self, address: Reg64, value: RegOrVal_ty, is_scalar):
        del is_scalar
        super().__init__(address, value, is_scalar=False, size=64)


class FStore128(FStore):
    def __init__(self, address: Reg64, value: RegOrVal_ty, is_scalar):
        del is_scalar
        super().__init__(address, value, is_scalar=False, size=128)
