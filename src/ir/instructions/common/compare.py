from src.ir.instructions.generic import GenericInstruction
from src.ir.registers.reg import PredReg, RegOrVal_ty


class BaseCompare(GenericInstruction):
    comparison = ""

    def __init__(
        self,
        destination: PredReg,
        operand1: RegOrVal_ty,
        operand2: RegOrVal_ty,
        is_scalar: bool = False,
    ):
        super().__init__(f"cmp.{self.comparison}", destination, operand1, operand2, is_scalar=is_scalar)
        self.destination = destination
        self.operand1 = operand1
        self.operand2 = operand2

    def _get_normalize_opcode(self) -> str:
        return f'v_cmp_{self.comparison}_i32'

    def get_parts(self) -> list[list[str]]:
        opcode = self._get_normalize_opcode()
        operand1 = self.operand1.name
        operand2 = self.operand2.name

        if self.is_scalar():
            return self._format_parts([[opcode, operand1, operand2]])

        destination = self.destination.name
        return [[opcode, destination, operand1, operand2]]


class CompareEq(BaseCompare):
    comparison = "eq"


class CompareNe(BaseCompare):
    comparison = "ne"


class CompareLt(BaseCompare):
    comparison = "lt"


class CompareLe(BaseCompare):
    comparison = "le"


class CompareGt(BaseCompare):
    comparison = "gt"


class CompareGe(BaseCompare):
    comparison = "ge"


_COMPARE_CLASSES = {
    "eq": CompareEq,
    "ne": CompareNe,
    "lg": CompareNe,
    "lt": CompareLt,
    "le": CompareLe,
    "gt": CompareGt,
    "ge": CompareGe,
}


def get_compare_class(comparison: str) -> type[BaseCompare]:
    try:
        return _COMPARE_CLASSES[comparison]
    except KeyError as exc:
        raise NotImplementedError(f"Unknown compare operation: {comparison}") from exc
