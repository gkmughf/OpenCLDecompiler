from typing import Any
from src.ir.instructions.types import IRType
from src.ir.registers.reg import PredReg
from src.ir.kernel_components import (
    KernelArguments,
    KernelLocalMemory,
    KernelPredicates,
    KernelInstructions,
)


class _OperationTypeUnset:
    pass


OP_TYPE_UNSET = _OperationTypeUnset()
OperationTypeArg = IRType | _OperationTypeUnset


class Kernel:
    def __init__(self, name: str, work_group_size: list[int]):
        self.name = name
        self.work_group_size = work_group_size

        self._instructions = KernelInstructions()
        self._arguments = KernelArguments()
        self._local_memory = KernelLocalMemory()
        self._predicates = KernelPredicates()


    @property
    def instructions(self) -> KernelInstructions:
        return self._instructions

    @property
    def arguments(self) -> KernelArguments:
        return self._arguments

    @property
    def local_memory(self) -> KernelLocalMemory:
        return self._local_memory
    
    @property
    def predicates(self) -> KernelPredicates:
        return self._predicates

    def create_instruction(
           self,
           instruction_class: type,
           *args: Any,
           predicate: str | PredReg | None = None,
           predicate_negated: bool = False,
           op_type: OperationTypeArg = OP_TYPE_UNSET,
        ) -> "Kernel":
           if op_type is OP_TYPE_UNSET:
               instruction = instruction_class(*args)
           elif isinstance(op_type, IRType):
               instruction = instruction_class(*args, op_type=op_type)
           else:
               raise TypeError("op_type must be an IRType")
           predicate_reg = self._coerce_predicate(predicate)
           if predicate_reg is not None:
               instruction.set_predicate(predicate_reg, predicate_negated)

           self.instructions.append(instruction)
           return self

    @staticmethod
    def _coerce_predicate(predicate: str | PredReg | None) -> PredReg | None:
        if predicate is None:
            return None
        if isinstance(predicate, PredReg):
            return predicate
        return PredReg(predicate)
