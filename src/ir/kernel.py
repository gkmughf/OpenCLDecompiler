from typing import Any
from src.ir.instructions.generic import GenericInstruction
from src.ir.registers.reg import PredReg
from src.ir.blocks import KernelBlocks
from src.ir.kernel_components import (
    KernelArguments,
    KernelLocalMemory,
    KernelPredicates,
)


class Kernel:
    def __init__(self, name: str, work_group_size: list[int]):
        self.name = name
        self.work_group_size = work_group_size

        self._blocks = KernelBlocks()
        self._arguments = KernelArguments()
        self._local_memory = KernelLocalMemory()
        self._predicates = KernelPredicates()


    @property
    def blocks(self) -> KernelBlocks:
        return self._blocks
    
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
           is_scalar: bool = False,
           predicate: str | PredReg | None = None,
        ) -> "Kernel":
           instruction = instruction_class(*args, is_scalar=is_scalar)
           predicate_reg = self._coerce_predicate(predicate)
           if predicate_reg is not None:
               instruction.set_predicate(predicate_reg)

           self.blocks.append_instruction(instruction)
           return self


    def prepend_instructions(self, instructions: list[GenericInstruction]) -> None:
        self.blocks.prepend_instructions(instructions)

    def append_instructions(self, instructions: list[GenericInstruction]) -> None:
        self.blocks.append_instructions(instructions)
   
    def get_instructions_parts(self) -> list[list[str]]:
        return self.blocks.instruction_parts()

    def get_instructions(self) -> list[GenericInstruction]:
        return self.blocks.get_instructions()

    @staticmethod
    def _coerce_predicate(predicate: str | PredReg | None) -> PredReg | None:
        if predicate is None:
            return None
        if isinstance(predicate, PredReg):
            return predicate
        return PredReg(predicate)
