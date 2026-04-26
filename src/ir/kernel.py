from typing import Any, Optional
from src.ir.instructions.generic import GenericInstruction
from src.model.config_data import KernelArgument
from src.opencl_types import evaluate_size, make_asm_type
from src.ir.registers.register_manager import RegisterManager
from src.ir.instructions.special.local_memory import LocalMemory
from src.ir.registers.reg import Reg64, Val, PredReg
from src.ir.blocks import KernelBlock
from src.ir.instructions.special.memory import Store as ArgumentStore



class Kernel:
    def __init__(self, name: str, work_group_size: list[int]):
        self.name = name
        self.work_group_size = work_group_size
        self.instructions: list[GenericInstruction] = []

        self._register_manager: Optional[RegisterManager] = None

        self._blocks: dict[int, KernelBlock] = {}

        self._arg_ptr: Reg64 | None = None
        self._arguments: list[KernelArgument] = []
        self._argument_stores_materialized = False
        self._arguments_by_name: dict[str, KernelArgument] = {}
        self._arguments_by_offset: dict[int, KernelArgument] = {}

        self._local_mem: dict[str, int] = {}
        self._local_memories_materialized = False

    def get_instructions(self) -> list[GenericInstruction]:
        return self.instructions

    def add_argument(self, name: str, type_name: str, const: bool = False, offset: int = 0, hidden: bool = False):
        argument = KernelArgument(
                type_name=type_name,
                name=name,
                offset=offset,
                size=self._argument_size(name, type_name),
                hidden=hidden,
                const=const,
            )
        self._arguments.append(argument)
        self._arguments_by_name[name] = argument
        self._arguments_by_offset[offset] = argument
        return self
    
    def close(self) -> "Kernel":
        self._invalidate_analyses()
        return self

    def set_local_memory(self, name: str, size: int):
        self._local_mem[name] = size

    def update_argument_type(self, arg_name: str, type_name: str) -> bool:
        argument = self._arguments_by_name.get(arg_name)
        if argument is None:
            return False

        argument.type_name = type_name
        argument.size = self._argument_size(argument.name, type_name)

        self._register_manager = None
        return True

    def update_argument_type_by_offset(self, offset: int, type_name: str) -> bool:
        argument = self._arguments_by_offset.get(offset)
        if argument is None:
            return False
        return self.update_argument_type(argument.name, type_name)

    def get_argument_by_offset(self, offset: int) -> KernelArgument | None:
        return self._arguments_by_offset.get(offset)
    
    def get_arguments(self) -> list[KernelArgument]:
        return self._arguments

    
    def create_instruction(
        self,
        instruction_class: type,
        *args: Any,
        is_scalar: bool = False,
        predicate: str | PredReg | None = None,
    ) -> "Kernel":
        self._register_manager = None
        self._blocks = {}

        instruction = instruction_class(*args, is_scalar=is_scalar)
        predicate_reg = self._coerce_predicate(predicate)
        if predicate_reg is not None:
            instruction.set_predicate(predicate_reg)

        self.instructions.append(instruction)
        return self
    

    def update_argument_type(self, arg_name: str, type_name: str) -> bool:
        argument = self._arguments_by_name.get(arg_name)
        if argument is None:
            return False

        argument.type_name = type_name
        argument.size = self._argument_size(argument.name, type_name)

        self._register_manager = None
        return True

    def update_argument_type_by_offset(self, offset: int, type_name: str) -> bool:
        argument = self._arguments_by_offset.get(offset)
        if argument is None:
            return False
        return self.update_argument_type(argument.name, type_name)

    def get_argument_by_offset(self, offset: int) -> KernelArgument | None:
        return self._arguments_by_offset.get(offset)

    def prepend_instructions(self, instructions: list[GenericInstruction]) -> None:
        if not instructions:
            return
        self.instructions = instructions + self.instructions
        self._invalidate_analyses()

    def set_blocks(self, blocks: dict[int, KernelBlock]) -> None:
        self._blocks = blocks

    def get_blocks(self) -> dict[int, KernelBlock]:
        return self._blocks

    def set_register_manager(self, register_manager: RegisterManager) -> None:
        self._register_manager = register_manager

    def get_register_manager(self) -> Optional[RegisterManager]:
        return self._register_manager
    
    def mark_local_memories_materialized(self) -> None:
        self._local_memories_materialized = True

    def are_argument_stores_materialized(self) -> bool:
        return self._argument_stores_materialized
    
    def mark_argument_stores_materialized(self) -> None:
        self._argument_stores_materialized = True


    def get_instructions_parts(self) -> list[list[str]]:
        result = []
        for instr in self.instructions:
            result.extend(instr.get_parts())
        return result

    def get_local_memories(self) -> dict[str, int]:
        return dict(self._local_mem)
    
    def are_local_memories_materialized(self) -> bool:
        return self._local_memories_materialized
    
    def set_arg_ptr(self, arg_ptr: Reg64) -> "Kernel":
        self._arg_ptr = arg_ptr
        self._argument_stores_materialized = False
        self._invalidate_analyses()
        return self

    def get_arg_ptr(self) -> Reg64 | None:
        return self._arg_ptr

    def get_normalize_instructions_parts(self) -> list[list[str]]:
        self._ensure_registers_normalized()
        assert self._register_manager is not None

        result = []
        for instr in self.instructions:
            result.extend(instr.get_parts(self._register_manager))
        return result

    def get_instructions(self) -> list[str]:
        return [instr.to_text() for instr in self.instructions]

    def to_text(self) -> str:
        lines = [
            f".WGS [{', '.join(map(str, self.work_group_size))}]",
            f"define kernel {self.name} (",
        ]

        for i, arg in enumerate(self.arguments):
            const_str = "const " if arg.const else ""
            suffix = "," if i < len(self.arguments) - 1 else ""
            lines.append(f"    {const_str}{arg.type_name} {arg.name}{suffix}")

        lines.append(")")
        lines.append("{")

        for instr in self.instructions:
            lines.append(f"  {instr.to_text()}")

        lines.append("}")

        return "\n".join(lines)


    def _ensure_registers_normalized(self) -> None:
        if self._register_manager is not None:
            return

        from src.ir.passes.base import PassContext
        from src.ir.passes.normalize_registers import NormalizeRegistersPass

        NormalizeRegistersPass().run(self, PassContext())

    @staticmethod
    def _coerce_predicate(predicate: str | PredReg | None) -> PredReg | None:
        if predicate is None:
            return None
        if isinstance(predicate, PredReg):
            return predicate
        return PredReg(predicate)
    
    @staticmethod
    def _argument_size(name: str, type_name: str) -> int:
        if name.startswith("*"):
            return 8

        size, _ = evaluate_size(make_asm_type(type_name))
        return size

    def _invalidate_analyses(self) -> None:
        self._register_manager = None
        self._blocks = {}
