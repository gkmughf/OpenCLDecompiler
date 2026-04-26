from __future__ import annotations

from src.ir.instructions.common.load import Load
from src.ir.instructions.common.typed_memory import TypedMemoryLoad, TypedMemoryStore
from src.ir.passes.base import KernelPass, PassContext
from src.ir.passes.register_flow import BuildRegisterFlowPass, RegisterFlowGraph
from src.ir.registers.reg import BaseReg
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ir.kernel import Kernel

class InferPTXPointerTypesPass(KernelPass):
    name = "infer-ptx-pointer-types"

    def run(self, kernel, context: PassContext) -> None:
        flow = context.metadata.get("register_flow")
        if not isinstance(flow, RegisterFlowGraph):
            BuildRegisterFlowPass().run(kernel, context)
            flow = context.metadata["register_flow"]

        inferred_types = context.metadata.setdefault("ptx_inferred_pointer_arg_types", {})
        inferred_count = 0
        for index, instruction in enumerate(kernel.instructions):
            if isinstance(instruction, TypedMemoryLoad):
                if self._propagate_pointer_type(
                    kernel,
                    flow,
                    instruction.address,
                    instruction.access_type.to_opencl_type(),
                    index,
                    inferred_types,
                    set(),
                ):
                    inferred_count += 1
                continue

            if isinstance(instruction, TypedMemoryStore):
                if self._propagate_pointer_type(
                    kernel,
                    flow,
                    instruction.address,
                    instruction.access_type.to_opencl_type(),
                    index,
                    inferred_types,
                    set(),
                ):
                    inferred_count += 1

        context.metadata["ptx_pointer_types_inferred"] = inferred_count

    def _propagate_pointer_type(
        self,
        kernel: Kernel,
        flow: RegisterFlowGraph,
        register: BaseReg,
        type_name: str,
        use_index: int,
        inferred_types: dict[int, str],
        visited: set[tuple[int, str]],
    ) -> bool:
        visit_key = (use_index, register.name)
        if visit_key in visited:
            return False
        visited.add(visit_key)

        writer_index = flow.get_writer(use_index, register.name)
        if writer_index is None:
            return False

        writer = kernel.instructions[writer_index]
        if self._try_update_argument_from_load(kernel, writer, type_name, inferred_types):
            return True

        updated = False
        for source in writer.get_read_registers():
            updated = (
                self._propagate_pointer_type(
                    kernel,
                    flow,
                    source,
                    type_name,
                    writer_index,
                    inferred_types,
                    visited,
                )
                or updated
            )

        return updated

    def _try_update_argument_from_load(
        self,
        kernel: Kernel,
        instruction,
        type_name: str,
        inferred_types: dict[int, str],
    ) -> bool:
        if not isinstance(instruction, Load):
            return False

        if instruction.address.name != kernel.get_arg_ptr().name:
            return False

        if instruction.size != 64:
            return False

        offset = self._parse_int_value(instruction.offset.value)
        if offset is None:
            return False

        argument = kernel.get_argument_by_offset(offset)
        if argument is None or not argument.name.startswith("*"):
            return False

        inferred_type = inferred_types.get(offset)
        if inferred_type is not None:
            return inferred_type == type_name

        if not kernel.update_argument_type_by_offset(offset, type_name):
            return False

        inferred_types[offset] = type_name
        return True

    @staticmethod
    def _parse_int_value(value: str) -> int | None:
        try:
            return int(value, 0)
        except ValueError:
            return None
