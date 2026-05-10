from __future__ import annotations

from src.ir.instructions.common.typed_memory import TypedMemoryLoad
from src.ir.passes.base import KernelPass, PassContext


class InferPTXArgumentTypesPass(KernelPass):
    name = "infer-ptx-argument-types"

    def run(self, kernel, context: PassContext) -> None:
        inferred_types = context.metadata.setdefault("ptx_inferred_argument_types", {})
        inferred_count = 0

        for instruction in kernel.instructions.get():
            if not isinstance(instruction, TypedMemoryLoad):
                continue

            if instruction.address.name != "argptr":
                continue

            if instruction.access_type.vector_width <= 1:
                continue

            offset = self._parse_int_value(instruction.offset.value)
            if offset is None:
                continue

            argument = kernel.arguments.get_by_offset(offset)
            if argument is None or argument.name.startswith("*"):
                continue

            type_name = instruction.access_type.to_value_type()
            inferred_type = inferred_types.get(offset)
            if inferred_type is not None:
                continue

            if not kernel.arguments.update_type_by_offset(offset, type_name):
                continue

            inferred_types[offset] = type_name
            inferred_count += 1

        context.metadata["ptx_argument_types_inferred"] = inferred_count

    @staticmethod
    def _parse_int_value(value: str) -> int | None:
        try:
            return int(value, 0)
        except ValueError:
            return None
