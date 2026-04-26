from __future__ import annotations

from src.ir.instructions.special.memory import MemoryAllocation, Store
from src.ir.passes.base import KernelPass, PassContext
from src.ir.registers.reg import Val


class MaterializeArgumentStoresPass(KernelPass):
    name = "materialize-argument-stores"

    def run(self, kernel, context: PassContext) -> None:
        if kernel.are_argument_stores_materialized():
            return

        arg_ptr = kernel.get_arg_ptr()
        if arg_ptr is None:
            return

        stores = [
            Store(
                arg_ptr,
                Val(argument.name),
                Val(argument.type_name),
                Val(str(argument.offset)),
                is_scalar=True,
            )
            for argument in kernel.get_arguments()
        ]
        instructions = [MemoryAllocation(arg_ptr, is_scalar=True), *stores]

        kernel.prepend_instructions(instructions)
        kernel.mark_argument_stores_materialized()
