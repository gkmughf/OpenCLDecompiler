from __future__ import annotations

from src.ir.passes.base import KernelPass, PassContext
from src.ir.registers.register_manager import RegisterManager


class NormalizeRegistersPass(KernelPass):
    name = "normalize-registers"

    def run(self, kernel, context: PassContext) -> None:
        register_manager = RegisterManager()
        operands = []

        for instruction in kernel.instructions:
            operands.extend(instruction.get_operands())
            if instruction.get_predicate() is not None:
                operands.append(instruction.get_predicate())

        

        register_manager.build_mapping(operands)
        kernel.set_register_manager(register_manager)
        context.metadata["register_manager"] = register_manager
