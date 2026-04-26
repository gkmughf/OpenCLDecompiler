from __future__ import annotations

from src.ir.instructions.special.local_memory import LocalMemory
from src.ir.passes.base import KernelPass, PassContext
from src.ir.registers.reg import Reg64, Val


class MaterializeLocalMemoryPass(KernelPass):
    name = "materialize-local-memory"

    def run(self, kernel, context: PassContext) -> None:
        if kernel.are_local_memories_materialized():
            return

        local_memories = [
            LocalMemory(destination=Reg64(name), size=Val(str(size)), is_scalar=True)
            for name, size in kernel.get_local_memories().items()
        ]

        kernel.prepend_instructions(local_memories)
        kernel.mark_local_memories_materialized()
