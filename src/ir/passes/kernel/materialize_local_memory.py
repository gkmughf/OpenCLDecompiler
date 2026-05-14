from src.ir.instructions.special.local_memory import LocalMemory
from src.ir.passes.base import PassContext
from src.ir.registers.reg import Reg64, Val


class MaterializeLocalMemoryPass:
    name = "materialize-local-memory"

    def run(self, kernel, _context: PassContext) -> None:
        if kernel.local_memory.is_materialized():
            return

        local_memories = [
            LocalMemory(destination=Reg64(name), size=Val(str(size)))
            for name, size in kernel.local_memory.all().items()
        ]

        kernel.instructions.prepend_list(local_memories)
        kernel.local_memory.mark_materialized()
