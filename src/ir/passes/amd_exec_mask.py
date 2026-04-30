from __future__ import annotations

from src.ir.blocks import BlockType
from src.ir.instructions.special.mask import ChangeMask
from src.ir.passes.base import KernelPass, PassContext
from src.ir.passes.build_blocks import BuildKernelBlocksPass
from src.ir.registers.reg import PredReg


class InsertAMDExecMaskPass(KernelPass):
    name = "insert-amd-exec-mask"

    def run(self, kernel, context: PassContext) -> None:
        blocks = kernel.get_blocks()
        if not blocks:
            BuildKernelBlocksPass().run(kernel, context)
            blocks = kernel.get_blocks()

        insert_after_indices = []
        for block in blocks.values():
            if block.block_type != BlockType.PREDICATE_UPDATE:
                continue

            if "exec" not in block.modified_predicates:
                continue

            if self._is_existing_exec_mask(kernel.instructions, block.end + 1):
                continue

            insert_after_indices.append(block.end)

        if not insert_after_indices:
            return

        instructions = list(kernel.instructions)
        for index in sorted(insert_after_indices, reverse=True):
            instructions.insert(index + 1, ChangeMask(PredReg("exec")))

        kernel.instructions = instructions
        kernel.close()

    @staticmethod
    def _is_existing_exec_mask(instructions, index: int) -> bool:
        if index >= len(instructions):
            return False

        return isinstance(instructions[index], ChangeMask)
