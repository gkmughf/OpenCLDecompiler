from __future__ import annotations

from src.ir.blocks import DEFAULT_PREDICATE_NAME, ENTRY_BLOCK_ID, BlockType, KernelBlock
from src.ir.instructions.control_flow import Branch, Label
from src.ir.instructions.generic import GenericInstruction
from src.ir.passes.base import KernelPass, PassContext
from src.ir.registers.reg import PredReg


class BuildKernelBlocksPass(KernelPass):
    name = "build-kernel-blocks"

    def run(self, kernel, context: PassContext) -> None:
        blocks = self._build_blocks(kernel.instructions)
        kernel.set_blocks(blocks)
        context.metadata["blocks"] = blocks

    def _build_blocks(self, instructions: list[GenericInstruction]) -> dict[int, KernelBlock]:
        if not instructions:
            return {}

        blocks = self._find_blocks(instructions)
        label_to_block_id = self._map_labels_to_blocks(instructions, blocks)
        successors = self._build_control_flow_edges(instructions, blocks, label_to_block_id)
        self._fill_predecessor_blocks(blocks, successors)
        return blocks

    def _find_blocks(self, instructions: list[GenericInstruction]) -> dict[int, KernelBlock]:
        blocks: dict[int, KernelBlock] = {}
        block_id = 0
        index = 0

        while index < len(instructions):
            instruction = instructions[index]

            if isinstance(instruction, Label):
                blocks[block_id] = self._make_single_instruction_block(
                    BlockType.LABEL,
                    instruction,
                    index,
                )
                block_id += 1
                index += 1
                continue

            if instruction.is_control_flow():
                block_type = BlockType.BRANCH if isinstance(instruction, Branch) else BlockType.TERMINATOR
                blocks[block_id] = self._make_single_instruction_block(block_type, instruction, index)
                block_id += 1
                index += 1
                continue

            modified_predicates = self._get_instruction_written_predicate_names(instruction)
            if modified_predicates:
                blocks[block_id] = self._make_single_instruction_block(
                    BlockType.PREDICATE_UPDATE,
                    instruction,
                    index,
                    modified_predicates=modified_predicates,
                )
                block_id += 1
                index += 1
                continue

            start = index
            index += 1
            while index < len(instructions) and not self._starts_standalone_block(instructions[index]):
                index += 1

            end = index - 1
            blocks[block_id] = KernelBlock(
                block_type=BlockType.LINEAR,
                start=start,
                end=end,
                predicate_name=self._get_instruction_predicate_name(instructions[start]),
            )
            block_id += 1

        return blocks

    def _make_single_instruction_block(
        self,
        block_type: BlockType,
        instruction: GenericInstruction,
        index: int,
        modified_predicates: set[str] | None = None,
    ) -> KernelBlock:
        if modified_predicates is None:
            modified_predicates = self._get_instruction_written_predicate_names(instruction)

        return KernelBlock(
            block_type=block_type,
            start=index,
            end=index,
            predicate_name=self._get_instruction_predicate_name(instruction),
            modified_predicates=modified_predicates,
        )

    def _starts_standalone_block(self, instruction: GenericInstruction) -> bool:
        return (
            isinstance(instruction, Label)
            or instruction.is_control_flow()
            or bool(self._get_instruction_written_predicate_names(instruction))
        )

    def _map_labels_to_blocks(
        self,
        instructions: list[GenericInstruction],
        blocks: dict[int, KernelBlock],
    ) -> dict[str, int]:
        label_to_block_id: dict[str, int] = {}

        for block_id, block in blocks.items():
            if block.block_type != BlockType.LABEL:
                continue

            label = instructions[block.start]
            if isinstance(label, Label):
                label_to_block_id[label.name.value] = block_id

        return label_to_block_id

    def _build_control_flow_edges(
        self,
        instructions: list[GenericInstruction],
        blocks: dict[int, KernelBlock],
        label_to_block_id: dict[str, int],
    ) -> dict[int, set[int]]:
        successors = {block_id: set() for block_id in blocks}
        ordered_block_ids = list(blocks.keys())
        next_block_by_id = {
            block_id: ordered_block_ids[index + 1]
            for index, block_id in enumerate(ordered_block_ids[:-1])
        }

        for block_id, block in blocks.items():
            block.successor_blocks.clear()
            last_instruction = instructions[block.end]
            next_block_id = next_block_by_id.get(block_id)

            if isinstance(last_instruction, Branch):
                target_block_id = label_to_block_id.get(last_instruction.target.value)
                if target_block_id is not None:
                    successors[block_id].add(target_block_id)

                if last_instruction.has_predicate() and next_block_id is not None:
                    successors[block_id].add(next_block_id)
                continue

            if last_instruction.is_control_flow() and not isinstance(last_instruction, Label):
                continue

            if next_block_id is not None:
                successors[block_id].add(next_block_id)

        for block_id, target_blocks in successors.items():
            blocks[block_id].successor_blocks.update(target_blocks)

        return successors

    def _fill_predecessor_blocks(
        self,
        blocks: dict[int, KernelBlock],
        successors: dict[int, set[int]],
    ) -> None:
        if not blocks:
            return

        first_block_id = next(iter(blocks))
        for block in blocks.values():
            block.predecessor_blocks.clear()

        blocks[first_block_id].predecessor_blocks.add(ENTRY_BLOCK_ID)

        for block_id, target_blocks in successors.items():
            for target_block_id in target_blocks:
                blocks[target_block_id].predecessor_blocks.add(block_id)

    @staticmethod
    def _get_instruction_predicate_name(instruction: GenericInstruction) -> str:
        predicate = instruction.get_predicate()
        if predicate is None:
            return DEFAULT_PREDICATE_NAME
        return normalize_predicate_name(predicate.name)

    @staticmethod
    def _get_instruction_written_predicate_names(instruction: GenericInstruction) -> set[str]:
        written_predicates = set(instruction.get_written_predicate_names())

        for register in instruction.get_written_registers():
            if isinstance(register, PredReg):
                written_predicates.add(normalize_predicate_name(register.name))

        return written_predicates
