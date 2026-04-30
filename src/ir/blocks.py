from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum, auto

from src.ir.instructions.generic import GenericInstruction


ENTRY_BLOCK_ID = -1

class BlockType(Enum):
    LABEL = auto()
    LINEAR = auto()
    BRANCH = auto()
    TERMINATOR = auto()


@dataclass
class KernelBlock:
    block_type: BlockType
    instructions: list[GenericInstruction] = field(default_factory=list)
    successor_blocks: set[int] = field(default_factory=set)
    predecessor_blocks: set[int] = field(default_factory=set)

    @property
    def block_ty(self) -> BlockType:
        return self.block_type

    @property
    def incoming_blocks(self) -> set[int]:
        return self.predecessor_blocks

    @property
    def first_instruction(self) -> GenericInstruction:
        if not self.instructions:
            raise ValueError("Kernel block has no instructions")
        return self.instructions[0]

    @property
    def last_instruction(self) -> GenericInstruction:
        if not self.instructions:
            raise ValueError("Kernel block has no instructions")
        return self.instructions[-1]

    def append_instruction(self, instruction: GenericInstruction) -> None:
        self.instructions.append(instruction)

    def prepend_instruction(self, instruction: GenericInstruction) -> None:
        self.instructions.append(instruction)

    def append_instructions(self, instructions: list[GenericInstruction]) -> None:
        self.instructions = self.instructions + instructions

    def prepend_instructions(self, instructions: list[GenericInstruction]) -> None:
        self.instructions = instructions + self.instructions

    def extend_instructions(self, instructions: Iterable[GenericInstruction]) -> None:
        self.instructions.extend(instructions)


class KernelBlocks:
    def __init__(self) -> None:
        self._blocks: dict[int, KernelBlock] = {}

    def append_instruction(self, instruction: GenericInstruction) -> None:
        self.ensure_blok()
        self.last_block().append_instruction(instruction)

    def prepend_instruction(self, instruction: GenericInstruction) -> None:
        self.ensure_blok()
        self.first_block().prepend_instruction(instruction)

    def append_instructions(self, instructions: list[GenericInstruction]) -> None:
        self.ensure_blok()
        self.last_block().append_instructions(instructions)

    def prepend_instructions(self, instructions: list[GenericInstruction]) -> None:
        self.ensure_blok()
        self.first_block().prepend_instructions(instructions)

    def ensure_blok(self):
        if not self._blocks:
            self._blocks[0] = KernelBlock(
                block_type=BlockType.LINEAR,
                instructions=[],
            )


    def set(self, blocks: dict[int, KernelBlock]) -> None:
        self._blocks = dict(blocks)

    def get(self) -> dict[int, KernelBlock]:
        return self._blocks

    def iter(self) -> Iterator[KernelBlock]:
        return iter(self._blocks.values())

    def iter_instructions(self) -> Iterator[GenericInstruction]:
        for block in self._blocks.values():
            yield from block.instructions

    def get_instructions(self) -> list[GenericInstruction]:
        return list(self.iter_instructions())

    def instruction_parts(self) -> list[list[str]]:
        result = []
        for instruction in self.iter_instructions():
            result.extend(instruction.get_parts())
        return result
    
    def last_block(self) -> KernelBlock:
        return next(iter(reversed(self._blocks.values())))
    
    def first_block(self) -> KernelBlock:
        return next(iter((self._blocks.values())))
