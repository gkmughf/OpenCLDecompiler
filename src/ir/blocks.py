from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


DEFAULT_PREDICATE_NAME = "__DEFAULT"
ENTRY_BLOCK_ID = -1


class BlockType(Enum):
    LABEL = auto()
    LINEAR = auto()
    DEFAULT_BLOCK = LINEAR
    PREDICATE_UPDATE = auto()
    BRANCH = auto()
    TERMINATOR = auto()


@dataclass
class KernelBlock:
    block_type: BlockType
    start: int
    end: int
    predicate_name: str = DEFAULT_PREDICATE_NAME
    modified_predicates: set[str] = field(default_factory=set)
    successor_blocks: set[int] = field(default_factory=set)
    predecessor_blocks: set[int] = field(default_factory=set)

    @property
    def block_ty(self) -> BlockType:
        return self.block_type

    @property
    def incoming_blocks(self) -> set[int]:
        return self.predecessor_blocks
