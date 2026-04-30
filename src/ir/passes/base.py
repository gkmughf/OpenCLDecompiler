from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

from src.ir.kernel import Kernel


@dataclass
class PassContext:
    metadata: dict[str, object] = field(default_factory=dict)


class KernelPass(ABC):
    name = "kernel-pass"

    @abstractmethod
    def run(self, kernel: Kernel, context: PassContext) -> None:
        pass


class PassPipeline:
    def __init__(self, passes: Iterable[KernelPass]):
        self._passes = list(passes)

    def run(self, kernel: Kernel, context: PassContext | None = None) -> PassContext:
        if context is None:
            context = PassContext()

        for kernel_pass in self._passes:
            kernel_pass.run(kernel, context)

        return context
