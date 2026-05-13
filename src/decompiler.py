from src.ir.kernel import Kernel
from src.ir.passes.pipelines import KERNEL_DECOMPILATION_PIPELINE


def process_src(kernel: Kernel) -> None:
    KERNEL_DECOMPILATION_PIPELINE.run(kernel)
