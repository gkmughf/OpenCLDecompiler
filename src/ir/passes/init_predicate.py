from src.ir.instructions.special.initPredicate import InitPredicate
from src.ir.passes.base import KernelPass, PassContext


class MaterializePredicatePass(KernelPass):
    name = "materialize-predicate"

    def run(self, kernel, context: PassContext) -> None:
        if kernel.predicates.is_materialized():
            return

        init_predicates = [
            InitPredicate(predicate=reg, is_scalar=True)
            for reg in kernel.predicates.all()
        ]

        kernel.prepend_instructions(init_predicates)
        kernel.predicates.mark_materialized()
