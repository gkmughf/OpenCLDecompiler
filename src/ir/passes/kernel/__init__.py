from src.ir.passes.kernel.init_predicate import MaterializePredicatePass
from src.ir.passes.kernel.materialize_arguments import MaterializeArgumentStoresPass
from src.ir.passes.kernel.materialize_local_memory import MaterializeLocalMemoryPass

__all__ = [
    "MaterializeArgumentStoresPass",
    "MaterializeLocalMemoryPass",
    "MaterializePredicatePass",
]
