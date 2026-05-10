from src.ir.passes.base import PassPipeline
from src.ir.passes.materialize_arguments import MaterializeArgumentStoresPass
from src.ir.passes.ptx_argument_types import InferPTXArgumentTypesPass
from src.ir.passes.materialize_local_memory import MaterializeLocalMemoryPass
from src.ir.passes.ptx_pointer_types import InferPTXPointerTypesPass
from src.ir.passes.register_flow import BuildRegisterFlowPass
from src.ir.passes.init_predicate import MaterializePredicatePass

PTX_PIPELINE = PassPipeline(
    [
        BuildRegisterFlowPass(),
        InferPTXArgumentTypesPass(),
        InferPTXPointerTypesPass(),

        MaterializeArgumentStoresPass(),
        MaterializeLocalMemoryPass(),
    ]
)
AMD_PIPELINE = PassPipeline(
    [
        MaterializeArgumentStoresPass(),
        MaterializeLocalMemoryPass(),
        MaterializePredicatePass(),
    ]
)
