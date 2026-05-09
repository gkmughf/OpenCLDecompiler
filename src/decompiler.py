import copy
from dataclasses import dataclass

from src.cfg import make_cfg_node, make_unresolved_node
from src.code_printer import create_opencl_body
from src.decompiler_data import DecompilerData, optimize_names_of_vars
from src.expression_manager.expression_manager import ExpressionManager
from src.flag_type import FlagType
from src.global_data import gdata_type_processing, process_global_data
from src.graph.control_flow_graph import CONTROL_FLOW_GRAPH_ENABLED_CONTEXT_KEY, ControlFlowGraph
from src.kernel_params import process_kernel_params
from src.logical_variable import ExecCondition
from src.model.config_data import ConfigData
from src.node import Node
from src.node_processor import check_realisation_for_node
from src.regions.functions_for_regions import make_region_graph_from_cfg, process_region_graph
from src.unrolled_loops_processing import process_unrolled_loops
from src.utils import get_context
from src.versions import change_values, check_for_use_new_version, find_max_and_prev_versions

from src.ir.instructions.special.mask import ChangeMask
from src.ir.instructions.common.endpgm import EndPgm
from src.ir.kernel import Kernel

CONTEXT = get_context()

def get_exec(node: Node) -> ExecCondition:
    return node.state["$MASK"].exec_condition

@dataclass
class _OpenMask:
    change_mask: Node
    previous_branch_end: Node | None = None

def _contains_identity(nodes: list[Node], target: Node) -> bool:
    return any(node is target for node in nodes)

def _connect(parent: Node, child: Node) -> None:
    if parent is child:
        return

    if not _contains_identity(child.parent, parent):
        child.add_parent(parent)

    if not _contains_identity(parent.children, child):
        parent.add_child(child)

def _append_unique_node(nodes: list[Node], node: Node) -> None:
    if not _contains_identity(nodes, node):
        nodes.append(node)


def _find_open_mask_index(
    open_masks: list[_OpenMask],
    exec_condition: ExecCondition,
    end: int | None = None,
) -> int | None:
    if end is None:
        end = len(open_masks)

    for index in range(end - 1, -1, -1):
        if get_exec(open_masks[index].change_mask) == exec_condition:
            return index
    return None


def _add_closed_mask_parent(parents: list[Node], open_mask: _OpenMask) -> None:
    if open_mask.previous_branch_end is not None:
        _append_unique_node(parents, open_mask.previous_branch_end)
        return

    _append_unique_node(parents, open_mask.change_mask)


def _close_open_masks(
    open_masks: list[_OpenMask],
    exec_condition: ExecCondition,
    parents: list[Node],
) -> Node | None:
    while (
        len(open_masks) > 1
        and get_exec(open_masks[-1].change_mask).is_strict_superset_of(exec_condition)
    ):
        open_mask = open_masks.pop()
        _add_closed_mask_parent(parents, open_mask)

def process_src(  # noqa: C901, PLR0912, PLR0915
    kernel: Kernel
):
    decompiler_data = DecompilerData()
    expression_manager = ExpressionManager()
    decompiler_data.reset(kernel.name)
    
    expression_manager.reset(kernel.name)

    expression_manager.set_size_of_workgroups(kernel.work_group_size)
    blocks = kernel.blocks.iter()

    #process_global_data(set_of_global_data_instruction, set_of_global_data_bytes) TODO

    decompiler_data.set_config_data(kernel)
    
    last_node = Node([""], [], decompiler_data.initial_state)
    decompiler_data.set_cfg(last_node)
    masked_blocks = [_OpenMask(last_node)]

    for bb in blocks:
        for i in bb.instructions:
            state = last_node.state
            parents = [last_node]
            change_mask_transition = None
            else_branch_index = None
            else_parent_index = None
            previous_branch_end = None

            
            if isinstance(i, ChangeMask):
                prev_exec_condition = get_exec(last_node)
                next_exec_condition = decompiler_data.exec_registers[i.predicate.name]
                else_parent_condition = prev_exec_condition.negated_sibling_parent(next_exec_condition)

                if next_exec_condition.is_strict_superset_of(prev_exec_condition):
                    change_mask_transition = "enter"
                elif next_exec_condition.is_strict_subset_of(prev_exec_condition):
                    change_mask_transition = "close"
                    _close_open_masks(masked_blocks, next_exec_condition, parents)
                elif else_parent_condition is not None:
                    change_mask_transition = "else"
                    else_branch_index = _find_open_mask_index(masked_blocks, prev_exec_condition)
                    if else_branch_index is not None:
                        previous_branch = masked_blocks[else_branch_index]
                        else_parent_index = _find_open_mask_index(
                            masked_blocks,
                            else_parent_condition,
                            else_branch_index,
                        )
                        if else_parent_index is None:
                            else_parent_index = max(else_branch_index - 1, 0)
                        parents = [previous_branch.change_mask]
                        state = previous_branch.change_mask.state
                        previous_branch_end = last_node
            
            last_node = i.to_fill_node(state, parents)

            if isinstance(i, ChangeMask):
                if change_mask_transition == "enter":
                    masked_blocks.append(_OpenMask(last_node))
                elif change_mask_transition == "else" and else_branch_index is not None:
                    if else_parent_index is None:
                        else_parent_index = max(else_branch_index - 1, 0)
                    del masked_blocks[else_parent_index + 1:]
                    masked_blocks.append(
                        _OpenMask(
                            last_node,
                            previous_branch_end,
                        )
                    )


            if isinstance(i, EndPgm):
                for masked_block in masked_blocks[1:]:
                    if masked_block.previous_branch_end is not None:
                        _connect(masked_block.previous_branch_end, last_node)
                    else:
                        _connect(masked_block.change_mask, last_node)

                
            if len(last_node.parent) > 1:
                find_max_and_prev_versions(last_node)

    optimize_names_of_vars()
    if decompiler_data.global_data:
        gdata_type_processing()
    check_for_use_new_version()
    decompiler_data.remove_unusable_versions()

    make_region_graph_from_cfg()
    process_region_graph()
    if CONTEXT.get(CONTROL_FLOW_GRAPH_ENABLED_CONTEXT_KEY):
        control_flow_graph = ControlFlowGraph.instance()
        for node in decompiler_data.starts_regions:
            control_flow_graph.build_from_node(
                node=node,
                program_name=decompiler_data.name_of_program,
                program_id=decompiler_data.pragram_id,
            )
        control_flow_graph.render()
    if decompiler_data.checked_variables != {} or decompiler_data.variables != {}:
        change_values()
    process_unrolled_loops()
    create_opencl_body()
