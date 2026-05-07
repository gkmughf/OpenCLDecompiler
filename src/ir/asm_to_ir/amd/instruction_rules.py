from src.ir.asm_to_ir.lowering import Emit, InstructionContext, Rule, op, same, tmp64
from src.ir.instructions.common.add import Add, AddC
from src.ir.instructions.common.barrier import Barrier
from src.ir.instructions.common.bfe import bfe, bfe_s
from src.ir.instructions.common.endpgm import EndPgm
from src.ir.instructions.common.load import Load32, Load64, Load128, FLoad32, FLoad64, FLoad128
from src.ir.instructions.common.logical import And, Or, Xor
from src.ir.instructions.common.lshl import AShr, AShr_Rev, LShl, LShl_Rev, LShr, LShr_Rev
from src.ir.instructions.common.mad import Mad
from src.ir.instructions.common.mov import Mov
from src.ir.instructions.common.mul import Mul24, MulHi, MulHi_s, MulLo, MulLo_s
from src.ir.instructions.common.store import Store8, Store16, Store32, Store64, Store128, FStore8, FStore16, FStore32, FStore64, FStore128
from src.ir.instructions.common.sub import Sub, SubRev
from src.ir.instructions.common.cvt import Cvt64_32, Cvt_i32_f32
from src.ir.instructions.special.local_memory import LocalAdd, LocalLoad, LocalStore
from src.ir.instructions.common.permute import Permute32
from src.ir.registers.reg import PredReg, Val
from src.ir.instructions.common.compare import get_compare_class
from src.ir.instructions.control_flow import Branch, BranchNot, Label
from src.ir.instructions.special.mask import ChangeMask

LOGICAL_INSTRUCTIONS = {
    "and": And,
    "or": Or,
    "xor": Xor,
}

def _ignore_explicit_vcc(instruction_class: type) -> Rule:
    def emit(ctx: InstructionContext) -> None:
        if len(ctx.operands) >= 4 and ctx.operands[1].name == "vcc":
            ctx.emit(instruction_class, ctx.operands[0], ctx.operands[2], ctx.operands[3])
            return

        ctx.emit(instruction_class, *ctx.operands)

    return Rule.dynamic(emit)

def _writes_exec(ctx: InstructionContext) -> bool:
    return bool(ctx.operands) and ctx.operand(0).name == "exec"

def _same_with_exec_mask(instruction_class: type) -> Rule:
    def emit(ctx: InstructionContext) -> None:
        ctx.emit(instruction_class, *ctx.operands)
        if _writes_exec(ctx):
            ctx.emit(ChangeMask, PredReg("exec"), is_scalar=True)

    return Rule.dynamic(emit)


def get_instruction_rule(opcode: str) -> Rule | None:
    if opcode.startswith("."):
        return _label(opcode)
    
    if opcode.startswith("s_cmp_"):
        return _compare_rule(opcode, PredReg("scc"), is_scalar=True)

    if opcode.startswith("v_cmpx_"):
        return _compare_rule(opcode, PredReg("exec"), is_scalar=False)

    if opcode.startswith("v_cmp_"):
        return _vector_compare_rule(opcode)

    return instruction_rules.get(opcode)


def _parse_compare_opcode(opcode: str) -> str:
    comparison = opcode.split("_")[2]
    if not comparison:
        raise NotImplementedError(opcode)
    return comparison


def _label(opcode: str) -> Rule:
    def emit(ctx: InstructionContext) -> None:
        ctx.emit(
            Label,
            opcode[:-1],
            is_scalar=True,
        )

    return Rule.dynamic(emit)

def _compare_rule(opcode: str, destination: PredReg, is_scalar: bool) -> Rule:
    comparison = _parse_compare_opcode(opcode)
    compare_class = get_compare_class(comparison)

    def emit(ctx: InstructionContext) -> None:
        ctx.emit(
            compare_class,
            destination,
            ctx.operand(0),
            ctx.operand(1),
            is_scalar=is_scalar,
        )
        if destination.name == "exec":
            ctx.emit(ChangeMask, PredReg("exec"), is_scalar=True)

    return Rule.dynamic(emit)


def _vector_compare_rule(opcode: str) -> Rule:
    comparison = _parse_compare_opcode(opcode)
    compare_class = get_compare_class(comparison)

    def emit(ctx: InstructionContext) -> None:
        ctx.emit(
            compare_class,
            ctx.operand(0),
            ctx.operand(1),
            ctx.operand(2),
            is_scalar=False,
        )

    return Rule.dynamic(emit)


def _branch(predicate: PredReg) -> Rule:
    def emit(ctx: InstructionContext) -> None:
        ctx.emit(
            Branch,
            predicate, 
            Val(ctx.operand(0).name),
            is_scalar=True,
        )

    return Rule.dynamic(emit)

def _branch_not(predicate: PredReg) -> Rule:
    def emit(ctx: InstructionContext) -> None:
        ctx.emit(
            BranchNot,
            predicate, 
            Val(ctx.operand(0).name),
            is_scalar=True,
        )

    return Rule.dynamic(emit)

def _saveexec(operation) -> Rule:
    return Rule(
        [
            Emit(Mov, op(0), PredReg("exec"), is_scalar=True),
            Emit(operation, PredReg("exec"), PredReg("exec"), op(1), is_scalar=True),
            Emit(ChangeMask, PredReg("exec"), is_scalar=True),
        ]
    )

_local_store_like = Rule(
    [
        Emit(Mov, tmp64("base"), op(2)),
        Emit(Cvt64_32, tmp64("index"), op(0)),
        Emit(Add, tmp64("address"), tmp64("base"), tmp64("index")),
        Emit(LocalStore, tmp64("address"), op(1)),
    ]
)

_local_add_like = Rule(
    [
        Emit(Mov, tmp64("base"), op(2)),
        Emit(Cvt64_32, tmp64("index"), op(0)),
        Emit(Add, tmp64("address"), tmp64("base"), tmp64("index")),
        Emit(LocalAdd, tmp64("address"), op(1)),
    ]
)

_local_load = Rule(
    [
        Emit(Mov, tmp64("base"), op(2)),
        Emit(Cvt64_32, tmp64("index"), op(1)),
        Emit(Add, tmp64("address"), tmp64("base"), tmp64("index")),
        Emit(LocalLoad, op(0), tmp64("address")),
    ]
)



instruction_rules = {
    "s_branch": _branch(None),
    "s_cbranch_scc0": _branch_not(PredReg("scc")),
    "s_cbranch_scc1": _branch(PredReg("scc")),
    "s_cbranch_execz": _branch_not(PredReg("exec")),
    "s_cbranch_execnz": _branch(PredReg("exec")),
    "s_cbranch_vccz": _branch_not(PredReg("vcc")),
    "s_cbranch_vccnz": _branch(PredReg("vcc")),

    "s_and_saveexec_b64": _saveexec(And),
    "s_or_saveexec_b64": _saveexec(Or),
    "s_xor_saveexec_b64": _saveexec(Xor),
    "s_andn2_saveexec_b64": _saveexec(Xor),

    "v_add_u32": _ignore_explicit_vcc(Add),
    "s_add_u32": _ignore_explicit_vcc(Add),
    "v_addc_u32": _ignore_explicit_vcc(AddC),
    "s_addc_u32": _ignore_explicit_vcc(AddC),

    "s_sub_u32": _ignore_explicit_vcc(Sub),
    "v_subrev_u32": _ignore_explicit_vcc(SubRev),
    "v_sub_u32": _ignore_explicit_vcc(Sub),
    "s_subb_u32": _ignore_explicit_vcc(Sub),
    "v_subb_u32": _ignore_explicit_vcc(Sub),

    "v_mul_u32": same(MulLo),
    "s_mul_u32": same(MulLo),
    "v_mul_hi_u32": same(MulHi),
    "s_mul_hi_u32": same(MulHi),
    "v_mul_lo_u32": same(MulLo),
    "s_mul_lo_u32": same(MulLo),
    "v_mul_i32": same(MulLo_s),
    "s_mul_i32": same(MulLo_s),
    "v_mul_hi_i32": same(MulHi_s),
    "s_mul_hi_i32": same(MulHi_s),
    "v_mul_i32_i24": same(Mul24),

    "v_mad_u32": same(Mad),
    "s_mad_u32": same(Mad),

    "v_lshl_b32": same(LShl),
    "s_lshl_b32": same(LShl),
    "s_lshl_b64": same(LShl),
    "v_lshlrev_b32": same(LShl_Rev),
    "s_lshlrev_b32": same(LShl_Rev),
    "v_lshlrev_b64": same(LShl_Rev),
    "v_lshr_b32": same(LShr),
    "s_lshr_b32": same(LShr),
    "v_lshrrev_b32": same(LShr_Rev),
    "s_lshrrev_b32": same(LShr_Rev),
    "v_lshrrev_b64": same(LShr_Rev),
    "s_ashr_i32": same(AShr),
    "v_ashrrev_i64": same(AShr_Rev),

    "s_and_b32": _same_with_exec_mask(And),
    "s_and_b64": _same_with_exec_mask(And),
    "v_and_b32": _same_with_exec_mask(And),
    "s_xor_b32": _same_with_exec_mask(Xor),
    "s_xor_b64": _same_with_exec_mask(Xor),
    "v_xor_b32": _same_with_exec_mask(Xor),
    "s_or_b32": _same_with_exec_mask(Or),
    "s_or_b64": _same_with_exec_mask(Or),
    "v_or_b32": _same_with_exec_mask(Or),

    "v_mov_b32": _same_with_exec_mask(Mov),
    "s_mov_b32": _same_with_exec_mask(Mov),
    "s_mov_b64": _same_with_exec_mask(Mov),
    "s_movk_i32": _same_with_exec_mask(Mov),

    "s_load_dword": same(Load32),
    "s_load_dwordx2": same(Load64),
    "s_load_dwordx4": same(Load128),
    "flat_load_dword": same(FLoad32),
    "flat_load_dwordx2": same(FLoad64),
    "flat_load_dwordx4": same(FLoad128),

    "flat_store_byte": same(FStore8),
    "flat_store_short": same(FStore16),
    "flat_store_dword": same(FStore32),
    "flat_store_dwordx2": same(FStore64),
    "flat_store_dwordx4": same(FStore128),
    "s_store_dword": same(Store32),
    "s_store_dwordx2": same(Store64),

    "ds_write_b32": _local_store_like,
    "ds_read_b32": _local_load,
    "ds_add_u32": _local_add_like,

    "s_bfe_u32": same(bfe),
    "v_bfe_u32": same(bfe),
    "s_bfe_i32": same(bfe_s),
    "v_bfe_i32": same(bfe_s),

    "v_perm_b32": same(Permute32),

    "v_cvt_i32_f32": same(Cvt_i32_f32),

    "s_endpgm": same(EndPgm),

    "s_waitcnt": Rule([Emit(Barrier)]),
}
