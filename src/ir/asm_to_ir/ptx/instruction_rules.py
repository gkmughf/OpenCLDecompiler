from src.ir.asm_to_ir.lowering import Emit, Rule, named64, op, same
from src.ir.instructions.common.add import Add, AddF
from src.ir.instructions.common.barrier import Barrier
from src.ir.instructions.common.bfe import bfe, bfe_s
from src.ir.instructions.common.compare import get_compare_class
from src.ir.instructions.common.cvt import Cvt32_16, Cvt64_32, Cvt64_32_s, Cvt_i32_f32, Cvt32_64, Cvt_f64_u32
from src.ir.instructions.common.endpgm import EndPgm
from src.ir.instructions.common.logical import And, Or
from src.ir.instructions.common.lshl import LShl, AShr
from src.ir.instructions.common.mad import Mad
from src.ir.instructions.common.mov import Mov
from src.ir.instructions.common.mul import MulHi, MulHi_s, MulLo, MulLo_s, MulWide, MulWide_s, Mul_f
from src.ir.instructions.common.sub import Sub
from src.ir.instructions.control_flow import Branch, BranchNot, Jump, Label
from src.ir.instructions.special.local_memory import LocalAdd, LocalLoad, LocalStore
from src.ir.instructions.common.typed_memory import (
    MemoryAccessType,
    TypedMemoryFLoad,
    TypedMemoryFStore,
    TypedMemoryLoad,
)
import re
from src.ir.registers.reg import Val, PredReg
from src.ir.instructions.common.Not import Not
from src.ir.instructions.common.cselect import CSelect
from src.ir.instructions.common.min import IRMin


_MEMORY_TYPE_PATTERN = re.compile(r"^[busf](8|16|32|64)$")
_MEMORY_ADDRESS_OFFSET_PATTERN = re.compile(
    r"^\[\s*(%[\w.$]+)\s*([+-])\s*([+-]?(?:0x[0-9a-fA-F]+|\d+))\s*\]$"
)

def get_instruction_rule(opcode: str, args_offset) -> Rule | None:
    if opcode.startswith("."):
        return _label(opcode)
    if opcode in ("bra", "bra.uni"):
        return _branch()
    if opcode.startswith("setp."):
        return _setp(opcode)
    if opcode.startswith("ld.param."):
        return _param_load(opcode, args_offset)
    if opcode.startswith("ld.global."):
        return _global_load(opcode)
    if opcode.startswith("st.global."):
        return _global_store(opcode)
    return instruction_rules.get(opcode)

def _label(opcode: str) -> Rule:
    def emit(ctx) -> None:
        ctx.emit(
            Label,
            "."+opcode[1:-1],
        )

    return Rule.dynamic(emit)

def _branch() -> Rule:
    def emit_branch(ctx) -> None:
        target = Val("."+ctx.operand(0).name)
        predicate = ctx.predicate

        if predicate is None:
            ctx.kernel.create_instruction(Jump, target)
        elif ctx.predicate_negated:
            ctx.kernel.create_instruction(Branch, predicate, target)
        else:
            tmo_predicate = PredReg(f"{predicate.name}Not")
            ctx.kernel.predicates.add(tmo_predicate)
            ctx.kernel.create_instruction(Not, tmo_predicate, predicate)
            ctx.kernel.create_instruction(BranchNot, tmo_predicate, target)

    return Rule.dynamic(emit_branch)


def _parse_setp_opcode(opcode: str) -> str:
    parts = opcode.split(".")
    if len(parts) != 3:
        raise NotImplementedError(opcode)

    _, comparison, _ = parts

    normalized_comparison = {
        "lo": "lt",
        "ls": "le",
        "leu": "le",
        "hi": "gt",
        "hs": "ge",
        "geu": "ge",
        "equ": "eq",
        "neu": "ne",
    }.get(comparison, comparison)

    return normalized_comparison


def _setp(opcode: str) -> Rule:
    comparison = _parse_setp_opcode(opcode)
    compare_class = get_compare_class(comparison)

    def emit_setp(ctx) -> None:
        ctx.emit(
            compare_class,
            ctx.operand(0),
            ctx.operand(1),
            ctx.operand(2),
        )

    return Rule.dynamic(emit_setp)



def _param_load(opcode: str, args_offset) -> Rule:
    access_type = _parse_memory_access_type(opcode)
    def emit_global_load(ctx) -> None:
        ctx.emit(TypedMemoryLoad, ctx.operand(0), named64("argptr"), args_offset[ctx.operand(1).name], access_type)

    return Rule.dynamic(emit_global_load)


def _global_load(opcode: str) -> Rule:
    access_type = _parse_memory_access_type(opcode)
    def emit_global_load(ctx) -> None:
        address = _emit_address_operand(ctx, 1)
        ctx.emit(TypedMemoryFLoad, ctx.operand(0), address, Val("0"), access_type)

    return Rule.dynamic(emit_global_load)


def _global_store(opcode: str) -> Rule:
    access_type = _parse_memory_access_type(opcode)
    def emit_global_store(ctx) -> None:
        address = _emit_address_operand(ctx, 0)
        ctx.emit(TypedMemoryFStore, address, ctx.operand(1), access_type)

    return Rule.dynamic(emit_global_store)


def _operand_token(ctx, operand_index: int) -> str:
    operand_tokens = ctx.extras.get("operand_tokens", [])
    if operand_index < len(operand_tokens):
        return operand_tokens[operand_index]

    return ctx.operand(operand_index).name


def _parse_memory_address_offset(token: str) -> Val | None:
    match = _MEMORY_ADDRESS_OFFSET_PATTERN.match(token)
    if match is None:
        return None

    sign = match.group(2)
    offset = int(match.group(3), 0)
    if sign == "-":
        offset = -offset

    return Val(str(offset))


def _emit_address_operand(ctx, operand_index: int):
    offset = _parse_memory_address_offset(_operand_token(ctx, operand_index))
    if offset is None:
        return ctx.operand(operand_index)

    offset32 = ctx.tmp("offset", "32")
    offset64 = ctx.tmp("offset64", "64")
    address = ctx.tmp("address", "64")

    ctx.emit(Mov, offset32, offset)
    ctx.emit(Cvt64_32_s, offset64, offset32)
    ctx.emit(Add, address, ctx.operand(operand_index), offset64)
    return address


def _parse_memory_access_type(opcode: str) -> MemoryAccessType:
    parts = opcode.split(".")
    address_space = "global"
    vector_width = 1
    base_type = None

    for part in parts[2:]:
        if part.startswith("v") and part[1:].isdigit():
            vector_width = int(part[1:])
            continue

        if _MEMORY_TYPE_PATTERN.match(part):
            base_type = part

    if base_type is None:
        raise NotImplementedError(opcode)

    return MemoryAccessType(
        address_space=address_space,
        base_type=base_type,
        vector_width=vector_width,
    )

instruction_rules = {
    "add.s32": same(Add),
    "add.s64": same(Add),
    "add.s16": same(Add),
    "add.f64": same(AddF),
    
    "sub.s32": same(Sub),
    "sub.s64": same(Sub),
    "sub.s16": same(Sub),

    "mul.lo.s16": same(MulLo_s),
    "mul.lo.s32": same(MulLo_s),
    "mul.hi.s32": same(MulHi_s),
    "mul.lo.u32": same(MulLo),
    "mul.hi.u32": same(MulHi),
    "mul.wide.s32": same(MulWide_s),
    "mul.wide.u32": same(MulWide),
    "mul.lo.s64": same(MulLo),
    "mul.f32": same(Mul_f),

    "mad.lo.s32": same(Mad),

    "mov.u16": same(Mov),
    "mov.b32": same(Mov),
    "mov.u32": same(Mov),
    "mov.b64": same(Mov),
    "mov.u64": same(Mov),
    "mov.f32": same(Mov),

    "cvt.s64.s32": same(Cvt64_32_s),
    "cvt.u64.u32": same(Cvt64_32),
    "cvt.u32.u64": same(Cvt32_64),
    "cvt.u16.u32": same(Cvt32_16),
    "cvt.rzi.s32.f32": same(Cvt_i32_f32),
    "cvt.rn.f64.u32": same(Cvt_f64_u32),
    "cvt.s32.s16": same(Cvt32_16),

    "shl.b16": same(LShl),
    "shl.b32": same(LShl),
    "shl.b64": same(LShl),
    "shr.s64": same(AShr),

    "and.b64": same(And),

    "st.shared.u32": same(LocalStore),
    "atom.shared.add.u32": Rule(
        [
            Emit(LocalAdd, op(1), op(2)),
        ]
    ),
    "ld.shared.u32": same(LocalLoad),

    "s_bfe_u32": same(bfe),
    "v_bfe_u32": same(bfe),
    "s_bfe_i32": same(bfe_s),
    "v_bfe_i32": same(bfe_s),

    "bar.sync": Rule([Emit(Barrier)]),
    
    "selp.b32": Rule([Emit(CSelect, op(0), op(2), op(1), op(3))]),
    "selp.b64": Rule([Emit(CSelect, op(0), op(2), op(1), op(3))]),
    "min.s32": same(IRMin),
    "ret": same(EndPgm),
    "or.pred": same(Or),
    "and.b32": same(And),
    "neg.s32": same(Not)
}
