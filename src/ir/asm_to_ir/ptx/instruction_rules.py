from src.ir.asm_to_ir.lowering import Emit, Rule, arg_offset, named64, op, same
from src.ir.instructions.common.add import Add
from src.ir.instructions.common.barrier import Barrier
from src.ir.instructions.common.bfe import bfe, bfe_s
from src.ir.instructions.common.cvt import Cvt32_16, Cvt64_32, Cvt64_32_s, Cvt_i32_f32
from src.ir.instructions.common.endpgm import EndPgm
from src.ir.instructions.common.load import Load32, Load64#, LoadParamVector2U8
from src.ir.instructions.common.logical import And
from src.ir.instructions.common.lshl import LShl
from src.ir.instructions.common.mad import Mad
from src.ir.instructions.common.mov import Mov
from src.ir.instructions.common.mul import MulHi, MulHi_s, MulLo, MulLo_s, MulWide, MulWide_s
from src.ir.instructions.common.store import FStore8, FStore32, FStore64#, StoreGlobalVector2U8
from src.ir.instructions.common.sub import Sub
from src.ir.instructions.special.local_memory import LocalAdd, LocalLoad, LocalStore
from src.ir.instructions.common.typed_memory import MemoryAccessType, TypedMemoryLoad, TypedMemoryStore
import re
from src.ir.registers.reg import Reg64, Val


_MEMORY_TYPE_PATTERN = re.compile(r"^[busf](8|16|32|64)$")





def get_instruction_rule(opcode: str, args_offset) -> Rule | None:
    if opcode.startswith("ld.param."):
        return _param_load(opcode, args_offset)
    if opcode.startswith("ld.global."):
        return _global_load(opcode)
    if opcode.startswith("st.global."):
        return _global_store(opcode)
    return instruction_rules.get(opcode)

def _param_load(opcode: str, args_offset) -> Rule:
    access_type = _parse_memory_access_type(opcode)
    def emit_global_load(ctx) -> None:
        ctx.emit(TypedMemoryLoad, ctx.operand(0), named64("argptr"), args_offset[ctx.operand(1).name], access_type, is_scalar=True)

    return Rule.dynamic(emit_global_load)


def _global_load(opcode: str) -> Rule:
    access_type = _parse_memory_access_type(opcode)
    def emit_global_load(ctx) -> None:
        ctx.emit(TypedMemoryLoad, ctx.operand(0), ctx.operand(1), Val("0"), access_type, is_scalar=False)

    return Rule.dynamic(emit_global_load)


def _global_store(opcode: str) -> Rule:
    access_type = _parse_memory_access_type(opcode)
    def emit_global_store(ctx) -> None:
        ctx.emit(TypedMemoryStore, ctx.operand(0), ctx.operand(1), access_type, is_scalar=False)

    return Rule.dynamic(emit_global_store)


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

# _param_load_vector2_u8 = Rule(
#     [
#         Emit(LoadParamVector2U8, op(0), named64("argptr"), arg_offset(1), is_scalar=True),
#     ]
# )


instruction_rules = {
    "add.s32": same(Add),
    "add.s64": same(Add),
    "add.s16": same(Add),

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

    "mad.lo.s32": same(Mad),

    "mov.u16": same(Mov),
    "mov.b32": same(Mov),
    "mov.u32": same(Mov),
    "mov.b64": same(Mov),
    "mov.u64": same(Mov),

    "cvt.s64.s32": same(Cvt64_32_s),
    "cvt.u64.u32": same(Cvt64_32),
    "cvt.u16.u32": same(Cvt32_16),
    "cvt.rzi.s32.f32": same(Cvt_i32_f32),

    "shl.b16": same(LShl),
    "shl.b32": same(LShl),
    "shl.b64": same(LShl),

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

    "bar.sync": same(Barrier),

    "ret": same(EndPgm),
}
