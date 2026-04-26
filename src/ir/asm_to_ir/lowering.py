from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from src.ir.TemporaryVariableAllocator import tva
from src.ir.registers.reg import PredReg, Reg32, Reg64, RegOrVal_ty, Reg_ty, Val

from src.ir.kernel import Kernel


class LoweringArg:
    def resolve(self, ctx: "InstructionContext") -> Any:
        raise NotImplementedError


@dataclass(frozen=True)
class OperandRef(LoweringArg):
    index: int

    def resolve(self, ctx: "InstructionContext") -> RegOrVal_ty:
        return ctx.operand(self.index)


@dataclass(frozen=True)
class TmpRegRef(LoweringArg):
    name: str
    kind: str

    def resolve(self, ctx: "InstructionContext") -> Reg_ty:
        return ctx.tmp(self.name, self.kind)


@dataclass(frozen=True)
class NamedRegRef(LoweringArg):
    name: str
    kind: str

    def resolve(self, ctx: "InstructionContext") -> Reg_ty:
        return create_register(self.name, self.kind)


@dataclass(frozen=True)
class ValueRef(LoweringArg):
    value: str

    def resolve(self, ctx: "InstructionContext") -> Val:
        return Val(self.value)


@dataclass(frozen=True)
class ArgOffsetRef(LoweringArg):
    operand_index: int
    extra_name: str = "args_offset"

    def resolve(self, ctx: "InstructionContext") -> Val:
        offsets = ctx.extras.get(self.extra_name)
        if offsets is None:
            raise KeyError(f"Lowering extra '{self.extra_name}' is required")

        operand = ctx.operand(self.operand_index)
        try:
            return offsets[operand.name]
        except KeyError as exc:
            raise KeyError(f"No argument offset for operand '{operand.name}'") from exc


@dataclass
class InstructionContext:
    kernel: Kernel
    operands: list[RegOrVal_ty]
    is_scalar: bool = False
    predicate: str | PredReg | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    _temps: dict[str, Reg_ty] = field(default_factory=dict, init=False)

    def operand(self, index: int) -> RegOrVal_ty:
        try:
            return self.operands[index]
        except IndexError as exc:
            raise IndexError(f"Lowering operand %{index} is not available") from exc

    def tmp(self, name: str, kind: str | int) -> Reg_ty:
        normalized_kind = normalize_register_kind(kind)
        existing = self._temps.get(name)
        if existing is not None:
            expected_class = register_class(normalized_kind)
            if not isinstance(existing, expected_class):
                raise TypeError(
                    f"Temporary '#{name}' was already created as "
                    f"{existing.__class__.__name__}, not {expected_class.__name__}"
                )
            return existing

        tmp_name = tva.generate(f"tmp_{name}")
        new_reg = create_register(tmp_name, normalized_kind)
        self._temps[name] = new_reg
        return new_reg

    def emit(
        self,
        instruction_class: type,
        *args: Any,
        is_scalar: bool | None = None,
        predicate: str | PredReg | None = None,
    ) -> None:
        instruction_args = tuple(resolve_arg(arg, self) for arg in args)
        actual_is_scalar = self.is_scalar if is_scalar is None else is_scalar

        self.kernel.create_instruction(
            instruction_class,
            *instruction_args,
            is_scalar=actual_is_scalar,
            predicate=predicate,
        )


@dataclass(frozen=True)
class Emit:
    instruction_class: type
    args: tuple[Any, ...]
    is_scalar: bool | None = None
    predicate: str | PredReg | None = None

    def __init__(
        self,
        instruction_class: type,
        *args: Any,
        is_scalar: bool | None = None,
        predicate: str | PredReg | None = None,
    ):
        object.__setattr__(self, "instruction_class", instruction_class)
        object.__setattr__(self, "args", tuple(args))
        object.__setattr__(self, "is_scalar", is_scalar)
        object.__setattr__(self, "predicate", predicate)

    def emit(self, ctx: InstructionContext) -> None:
        ctx.emit(
            self.instruction_class,
            *self.args,
            is_scalar=self.is_scalar,
            predicate=self.predicate,
        )


RuleCallback = Callable[[InstructionContext], None]


class Rule:
    def __init__(
        self,
        emits: Iterable[Emit] = (),
        callback: RuleCallback | None = None,
    ):
        self._emits = tuple(emits)
        self._callback = callback

    @classmethod
    def dynamic(cls, callback: RuleCallback) -> "Rule":
        return cls(callback=callback)

    def emit(self, ctx: InstructionContext) -> None:
        if self._callback is not None:
            self._callback(ctx)
            return

        for item in self._emits:
            item.emit(ctx)


def same(instruction_class: type) -> Rule:
    def emit_same(ctx: InstructionContext) -> None:
        ctx.emit(instruction_class, *list(ctx.operands))

    return Rule.dynamic(emit_same)


def op(index: int) -> OperandRef:
    return OperandRef(index)


def tmp32(name: str) -> TmpRegRef:
    return TmpRegRef(name, "32")


def tmp64(name: str) -> TmpRegRef:
    return TmpRegRef(name, "64")


def tmppred(name: str) -> TmpRegRef:
    return TmpRegRef(name, "pred")


def named32(name: str) -> NamedRegRef:
    return NamedRegRef(name, "32")


def named64(name: str) -> NamedRegRef:
    return NamedRegRef(name, "64")


def namedpred(name: str) -> NamedRegRef:
    return NamedRegRef(name, "pred")


def val(value: str | int) -> ValueRef:
    return ValueRef(str(value))


def arg_offset(operand_index: int) -> ArgOffsetRef:
    return ArgOffsetRef(operand_index)


def resolve_arg(arg: Any, ctx: InstructionContext) -> Any:
    if isinstance(arg, LoweringArg):
        return arg.resolve(ctx)
    return arg


def create_register(name: str, kind: str | int) -> Reg_ty:
    normalized_kind = normalize_register_kind(kind)
    if normalized_kind == "32":
        return Reg32(name)
    if normalized_kind == "64":
        return Reg64(name)
    if normalized_kind == "pred":
        return PredReg(name)
    raise ValueError(f"Unknown register kind: {kind}")


def register_class(kind: str | int) -> type:
    normalized_kind = normalize_register_kind(kind)
    if normalized_kind == "32":
        return Reg32
    if normalized_kind == "64":
        return Reg64
    if normalized_kind == "pred":
        return PredReg
    raise ValueError(f"Unknown register kind: {kind}")


def normalize_register_kind(kind: str | int) -> str:
    if kind in (32, "32", "b32"):
        return "32"
    if kind in (64, "64", "b64"):
        return "64"
    if kind in ("pred", "predicate"):
        return "pred"
    raise ValueError(f"Unknown register kind: {kind}")
