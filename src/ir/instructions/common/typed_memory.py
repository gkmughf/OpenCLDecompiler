from __future__ import annotations

from dataclasses import dataclass

from src.ir.TemporaryVariableAllocator import tva
from src.ir.instructions.common.load import Load
from src.ir.instructions.common.store import Store
from src.ir.registers.reg import CompositeReg, Reg32, Reg64, RegOrVal_ty, Reg_ty, Val
from src.ir.instructions.lowering import NodeLoweringContext
from src.instructions.sop2.s_and import SAnd
from src.instructions.sop2.s_bfe import SBfe
from src.instructions.sop1.s_mov import SMov
from src.instructions.vop3.v_perm import VPerm


@dataclass(frozen=True)
class MemoryAccessType:
    address_space: str
    base_type: str
    vector_width: int = 1

    @property
    def element_bits(self) -> int:
        return int(self.base_type[1:])

    @property
    def total_bits(self) -> int:
        return self.element_bits * self.vector_width

    def to_opencl_type(self) -> str:
        return f"__{self.address_space} {self.to_value_type()}"

    def to_value_type(self) -> str:
        base = {
            "b8": "char",
            "u8": "char",
            "s8": "char",
            "b16": "short",
            "u16": "short",
            "s16": "short",
            "b32": "uint",
            "u32": "uint",
            "s32": "int",
            "b64": "ulong",
            "u64": "ulong",
            "s64": "long",
            "f16": "half",
            "f32": "float",
            "f64": "double",
        }.get(self.base_type, self.base_type)

        if self.vector_width > 1:
            base = f"{base}{self.vector_width}"

        return base

    def to_element_type(self) -> str:
        return MemoryAccessType(self.address_space, self.base_type, 1).to_value_type()


class TypedMemoryLoad(Load):
    def __init__(
        self,
        destination: Reg_ty,
        address: Reg64,
        offset: Val,
        access_type: MemoryAccessType,
        is_scalar: bool = False,
    ):
        super().__init__(
            destination,
            address,
            offset,
            is_scalar=is_scalar,
            size=max(32, access_type.total_bits),
        )
        self.access_type = access_type
        self.packed_value = self._make_internal_reg("typed_ld")

    def get_operands(self):
        if self.packed_value is not None:
            assert isinstance(self.destination, CompositeReg)
            return (
                *self.destination.regs,
                self.address,
                self.offset,
                self.packed_value,
            )

        if self._needs_dword_vector_load():
            assert isinstance(self.destination, CompositeReg)
            return (
                self.destination,
                *self.destination.regs,
                self.address,
                self.offset,
            )

        operands = list(super().get_operands())
        return tuple(operands)

    def get_parts(self) -> list[list[str]]:
        if self.packed_value is None:
            if self._needs_dword_vector_load():
                destination = self.destination.name
                address = self.address.name
                offset = self.offset.name
                return [[self._get_normalize_opcode(), destination, address, offset]]
                
            return super().get_parts()
        
        opcode = self._get_normalize_opcode()
        packed_value_str = self.packed_value.name
        address_str = self.address.name
        offset_str = self.offset.name

        result = [[opcode, packed_value_str, address_str, offset_str]]
        
        assert isinstance(self.destination, CompositeReg)
        for index, destination in enumerate(self.destination.regs):
            destination_str = destination.name
            if index == 0:
                result.append(["s_and_b32", destination_str, packed_value_str, self._low_mask()])
                continue

            selector = self._bfe_selector(index)
            result.append(["s_bfe_u32", destination_str, packed_value_str, selector])

        return result

    def to_fill_node(self, state, parents):
        if self.packed_value is None:              
            if self._needs_dword_vector_load():
                return NodeLoweringContext(state, parents).emit_backend(self._get_opcode(), self._get_normalize_opcode(), [self.destination, self.address, self.offset], self.get_suffix())
            return super().to_fill_node(state, parents)
        
        ctx = NodeLoweringContext(state, parents)
        ctx.emit_backend(self._get_opcode(), self._get_normalize_opcode(), [self.packed_value, self.address, self.offset], self.get_suffix())


        assert isinstance(self.destination, CompositeReg)
        for index, destination in enumerate(self.destination.regs):
            if index == 0:
                result = ctx.emit_backend(SAnd, "s_and_b32", [destination, self.packed_value, self._low_mask()], "b32")
                continue

            selector = self._bfe_selector(index)
            result = ctx.emit_backend(SBfe, "s_bfe_u32", [destination, self.packed_value, selector], "u32")

        return result

    def _make_internal_reg(self, prefix: str) -> Reg32 | None:
        if self._needs_small_vector_unpack():
            return Reg32(tva.generate(prefix))
        return None

    def _needs_small_vector_unpack(self) -> bool:
        return (
            isinstance(self.destination, CompositeReg)
            and self.access_type.vector_width > 1
            and self.access_type.element_bits < 32
            and self.access_type.total_bits <= 32
        )

    def _low_mask(self) -> Val:
        return Val(hex((1 << self.access_type.element_bits) - 1))

    def _bfe_selector(self, index: int) -> Val:
        offset = index * self.access_type.element_bits
        width = self.access_type.element_bits
        return Val(hex((width << 16) | offset))
    
    def _needs_dword_vector_load(self) -> bool:
        return (
            isinstance(self.destination, CompositeReg)
            and self.access_type.vector_width > 1
            and self.access_type.element_bits == 32
            and self.access_type.total_bits in {64, 128}
        )
    


class TypedMemoryStore(Store):
    PACK_SELECTOR = Val("0x2010004")

    def __init__(
        self,
        address: Reg64,
        value: RegOrVal_ty,
        access_type: MemoryAccessType,
        is_scalar: bool = False,
    ):
        super().__init__(
            address,
            value,
            is_scalar=is_scalar,
            size=max(8, access_type.total_bits),
        )
        self.access_type = access_type
        self.selector = self._make_internal_reg("perm_selector")
        self.packed_value = self._make_internal_reg("typed_st")
        self.store_value = self._make_dword_vector_store_value()


    def get_operands(self):
        if not self._needs_small_vector_pack():
            if self.store_value is not None:
                assert isinstance(self.value, CompositeReg)
                return (
                    self.address,
                    *self.value.regs,
                    self.store_value,
                    *self.store_value.regs,
                )

            return super().get_operands()

        assert isinstance(self.value, CompositeReg)
        operands = [self.address, *self.value.regs, self.selector, self.packed_value]

        return tuple(operands)


    def get_parts(self) -> list[list[str]]:
        if not isinstance(self.value, CompositeReg):
            return super().get_operands()
        
        if self._needs_small_vector_pack():
            return self._get_small_vector_store_parts()
        
        if self.store_value is not None:
            return self._get_dword_vector_store_parts()
        
        return super().get_operands()

    def to_fill_node(self, state, parents):
        if not isinstance(self.value, CompositeReg):
            return super().to_fill_node(state, parents)
        

        if self._needs_small_vector_pack():
            return self._to_fill_small_vector_store_parts(state, parents)
        
        if self.store_value is not None:
            return self._to_fill_dword_vector_store_parts(state, parents)
        
        return super().to_fill_node(state, parents)
    
    def _to_fill_small_vector_store_parts(self, state, parents):
        assert isinstance(self.value, CompositeReg)
        ctx = NodeLoweringContext(state, parents)
        opcode = self._get_opcode()
        first_value = self.value.get_element(0)
        second_value = self.value.get_element(1)
        ctx.emit_backend(SMov, "s_mov_b32", [self.selector, self.PACK_SELECTOR], "b32")
        if self.access_type.vector_width == 2:
            ctx.emit_backend(VPerm, "v_perm_b32", [self.packed_value, first_value, second_value, self.selector], "b32")
            return ctx.emit_backend(opcode, self._get_normalize_opcode(), [self.address, self.packed_value], self.get_suffix())

        return super().to_fill_node(ctx.state, ctx.parents)
    
    def _get_small_vector_store_parts(self) -> list[list[str]]:
        assert isinstance(self.value, CompositeReg)

        opcode = self._get_normalize_opcode()
        address = self.address.name
        selector = self.selector.name
        packed_value = self.packed_value.name
        first_value = self.value.get_element(0).name
        second_value = self.value.get_element(1).name

        result = [
            ["v_mov_b32", selector, self.PACK_SELECTOR],
        ]

        if self.access_type.vector_width == 2:
            result.extend(
                [
                    ["v_perm_b32", packed_value, first_value, second_value, selector],
                    [opcode, address, packed_value],
                ]
            )
            return result

        return super().get_parts()

    def _needs_small_vector_pack(self) -> bool:
        return (
            isinstance(self.value, CompositeReg)
            and self.access_type.vector_width in {2, 4}
            and self.access_type.element_bits < 32
            and self.access_type.total_bits <= 32
        )

    def _make_internal_reg(self, prefix: str) -> Reg32 | None:
        if self._needs_small_vector_pack():
            return Reg32(tva.generate(prefix))
        return None
    
    def _needs_dword_vector_store(self) -> bool:
        return (
            isinstance(self.value, CompositeReg)
            and self.access_type.vector_width > 1
            and self.access_type.element_bits == 32
            and self.access_type.total_bits in {64, 128}
        )
    
    def _to_fill_dword_vector_store_parts(self, state, parents):
        assert isinstance(self.value, CompositeReg)
        assert self.store_value is not None

        ctx = NodeLoweringContext(state, parents)
        for index, source in enumerate(self.value.regs):
            ctx.emit_backend(SMov, "s_mov_b32", [self.store_value.get_element(index), source], "b32")
            
        return ctx.emit_backend(
                self._get_opcode(),
                self._get_normalize_opcode(),
                [self.address, self.store_value],
                self.get_suffix()
        )
    
    def _get_dword_vector_store_parts(self) -> list[list[str]]:
        assert isinstance(self.value, CompositeReg)
        assert self.store_value is not None

        result = []
        for index, source in enumerate(self.value.regs):
            destination = self.store_value.get_element(index).name
            source_text = source.name
            result.append(["v_mov_b32", destination, source_text])
        result.append(
            [
                self._get_normalize_opcode(),
                self.address.name,
                self.store_value.name,
            ]
        )
        return result
    
    def _make_dword_vector_store_value(self) -> CompositeReg | None:
        if not self._needs_dword_vector_store():
            return None

        assert isinstance(self.value, CompositeReg)
        if len(self.value) != self.access_type.vector_width:
            raise ValueError(
                f"{self.access_type.to_value_type()} store expects "
                f"{self.access_type.vector_width} source registers"
            )

        name = tva.generate("typed_st_vec")
        regs = [
            Reg32(tva.generate("typed_st_vec_part"))
            for _ in range(self.access_type.vector_width)
        ]
        return CompositeReg(name, regs)