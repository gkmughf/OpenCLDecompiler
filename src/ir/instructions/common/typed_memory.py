from __future__ import annotations

from dataclasses import dataclass

from src.ir.TemporaryVariableAllocator import tva
from src.ir.instructions.common.load import Load
from src.ir.instructions.common.store import Store
from src.ir.registers.reg import CompositeReg, Reg32, Reg64, RegOrVal_ty, Reg_ty, Val
from src.ir.registers.register_manager import IDENTITY_MANAGER, RegisterManager


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

    def get_parts(self, manager: RegisterManager = IDENTITY_MANAGER) -> list[list[str]]:
        if self.packed_value is None:
            if self._needs_dword_vector_load():
                destination = manager.map(self.destination)
                address = manager.map(self.address)
                offset = manager.map(self.offset)
                return [[self._get_normalize_opcode(), destination, address, offset]]
                
            
            return super().get_parts(manager)
        
        opcode = self._get_normalize_opcode()
        packed_value_str = manager.map(self.packed_value)
        address_str = manager.map(self.address)
        offset_str = manager.map(self.offset)

        result = [[opcode, packed_value_str, address_str, offset_str]]
        
        assert isinstance(self.destination, CompositeReg)
        for index, destination in enumerate(self.destination.regs):
            destination_str = manager.map(destination)
            if index == 0:
                result.append(["s_and_b32", destination_str, packed_value_str, self._low_mask()])
                continue

            selector = self._bfe_selector(index)
            result.append(["s_bfe_u32", destination_str, packed_value_str, selector])

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

    def _low_mask(self) -> str:
        return hex((1 << self.access_type.element_bits) - 1)

    def _bfe_selector(self, index: int) -> str:
        offset = index * self.access_type.element_bits
        width = self.access_type.element_bits
        return hex((width << 16) | offset)
    
    def _needs_dword_vector_load(self) -> bool:
        return (
            isinstance(self.destination, CompositeReg)
            and self.access_type.vector_width > 1
            and self.access_type.element_bits == 32
            and self.access_type.total_bits in {64, 128}
        )
    


class TypedMemoryStore(Store):
    PACK_SELECTOR = "0x2010004"

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


    def get_parts(self, manager: RegisterManager = IDENTITY_MANAGER) -> list[list[str]]:
        if not isinstance(self.value, CompositeReg):
            return super().get_operands()
        
        if self._needs_small_vector_pack():
            return self._get_small_vector_store_parts(manager)
        
        if self.store_value is not None:
            return self._get_dword_vector_store_parts(manager)
        
        return super().get_operands()
        
        opcode = self._get_normalize_opcode()
        address = manager.map(self.address)
        first_value = manager.map(self.value.get_element(0))
        second_value = manager.map(self.value.get_element(1))
        first_stor_value = manager.map(self.stor_val.get_element(0))
        second_stor_value = manager.map(self.stor_val.get_element(1))
        stor_val_str = manager.map(self.stor_val)
        result = [
            ["v_mov_b32", first_value, first_stor_value],
            ["v_mov_b32", second_value, second_stor_value],
            [opcode, address, stor_val_str],
        ]
        return result


    def _get_small_vector_store_parts(self, manager: RegisterManager) -> list[list[str]]:
        assert isinstance(self.value, CompositeReg)

        opcode = self._get_normalize_opcode()
        address = manager.map(self.address)
        selector = manager.map(self.selector)
        packed_value = manager.map(self.packed_value)
        first_value = manager.map(self.value.get_element(0))
        second_value = manager.map(self.value.get_element(1))

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

        return super().get_parts(manager)

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
    
    def _get_dword_vector_store_parts(self, manager: RegisterManager) -> list[list[str]]:
        assert isinstance(self.value, CompositeReg)
        assert self.store_value is not None

        result = []
        for index, source in enumerate(self.value.regs):
            destination = manager.map(self.store_value.get_element(index))
            source_text = manager.map(source)
            result.append(["v_mov_b32", destination, source_text])
        address = manager.map(self.address)
        result.append(
            [
                self._get_normalize_opcode(),
                manager.map(self.address),
                manager.map(self.store_value),
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