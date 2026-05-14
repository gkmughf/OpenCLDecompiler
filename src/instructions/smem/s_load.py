from src.base_instruction import BaseInstruction
from src.decompiler_data import DecompilerData
from src.register import Register, check_and_split_regs
from src.register_type import RegisterType
from src.upload import upload_by_offset, upload_global_data_pointer, upload_kernel_param, upload_usesetup
from src.ir.registers.reg import expand_register_names, get_reg_rang


class SLoad(BaseInstruction):
    def __init__(self, node, suffix):
        super().__init__(node, suffix)
        self.sdata = self.operand[0]
        self.sbase = self.operand[1]
        self.offset = self.operand[2].value

        self.from_registers_name = get_reg_rang(self.sbase)[0]

    # def to_print_unresolved(self):
    #     if self.suffix == "dword":
    #         self.decompiler_data.write(f"{self.sdata} = *(uint*)(smem + ({self.offset} & ~3)) // {self.name}\n")
    #         return self.node
    #     if self.suffix == "dwordx2":
    #         self.decompiler_data.write(f"{self.sdata} = *(ulong*)(smem + ({self.offset} & ~3)) // {self.name}\n")
    #         return self.node
    #     if self.suffix in {"dwordx4", "dwordx8"}:
    #         i_cnt = self.suffix[-1]
    #         self.decompiler_data.write(f"for (BYTE i = 0; i < {i_cnt}; i++) // {self.name}\n")
    #         self.decompiler_data.write(f"    {self.sdata}[i] = *(uint*)(SMEM + i*4 + ({self.offset} & ~3))\n")
    #         return self.node
    #     return super().to_print_unresolved()

    def to_fill_node(self):
        sbase: Register | None = self.node.get_from_state(self.from_registers_name)

        if sbase is not None and self.suffix in {"dword", "dwordx2", "dwordx4", "dwordx8", "b32", "b64", "b128"}:
            if sbase.val.isdigit():
                self.offset = hex(int(self.offset, 16) + int(sbase.val))
            if sbase.type in {RegisterType.GLOBAL_DATA_POINTER, RegisterType.ARGUMENTS_POINTER}:
                if sbase.type == RegisterType.ARGUMENTS_POINTER:
                    upload_kernel_param(self.node.state, int(self.offset, 16), self.sdata, self.sbase)
                elif sbase.type == RegisterType.GLOBAL_DATA_POINTER:
                    upload_global_data_pointer(self.node.state, self.sdata, self.sbase)
            else:
                #TODO(GFV) эта ветка не исполняется 
                upload_usesetup(self.node.state, self.sdata, self.offset)
            return self.node
        return super().to_fill_node()
