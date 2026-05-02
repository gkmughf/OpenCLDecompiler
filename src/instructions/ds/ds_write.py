from src.base_instruction import BaseInstruction
from src.decompiler_data import set_reg_value
from src.expression_manager.types.opencl_types import OpenCLTypes
from src.ir.registers.reg import Val


class DsWrite(BaseInstruction):
    def __init__(self, node, suffix):
        super().__init__(node, suffix)
        #self.base_ptr = self.instruction[1]
        self.addr = self.operand[0]
        self.vdata0 = self.operand[1]
        #self.decompiler_data.check_lds_vars(self.offset, suffix)

    # def to_print_unresolved(self):
    #     if self.suffix == "b32":
    #         v = f"V{self.decompiler_data.number_of_v}"
    #         self.decompiler_data.write(f"uint* {v} // {self.name}\n")
    #         self.decompiler_data.write(f"{v} = (uint*)(ds + (({self.addr} + {self.offset}) & ~3))\n")
    #         self.decompiler_data.write(f"*{v} = {self.vdata0}\n")
    #         self.decompiler_data.number_of_v += 1
    #         return self.node
    #     if self.suffix == "b64":
    #         v = f"V{self.decompiler_data.number_of_v}"
    #         self.decompiler_data.write(f"ulong* {v} // {self.name}\n")
    #         self.decompiler_data.write(f"{v} = (ulong*)(ds + (({self.addr} + {self.offset}) & ~7))\n")
    #         self.decompiler_data.write(f"*{v} = {self.vdata0}\n")
    #         self.decompiler_data.number_of_v += 1
    #         return self.node
    #     return super().to_print_unresolved()

    def get_lds_var_name_with_offset(self):
        return self.expression_manager.expression_to_string(
            self.get_expression_node(self.addr)
        )
    
    def to_fill_node(self):
        if self.suffix == "b32":
            var_node_with_offset = self.get_expression_node(self.addr)
            name = self.expression_manager.expression_to_string(var_node_with_offset)
            new_value = self.node.get_from_state(self.vdata0).val
            reg_type = self.node.get_from_state(self.vdata0).type
            return set_reg_value(
                self.node,
                new_value,
                name,
                [],
                f"u{self.suffix[1:]}",
                reg_type=reg_type,
                expression_node=self.get_expression_node(self.vdata0),
            )
        return super().to_fill_node()

    def to_print(self):
        if self.suffix == "b32":
            name = self.get_lds_var_name_with_offset()
            self.output_string = (
                f"{name} = {self.expression_manager.expression_to_string(self.get_expression_node(Val(name)))}"
            )
            return self.output_string
        return super().to_print()
