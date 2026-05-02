from src.base_instruction import BaseInstruction
from src.decompiler_data import set_reg_value

import re

class InitPred(BaseInstruction):
    def to_fill_node(self):
        dest  = self.node.operands[0]
        exec_condition = self.decompiler_data.exec_registers["exec"]

        prev_exec_cond_node = self.get_expression_node("exec")

        self.decompiler_data.exec_registers[dest] = exec_condition
        set_reg_value(
            self.node,
            exec_condition.top(),
            dest,
            ["exec"],
            None,
            expression_node=prev_exec_cond_node,
        )
        
        return self.node