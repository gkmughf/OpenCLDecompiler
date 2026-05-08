from src.base_instruction import BaseInstruction
from src.decompiler_data import set_reg_value
from src.register import Register
from src.integrity import Integrity

import re

class InitPred(BaseInstruction):
    def to_fill_node(self):
        dest  = self.node.operands[0]
        self.decompiler_data.init_predicate(self.node.state, dest.name)
        self.decompiler_data.init_predicate(self.node.state, dest.get_element(0).name)
        self.decompiler_data.init_predicate(self.node.state, dest.get_element(1).name)
        
        return self.node