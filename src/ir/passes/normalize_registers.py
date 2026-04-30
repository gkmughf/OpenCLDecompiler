from src.ir.passes.base import KernelPass, PassContext
from src.ir.registers.register_manager import RegisterManager
from src.ir.registers.reg import Reg32, CompositeReg, Val, RegOrVal_ty


class NormalizeRegistersPass(KernelPass):
    name = "normalize-registers"

    def run(self, kernel, context: PassContext) -> None:
        rm = RegisterManager()
        operands = []
        
        for instruction in kernel.get_instructions():
            operands.extend(instruction.get_operands())
            if instruction.get_predicate() is not None:
                operands.append(instruction.get_predicate())

        

        rm.build_mapping(operands)
        for instruction in kernel.get_instructions():
            new_operand = []
            for op in instruction.get_operands():
                if isinstance(op, CompositeReg):
                    new_operand.append(op.create_new(rm.map(op), [Reg32(rm.map(subreg)) for subreg in op.regs]))
                    continue
                new_operand.append(op.create_new(rm.map(op)))
            instruction.update_operands(*new_operand)
