from src.ir.instructions.generic import GenericInstruction
from src.ir.instructions.special.mask import ChangeMask, Unmask
from src.ir.instructions.common.Not import Not
from src.ir.passes.base import KernelPass, PassContext
from src.ir.registers.reg import PredReg


class PTXPredicatesPass(KernelPass):
    name = "ptx-predicates"

    def run(self, kernel, context: PassContext) -> None:
        instructions: list[GenericInstruction] = []
        open_predicate: PredReg | None = None

        for instruction in kernel.instructions.get():
            predicate = instruction.get_predicate()

            if predicate is None or instruction.is_control_flow():
                if open_predicate is not None:
                    instructions.append(Unmask(is_scalar=True))
                    open_predicate = None
                instructions.append(instruction)
                continue

            predicate_negated = instruction.is_predicate_negated()
            mask_predicate = predicate
            if predicate_negated:
                mask_predicate = PredReg(f"{predicate.name}_not")
                kernel.predicates.add(mask_predicate)

            if open_predicate is None:
                if predicate_negated:
                    instructions.append(Not(mask_predicate, predicate, is_scalar=True))
                instructions.append(ChangeMask(mask_predicate, is_scalar=True))
                open_predicate = mask_predicate
            elif open_predicate.name != mask_predicate.name:
                instructions.append(Unmask(is_scalar=True))
                if predicate_negated:
                    instructions.append(Not(mask_predicate, predicate, is_scalar=True))
                instructions.append(ChangeMask(mask_predicate, is_scalar=True))
                open_predicate = mask_predicate

            instruction.set_predicate(None)
            instructions.append(instruction)

        if open_predicate is not None:
            instructions.append(Unmask(is_scalar=True))

        kernel.instructions.replace_all(instructions)
