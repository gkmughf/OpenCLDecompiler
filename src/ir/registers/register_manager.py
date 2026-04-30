from src.ir.registers.reg import Reg32, CompositeReg, Val, RegOrVal_ty
from src.ir.utils.RegisterGraph import RegisterGraph

class RegisterManager:
    def __init__(self):
        self._mapping: dict[str, str] = {}

    def build_mapping(self, regs: list[RegOrVal_ty]) -> bool:
        self.reset()

        graph = RegisterGraph()
        for reg in regs:
            if isinstance(reg, CompositeReg):
                graph.add(*[op.name for op in reg._regs])
            elif isinstance(reg, Reg32):
                graph.add(reg.name)
        
        numbering = graph.build()
        if numbering is not None:
            self._fill_mapping(regs, numbering)
            return True
        return False

    def _fill_mapping(self, regs: list[RegOrVal_ty], numbering: dict[str, int]) -> None:
        for reg in regs:
            if isinstance(reg, CompositeReg):
                start = numbering[reg._regs[0].name]
                end = numbering[reg._regs[-1].name]
                self._mapping[reg.name] = f"s[{start}:{end}]"
                for i, ri in enumerate(range(start, end + 1)):
                    self._mapping[reg._regs[i].name] = f"s{ri}" 
            elif isinstance(reg, Reg32):
                self._mapping[reg.name] = f"s{numbering[reg.name]}"
        

    def map(self, reg: RegOrVal_ty) -> str:
        if isinstance(reg, Val):
            return reg.value
        return self._mapping.get(reg.name, reg.name)

    def get_mapping(self) -> dict[str, str]:
        return self._mapping

    def reset(self):
        self._mapping.clear()

class IdentityRegisterManager:
    def map(self, reg: RegOrVal_ty) -> str:
        if isinstance(reg, Val):
            return reg.value
        return reg.name
    
IDENTITY_MANAGER = IdentityRegisterManager()