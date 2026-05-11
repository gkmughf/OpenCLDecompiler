from src.ir.registers.reg import PredReg, Reg64, Reg32, Val, Reg_ty, RegOrVal_ty, CompositeReg

class RegFactory:
    def __init__(self):
        self._registry: dict[str, Reg_ty] = {}


    def _create_register(self, reg_name: str) -> RegOrVal_ty:
        
        if reg_name[0] == "{" and reg_name[-1] == "}":
            normalized_name = "{" + ", ".join(part.strip() for part in reg_name[1:-1].split(",")) + "}"
            sub_registers = [
                self.get_or_create_auto(part.strip())
                for part in reg_name[1:-1].split(",")
                if part.strip()
            ]
            return CompositeReg(normalized_name, sub_registers)
        
        if reg_name[0] == "[" and reg_name[-1] == "]":
            reg_name = reg_name[1:-1]
        
        if reg_name.startswith("%rd"):
            return Reg64(reg_name)
        
        if reg_name.startswith("%fd"):
            return Reg64(reg_name)

        if reg_name.startswith("%p"):
            return PredReg(reg_name)

        if reg_name.startswith("%r") or reg_name.startswith("%rs") or reg_name.startswith("%f"):
            return Reg32(reg_name)

        special_prefixes = ("%tid", "%ctaid", "%ntid", "%envreg")
        if any(reg_name.startswith(p) for p in special_prefixes):
            return Reg32(reg_name)

        return Val(reg_name)

    def get_or_create_auto(self, reg_name: str) -> RegOrVal_ty:
        if reg_name in self._registry:
            return self._registry[reg_name]

        reg = self._create_register(reg_name)
        self._registry[reg_name] = reg

        return reg
    
    def get_or_create(self, reg_name: str, reg_type: str) -> Reg_ty:
        if reg_name in self._registry:
            return self._registry[reg_name]

        if reg_type == "32":
            new_reg = Reg32(reg_name)
        elif reg_type == "64":
            new_reg = Reg64(reg_name)
        elif reg_type in ("pred", "predicate"):
            new_reg = PredReg(reg_name)
        else:
            raise ValueError(f"Unknown PTX register type: {reg_type}")

        self._registry[reg_name] = new_reg

        return new_reg

