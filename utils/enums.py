from enum import Enum

class BetterEnum(Enum):
    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return self.value == other.value
        elif other.__class__ == str:
            return self.value == other
        return NotImplemented
    
    def __str__(self) -> str:
        return str(self.value)
    
    def __hash__(self) -> int:
        return self.value.__hash__()
    def __repr__(self) -> str:
        return self.value.__repr__()

class Phase(BetterEnum):
    TRAIN = "Train"
    VALIDATION = "Validation"
    TEST = "Test"

class Task(BetterEnum):
    VESSEL_SEGMENTATION = "ves-seg"
    GAN_VESSEL_SEGMENTATION = "gan-ves-seg"