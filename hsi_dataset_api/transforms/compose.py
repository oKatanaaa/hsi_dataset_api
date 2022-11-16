from typing import List

from .transform import Transform


class Compose(Transform):
    def __init__(self, transforms: List[Transform]):
        super().__init__()
        self.transforms = transforms
    
    def apply(self, hsi, mask):
        for transform in self.transforms:
            hsi, mask = transform(hsi, mask)
        return hsi, mask
