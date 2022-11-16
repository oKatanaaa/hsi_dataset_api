from .transform import Transform


class Lambda(Transform):
    def __init__(self, lamb):
        super().__init__()
        self.lamb = lamb
    
    def apply(self, hsi, mask):
        hsi, mask = self.lamb((hsi, mask))
        return hsi, mask
