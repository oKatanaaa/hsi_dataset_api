from abc import abstractmethod


class Transform:
    @abstractmethod
    def apply(self, hsi, mask):
        pass
    
    def __call__(self, hsi, mask):
        return self.apply(hsi, mask)