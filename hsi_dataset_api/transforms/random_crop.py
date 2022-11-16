from .transform import Transform

import random


class RandomCrop(Transform):
    def __init__(self, w, h):
        super().__init__()
        self.w = w
        self.h = h
    
    def apply(self, hsi, mask):
        c, h, w = hsi.shape
        
        start_h = random.randint(0, h - self.h - 1)
        start_w = random.randint(0, w - self.w - 1)
        
        hsi_crop = hsi[:, start_h:start_h + self.h, start_w: start_w + self.w]
        mask_crop = mask[start_h:start_h + self.h, start_w: start_w+ self.w]
        return hsi_crop, mask_crop
