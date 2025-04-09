import numpy as np


def norm(x1, y1, x2, y2):
    """
    Normalisation de deux points
    """
    x = x2-x1
    y = y2-y1
    return np.sqrt(x**2+y**2)


def scale_mm_per_px(len_px, len_m):
    return (len_m * 1e3)/len_px


# ! 2803
x1, y1 = (42, 98)
x2, y2 = (36.2, 665.6)
len_px = norm(x1, y1, x2, y2)
len_cm = 19*1e-2
print("2803", scale_mm_per_px(len_px, len_cm))

# ! 0404
x1, y1 = (58.5, 165.4)
x2, y2 = (56.6, 710.4)
len_px = norm(x1, y1, x2, y2)
len_cm = 20*1e-2
print("0404", scale_mm_per_px(len_px, len_cm))
