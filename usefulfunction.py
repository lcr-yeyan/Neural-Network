import numpy as np


def normalize_vector(v):  # 向量归一化
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm
