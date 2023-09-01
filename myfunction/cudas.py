from torch import cuda

def devices():
    return "cuda" if cuda.is_available() else "cpu"