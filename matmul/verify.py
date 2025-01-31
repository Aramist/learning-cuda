from pathlib import Path
import time 

import numpy as np

if not all([Path('L.npy').exists(), Path('R.npy').exists(), Path("result.npy").exists()]):
    raise FileNotFoundError("Did not find matmul output")

L = np.load("L.npy")
R = np.load("R.npy")
res = np.load('result.npy')
start_time = time.time()
trueprod = L @ R
end_time = time.time()

print(f'numpy multiply time: {end_time - start_time:.1f}s')

diff = res - trueprod

if not np.allclose(res, trueprod, 1e-3):
    print("Something is wrong.")
    print("L:", L)
    print("R:", R)
    print("Alleged result: ", res)
    print("True result: ", trueprod)
    exit(1)
else:
    print("everything works")
    exit(0)
