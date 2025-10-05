import pandas as pd
import numpy as np
import matplotlib
import sklearn
import dowhy
import transformers
import torch
import gym
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("matplotlib:", matplotlib.__version__)
print("scikit-learn:", sklearn.__version__)
print("dowhy:", dowhy.__version__)
print("transformers:", transformers.__version__)
print("torch:", torch.__version__)
print("gym:", gym.__version__)
with open("deps_versions.txt", "w") as f:
    f.write(f"pandas: {pd.__version__}\n")
    f.write(f"numpy: {np.__version__}\n")
    f.write(f"matplotlib: {matplotlib.__version__}\n")
    f.write(f"scikit-learn: {sklearn.__version__}\n")
    f.write(f"dowhy: {dowhy.__version__}\n")
    f.write(f"transformers: {transformers.__version__}\n")
    f.write(f"torch:{torch.__version__}\n")
    f.write(f"gym:{gym.__version__}\n")