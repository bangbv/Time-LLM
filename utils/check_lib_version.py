import torch # 1
import accelerate # 2
import einops # 3
import matplotlib as mpl # 4
import numpy as np # 5
import pandas as pd # 6
import sklearn # 7
import scipy as sp # 8
import tqdm # 9
import peft # 10
import transformers as tr # 11
import cpuinfo
import pydantic
import deepspeed as ds # 12
import sentencepiece as spm # 13

print("1_torch_version_2.2.2:", torch.__version__)
print("2_accelerate_version_0.28.0:", accelerate.__version__)
print("3_einops_version_0.7.0:", einops.__version__)
print("4_matplotlib_version_3.7.0:", mpl.__version__)
print("5_numpy_version_1.23.5:", np.__version__)
print("6_pandas_version_1.5.3:", pd.__version__)
print("7_sklearn_version_1.2.2:", sklearn.__version__)
print("8_scipy_version_1.12.0:", sp.__version__)
print("9_tqdm_version_4.65.0:", tqdm.__version__)
print("10_peft_version_0.4.0:", peft.__version__)
print("11_transformers_version_4.31.0:", tr.__version__)
print("12_cpuinfo_version_8.0.0:", cpuinfo.__version__)
print("13_pydantic_version_1.11.0:", pydantic.__version__)
print("14_deepspeed_version_0.14.0:", ds.__version__)
print("15_sentencepiece_version_0.2.0:", spm.__version__)
