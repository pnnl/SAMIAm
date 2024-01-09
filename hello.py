
# # import logging
# from models.FENet18 import Net as FENet18
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import cv2
# import math
# import torchvision.transforms as transforms
# import pandas as pd
# import shutil
# import json
# import torch
# import torch.nn.functional as F

# import torchvision.models as models

# import torch.nn as nn
# import argparse
# import pickle

# import os
# from datetime import datetime

# print("BEFORE MAIN")

# if __name__ == '__main__':

#     #torch.cuda.empty_cache()
#     print("IN MAIN")

#     NOW = str(datetime.now()).replace(" ","--").split(".")[0]
#     OUTPUT_DIR = '/logs/' + NOW + '/'

#     os.makedirs(os.path.dirname(os.path.abspath(os.getcwd()) + OUTPUT_DIR), exist_ok=True)


from models.FENet18 import Net as FENet18


print("BEFORE MAIN")

if __name__ == '__main__':
    print("IN MAIN")