import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import torch.quantization
import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from datetime import datetime


sys.path.append(".")
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cpu"

startTime = datetime.now()
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
#sam.to(device=device)

# Set model to evaluation mode
sam.eval()
# Apply the quantization transformation to the model
qsam = torch.quantization.quantize_dynamic(
    sam, dtype=torch.qint8
)
print("QSAM",device,"quantization time", datetime.now() - startTime)

path = 'qsam_vit_h_4b8939.pth'

torch.save(qsam.state_dict(), path)

