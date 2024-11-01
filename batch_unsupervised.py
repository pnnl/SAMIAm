import os
import glob
import shutil
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, Subset, DataLoader
import torch
import matplotlib.pyplot as plt
import cv2
import torch.quantization
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import random
import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from datetime import datetime
from utils import *

class ChessDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.image_paths = glob.glob(os.path.join(folder_path, '*.tiff')) + glob.glob(os.path.join(folder_path, '*.jpg'))
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)

        if self.transform is not None:
            image = self.transform(image)

        # Extract the filename from the image path
        filename = os.path.basename(image_path)

        return image, filename

    def __len__(self):
        return len(self.image_paths)

NOW = str(datetime.now()).replace(" ","--").split(".")[0]

#Define default global variable values
SAM = 'sam_vit_h_4b8939.pth'
BATCH_SIZE = 1
SUBSET_SIZE = 10
IDEAL_CHIP_SIZE = 60
DATA_DIR = './CHESS_Labeling_Round_1/12-15-23_Spurgeon_CHESS_Labeling_Round_1/dataset/' # Expert 1
#DATA_DIR = './CHESS_Labeling_Round_1/12-17-23_Doty_CHESS_Labeling_Round_1/dataset/'     # Expert 2
#DATA_DIR = './chessdataset_pychip/' # For ICML
OUTPUT_DIR = '/logs/' + NOW + '/'
COPY_ORIGINAL_DATA = True
OVERLAP = 90
K = 10

#SamAutoMaskGenerator params
POINTS_PER_SIDE = 12
CROP_N_LAYERS = 1
PRED_IOU_THRESH = 0.9
STABILITY_SCORE_THRESH = 0.92
CROP_N_POINTS_DOWNSCALE_FACTOR = 2



if __name__ == '__main__':

    #torch.cuda.empty_cache()
    print("IN MAIN")

    args = get_args()

    SAM = args.sam
    POINTS_PER_SIDE = args.grid
    IDEAL_CHIP_SIZE = args.chipsize
    DILATION = args.dilation
    IOU_THRESH = args.iou_thresh

    OPTIONS = {
    'SAM': SAM,
    'Image batch size': BATCH_SIZE,
    'Experiment data subset size': SUBSET_SIZE,
    'Ideal chip size': IDEAL_CHIP_SIZE,
    'Data dir': DATA_DIR,
    'Chip mask overlap percentage': OVERLAP,
    'Chip sample size (K)': K,
    'Chip classifier/encoder': args.model,
    'Embed chips (0 or 1)': args.embed,
    'Post processing (0 or 1)': args.post,
    'POINTS_PER_SIDE': POINTS_PER_SIDE,
    'CROP_N_LAYERS': CROP_N_LAYERS,
    'PRED_IOU_THRESH': PRED_IOU_THRESH,
    'STABILITY_SCORE_THRESH': STABILITY_SCORE_THRESH,
    'CROP_N_POINTS_DOWNSCALE_FACTOR': CROP_N_POINTS_DOWNSCALE_FACTOR,
    'BOUNDARY MASK DILATION':DILATION,
    'IOU_THRESH for pairing mask and label': IOU_THRESH,
    'Par computation (0 or 1)': args.par,
    }

    os.makedirs(os.path.dirname(os.path.abspath(os.getcwd()) + OUTPUT_DIR), exist_ok=True)

    #Write OPTIONS to file
    titles = ["OPTIONS", "VALUES"]
    spacing = [40,20]
    write_to_file(OPTIONS, titles, os.path.abspath(os.getcwd()) + OUTPUT_DIR + 'options.log',spacing)

    logger = get_logger()

    #sys.path.append(".")
    sam_checkpoint = SAM
    if 'vit_h' in SAM:
        model_type = "vit_h"
    elif 'vit_l' in SAM:
        model_type = "vit_l"
    elif 'vit_b' in SAM:
        model_type = "vit_b"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE:", device)

    startTime = datetime.now()
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    print("SAM",device,"model loading time:", datetime.now() - startTime)

    # Define the transformation(s) you want to apply to the images
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Add more transformations as needed
    ])

    # Instantiate the custom dataset
    dataset = ChessDataset(DATA_DIR, transform=transform)

    # Slice the dataset to get the first X images
    subset_dataset = Subset(dataset, indices=range(SUBSET_SIZE))

    # Instantiate the DataLoader with the subset dataset
    dataloader = DataLoader(subset_dataset, batch_size=BATCH_SIZE, shuffle=False)

    #Instantiate num materials for each material
    #Expert 1:
    num_materials = {
                    "STEM_ADF_11-02-18_10_nm_LFO-STO_A_LO_0_103118_0002":3,
                    "STEM_JEOL_HAADF_04-27-17_13_nm_STO_p-Ge_033117_LO_110_042617_HAADF_0001":4,
                    "STEM_JEOL_ADF1_02-20-20_Yingge_4nm_WO3_-_NbSTO_052617_LO_020620_0005":3,
                    "STEM_JEOL_ADF1_10-12-20_La0-8Sr0-2FeO3-STO-080317-2-LO-zero-deg_0004_1":4,
                    "STEM_JEOL_ADF1_02-20-20_Yingge_4nm_WO3_-_NbSTO_052617_LO_020620_0001":3,
                    "STEM_ADF_11-02-18_10_nm_LFO-STO_A_LO_0_103118_0001":3,
                    "STEM_JEOL_ADF1_02-20-20_Yingge_4nm_WO3_-_NbSTO_052617_LO_020620_0002":3,
                    "STEM_ADF_09-24-18_30_nm_LMO_STO_081317B_LO_091618_0004":3,
                    "STEM_JEOL_ADF1_03-16-20_Wangoh_LSTO_-_STO_0-25_TEM_012020_LO_0_031020_0005_1":3,
                    "STEM_JEOL_ADF1_03-16-20_Wangoh_LSTO_-_STO_0-25_TEM_012020_LO_0_031020_0001":4,
                    "STEM_JEOL_ADF1_10-12-20_La0-8Sr0-2FeO3-STO-080317-2-LO-zero-deg_0001_1":4,
                    "STEM_JEOL_ADF1_10-12-20_La0-8Sr0-2FeO3-STO-080317-2-LO-zero-deg_0005_1":3,
                    "STEM_ADF_11-02-18_10_nm_LFO-STO_A_LO_0_103118_0004":4,
                    "STEM_JEOL_ADF1_03-16-20_Wangoh_LSTO_-_STO_0-25_TEM_012020_LO_0_031020_0003":4,
                    "STEM_JEOL_ADF1_06-29-20_Wangoh_LSTO_-_STO_0-25_TEM_012020_LO_0_031020_Thinner_0004":3,
                    "STEM_JEOL_ADF1_03-16-20_Wangoh_LSTO_-_STO_0-25_TEM_012020_LO_0_031020_0002":4,
                    "STEM_JEOL_ADF1_03-16-20_Wangoh_LSTO_-_STO_0-25_TEM_012020_LO_0_031020_0006":2,
                    "STEM_JEOL_ADF1_06-29-20_Wangoh_LSTO_-_STO_0-25_TEM_012020_LO_0_031020_Thinner_0002":4,
                    "STEM_ADF_09-24-18_30_nm_LMO_STO_081317B_LO_091618_0006":3,
                    "STEM_ADF_09-24-18_30_nm_LMO_STO_081317B_LO_091618_0008":3,
                    "STEM_JEOL_ADF1_10-12-20_La0-8Sr0-2FeO3-STO-080317-2-LO-zero-deg_0002":4,
                    "STEM_JEOL_ADF1_06-29-20_Wangoh_LSTO_-_STO_0-25_TEM_012020_LO_0_031020_Thinner_0003_1":4,
                    "STEM_ADF_09-24-18_30_nm_LMO_STO_081317B_LO_091618_0005":3,
                    "STEM_ADF_11-02-18_10_nm_LFO-STO_A_LO_0_103118_0003":3,
                    "STEM_JEOL_HAADF_04-27-17_13_nm_STO_p-Ge_033117_LO_110_042617_HAADF_0005":3,
                    "STEM_JEOL_ADF1_03-16-20_Wangoh_LSTO_-_STO_0-25_TEM_012020_LO_0_031020_0004_1":3,
                    "STEM_JEOL_HAADF_04-27-17_13_nm_STO_p-Ge_033117_LO_110_042617_HAADF_0006":3,
                    "STEM_JEOL_HAADF_04-27-17_13_nm_STO_p-Ge_033117_LO_110_042617_HAADF_0003":4,
                    "STEM_JEOL_ADF1_10-12-20_La0-8Sr0-2FeO3-STO-080317-2-LO-zero-deg_0003_1":4,
                    "STEM_JEOL_ADF1_10-12-20_La0-8Sr0-2FeO3-STO-080317-2-LO-zero-deg_0006_1":2,
                    "STEM_JEOL_HAADF_04-27-17_13_nm_STO_p-Ge_033117_LO_110_042617_HAADF_0007":3,
                    "STEM_JEOL_ADF1_06-29-20_Wangoh_LSTO_-_STO_0-25_TEM_012020_LO_0_031020_Thinner_0001":4,
                    }

    mixin_coeef = {
        "STEM_ADF_11-02-18_10_nm_LFO-STO_A_LO_0_103118_0002":{},
        "STEM_JEOL_HAADF_04-27-17_13_nm_STO_p-Ge_033117_LO_110_042617_HAADF_0001":{'task-12-annotation-20-by-2-tag-Ge-10.png':10,'task-12-annotation-20-by-2-tag-PtC-10.png':10,'task-12-annotation-20-by-2-tag-SrTiO3-1.png':1,'task-12-annotation-20-by-2-tag-Vac-10.png':10},
        "STEM_JEOL_ADF1_02-20-20_Yingge_4nm_WO3_-_NbSTO_052617_LO_020620_0005":{},
        "STEM_JEOL_ADF1_10-12-20_La0-8Sr0-2FeO3-STO-080317-2-LO-zero-deg_0004_1":{'SrTiO3-merged-2.png':10,'task-30-annotation-34-by-2-tag-LSFO1-1.png':1,'task-30-annotation-34-by-2-tag-LSFO2-1.png':1,'task-30-annotation-34-by-2-tag-PtC-2.png':10},
        "STEM_JEOL_ADF1_02-20-20_Yingge_4nm_WO3_-_NbSTO_052617_LO_020620_0001":{},
        "STEM_ADF_11-02-18_10_nm_LFO-STO_A_LO_0_103118_0001":{},
        "STEM_JEOL_ADF1_02-20-20_Yingge_4nm_WO3_-_NbSTO_052617_LO_020620_0002":{'task-10-annotation-14-by-2-tag-PtC-10.png':10,'task-10-annotation-14-by-2-tag-SrTiO3-10.png':10,'task-10-annotation-14-by-2-tag-WO3-1.png':1},
        "STEM_ADF_09-24-18_30_nm_LMO_STO_081317B_LO_091618_0004":{},
        "STEM_JEOL_ADF1_03-16-20_Wangoh_LSTO_-_STO_0-25_TEM_012020_LO_0_031020_0005_1":{'La-merged-2.png':10,'task-25-annotation-29-by-2-tag-PtC-1.png':1,'task-25-annotation-29-by-2-tag-Unk-1.png':1},
        "STEM_JEOL_ADF1_03-16-20_Wangoh_LSTO_-_STO_0-25_TEM_012020_LO_0_031020_0001":{'PtC-merged-1.png':1,'task-21-annotation-26-by-2-tag-La-SrTiO3-1.png':1,'task-21-annotation-26-by-2-tag-SrTiO3-2.png':10,'task-21-annotation-26-by-2-tag-Vac-2.png':10},
        "STEM_JEOL_ADF1_10-12-20_La0-8Sr0-2FeO3-STO-080317-2-LO-zero-deg_0001_1":{},
        "STEM_JEOL_ADF1_10-12-20_La0-8Sr0-2FeO3-STO-080317-2-LO-zero-deg_0005_1":{'task-31-annotation-35-by-2-tag-LSFO1-1.png':1,'task-31-annotation-35-by-2-tag-LSFO2-1.png':1,'task-31-annotation-35-by-2-tag-SrTiO3-2.png':10},
        "STEM_ADF_11-02-18_10_nm_LFO-STO_A_LO_0_103118_0004":{'task-4-annotation-8-by-2-tag-LaFeO3-1.png':1,'task-4-annotation-8-by-2-tag-PtC-10.png':10,'task-4-annotation-8-by-2-tag-SrTiO3-10.png':10,'task-4-annotation-8-by-2-tag-Vac-10.png':10},
        "STEM_JEOL_ADF1_03-16-20_Wangoh_LSTO_-_STO_0-25_TEM_012020_LO_0_031020_0003":{'task-23-annotation-27-by-2-tag-La-SrTiO3-1.png':1,'task-23-annotation-27-by-2-tag-PtC-10.png':10,'task-23-annotation-27-by-2-tag-SrTiO3-10.png':10,'task-23-annotation-27-by-2-tag-Vac-10.png':10},
        "STEM_JEOL_ADF1_06-29-20_Wangoh_LSTO_-_STO_0-25_TEM_012020_LO_0_031020_Thinner_0004":{},
        "STEM_JEOL_ADF1_03-16-20_Wangoh_LSTO_-_STO_0-25_TEM_012020_LO_0_031020_0002":{'task-22-annotation-25-by-2-tag-La-SrTiO3-1.png':1,'task-22-annotation-25-by-2-tag-PtC-10.png':10,'task-22-annotation-25-by-2-tag-SrTiO3-10.png':10,'task-22-annotation-25-by-2-tag-Vac-10.png':10},
        "STEM_JEOL_ADF1_03-16-20_Wangoh_LSTO_-_STO_0-25_TEM_012020_LO_0_031020_0006":{},
        "STEM_JEOL_ADF1_06-29-20_Wangoh_LSTO_-_STO_0-25_TEM_012020_LO_0_031020_Thinner_0002":{},
        "STEM_ADF_09-24-18_30_nm_LMO_STO_081317B_LO_091618_0006":{},
        "STEM_ADF_09-24-18_30_nm_LMO_STO_081317B_LO_091618_0008":{},
        "STEM_JEOL_ADF1_10-12-20_La0-8Sr0-2FeO3-STO-080317-2-LO-zero-deg_0002":{},
        "STEM_JEOL_ADF1_06-29-20_Wangoh_LSTO_-_STO_0-25_TEM_012020_LO_0_031020_Thinner_0003_1":{'task-19-annotation-23-by-2-tag-La-SrTiO3-10.png':10,'task-19-annotation-23-by-2-tag-PtC-10.png':10,'task-19-annotation-23-by-2-tag-SrTiO3-10.png':10,'task-19-annotation-23-by-2-tag-Unk-1.png':1},
        "STEM_ADF_09-24-18_30_nm_LMO_STO_081317B_LO_091618_0005":{},
        "STEM_ADF_11-02-18_10_nm_LFO-STO_A_LO_0_103118_0003":{},
        "STEM_JEOL_HAADF_04-27-17_13_nm_STO_p-Ge_033117_LO_110_042617_HAADF_0005":{},
        "STEM_JEOL_ADF1_03-16-20_Wangoh_LSTO_-_STO_0-25_TEM_012020_LO_0_031020_0004_1":{'task-24-annotation-28-by-2-tag-La-SrTiO3-10.png':10,'task-24-annotation-28-by-2-tag-PtC-10.png':10,'task-24-annotation-28-by-2-tag-SrTiO3-1.png':1},
        "STEM_JEOL_HAADF_04-27-17_13_nm_STO_p-Ge_033117_LO_110_042617_HAADF_0006":{},
        "STEM_JEOL_HAADF_04-27-17_13_nm_STO_p-Ge_033117_LO_110_042617_HAADF_0003":{},
        "STEM_JEOL_ADF1_10-12-20_La0-8Sr0-2FeO3-STO-080317-2-LO-zero-deg_0003_1":{'LSFO1-merged-1.png':1,'task-29-annotation-33-by-2-tag-LSFO2-1.png':1,'task-29-annotation-33-by-2-tag-PtC-2.png':10,'task-29-annotation-33-by-2-tag-SrTiO3-2.png':10},
        "STEM_JEOL_ADF1_10-12-20_La0-8Sr0-2FeO3-STO-080317-2-LO-zero-deg_0006_1":{'LSFO1-merged-1.png':1,'task-32-annotation-36-by-2-tag-LSFO2-1.png':1},
        "STEM_JEOL_HAADF_04-27-17_13_nm_STO_p-Ge_033117_LO_110_042617_HAADF_0007":{},
        "STEM_JEOL_ADF1_06-29-20_Wangoh_LSTO_-_STO_0-25_TEM_012020_LO_0_031020_Thinner_0001":{'PtC-merged-2.png':10,'task-17-annotation-21-by-2-tag-La-SrTiO3-1.png':1,'task-17-annotation-21-by-2-tag-SrTiO3-2.png':10,'task-17-annotation-21-by-2-tag-Vac-2.png':10},
    }

    #Expert 2:

    num_materials = {
        "STEM_ADF_09-24-18-50-nm-LMO-STO-081618-LO-091618_0002":4,
        "STEM_JEOL-ADF1_12-10-2020-LaFeO3-STO-092917-b-LO-zero-deg-12-8-2020_0112_1":0,
        "STEM_JEOL-ADF1_11-20-19-Spurgeon-60-nm-LaMnO3-STO-001-073119-LO-103019_0007":0,
        "STEM_ADF_09-24-18-50-nm-LMO-STO-081618-LO-091618_0008":2,
        "STEM_JEOL-ADF1_12-10-2020-LaFeO3-STO-092917-b-LO-zero-deg-12-8-2020_0109_1":0,
        "STEM_JEOL-ADF1_03-16-20-Wangoh-LSTO-STO-0.25-TEM-012020-LO-0-031020_0001":2,
        "STEM_JEOL-ADF1_03-16-20-Wangoh-LSTO-STO-0.25-TEM-012020-LO-0-031020_0005_1":3,
        "STEM_JEOL-ADF1_11-20-19-Spurgeon-60-nm-LaMnO3-STO-001-073119-LO-103019_0011":0,
        "STEM_JEOL-ADF1_11-20-19-Spurgeon-60-nm-LaMnO3-STO-001-073119-LO-103019_0015":0,
        "STEM_JEOL-ADF1_02-20-20-Yingge-4nm-WO3-NbSTO-052617-LO-020620_0002":3,
        "STEM_JEOL-HAADF_04-27-17-13-nm-STO-p-Ge-033117-LO-110-042617-HAADF_0006":3,
        "STEM_JEOL-ADF1_07-23-19-Du-STO-Ge-LO-45-071919_0007":0,
        "STEM_ADF_11-02-18-10-nm-LFO-STO-A-LO-0-103118_0003":3,
        "STEM_JEOL-ADF1_12-10-2020-LaFeO3-STO-092917-b-LO-zero-deg-12-8-2020_0115_1":0,
        "STEM_ADF_09-24-18-50-nm-LMO-STO-081618-LO-091618_0004":3,
        "STEM_JEOL-ADF1_08-13-20-Wang-1-1-STO-SNO-LSAT-062620-a-LO-45-081320_0010_1":0,
        "STEM_JEOL-ADF1_03-16-20-Wangoh-LSTO-STO-0.25-TEM-012020-LO-0-031020_0006":2,
        "STEM_JEOL-ADF1_12-10-2020-LaFeO3-STO-092917-b-LO-zero-deg-12-8-2020_0106_1":0,
        "STEM_JEOL-ADF1_07-23-19-Du-STO-Ge-LO-45-071919_0010":0,
        "STEM_JEOL-ADF1_08-13-20-Wang-1-1-STO-SNO-LSAT-062620-a-LO-45-081320_0006_1":0,
        "STEM_JEOL-ADF1_12-10-2020-LaFeO3-STO-092917-b-LO-zero-deg-12-8-2020_0117_1":0,
        "STEM_JEOL-ADF1_12-10-2020-LaFeO3-STO-092917-b-LO-zero-deg-12-8-2020_0118_1":0,
        "STEM_JEOL-ADF1_06-29-20-Wangoh-LSTO-STO-0.25-TEM-012020-LO-0-031020-Thinner_0003_1":3,
        "STEM_JEOL-ADF1_08-03-20-Wang-1-STO-1-SNO-LSAT-062620-a-LO-45-072720_0001":0,
        "STEM_JEOL-ADF1_11-20-19-Spurgeon-60-nm-LaMnO3-STO-001-073119-LO-103019_0013":0,
        "STEM_ADF_11-02-18-10-nm-LFO-STO-A-LO-0-103118_0002":3,
        "STEM_JEOL-ADF1_08-13-20-Wang-1-1-STO-SNO-LSAT-062620-a-LO-45-081320_0003":0,
        "STEM_JEOL-ADF1_08-13-20-Wang-1-1-STO-SNO-LSAT-062620-a-LO-45-081320_0012":0,
        "STEM_JEOL-ADF1_08-13-20-Wang-1-1-STO-SNO-LSAT-062620-a-LO-45-081320_0005_1":0,
        "STEM_JEOL-ADF1_06-29-20-Wangoh-LSTO-STO-0.25-TEM-012020-LO-0-031020-Thinner_0004":3,
        "STEM_JEOL-ADF1_11-20-19-Spurgeon-60-nm-LaMnO3-STO-001-073119-LO-103019_0009":0,
        "STEM_JEOL-ADF1_07-23-19-Du-STO-Ge-LO-45-071919_0012":0,
        "STEM_JEOL-ADF1_02-20-20-Yingge-4nm-WO3-NbSTO-052617-LO-020620_0005":3,
        "STEM_JEOL-ADF1_11-20-19-Spurgeon-60-nm-LaMnO3-STO-001-073119-LO-103019_0005":0,
        "STEM_JEOL-ADF1_08-13-20-Wang-1-1-STO-SNO-LSAT-062620-a-LO-45-081320_0008":0,
        "STEM_JEOL-HAADF_05-02-17-5-uc-STO-p-Ge-033117-LO-110-051116-HAADF_0007":0,
        "STEM_JEOL-ADF1_12-10-2020-LaFeO3-STO-092917-b-LO-zero-deg-12-8-2020_0105":0,
        "STEM_JEOL-ADF1_10-12-20-La0.8Sr0.2FeO3-STO-080317-2-LO-zero-deg_0005_1":2,
        "STEM_ADF_09-24-18-50-nm-LMO-STO-081618-LO-091618_0009":2,
        "STEM_JEOL-HAADF_05-02-17-5-uc-STO-p-Ge-033117-LO-110-051116-HAADF_0003":0,
        "STEM_ADF_01-12-18-4-uc-LaMnO3-072817A-LO-100-01118-EELS_0004":3,
        "STEM_ADF_09-24-18-50-nm-LMO-STO-081618-LO-091618_0007":2,
        "STEM_JEOL-ADF1_07-23-19-Du-STO-Ge-LO-45-071919_0001":0,
        "STEM_JEOL-ADF1_03-16-20-Wangoh-LSTO-STO-0.25-TEM-012020-LO-0-031020_0004_1":3,
        "STEM_JEOL-ADF1_02-20-20-Yingge-4nm-WO3-NbSTO-052617-LO-020620_0001":3,
        "STEM_JEOL-HAADF_05-02-17-5-uc-STO-p-Ge-033117-LO-110-051116-HAADF_0005":0,
        "STEM_JEOL-HAADF_04-27-17-13-nm-STO-p-Ge-033117-LO-110-042617-HAADF_0001":3,
        "STEM_ADF_09-24-18-30-nm-LMO-STO-081317B-LO-091618_0006":3,
        "STEM_JEOL-HAADF_05-02-17-5-uc-STO-p-Ge-033117-LO-110-051116-HAADF_0001":0,
        "STEM_JEOL-HAADF_05-02-17-5-uc-STO-p-Ge-033117-LO-110-051116-HAADF_0002":0,
        "STEM_JEOL-HAADF_04-27-17-13-nm-STO-p-Ge-033117-LO-110-042617-HAADF_0003":3,
        "STEM_JEOL-ADF1_10-12-20-La0.8Sr0.2FeO3-STO-080317-2-LO-zero-deg_0004_1":3,
        "STEM_JEOL-ADF1_08-13-20-Wang-1-1-STO-SNO-LSAT-062620-a-LO-45-081320_0002":0,
        "STEM_JEOL-ADF1_12-10-2020-LaFeO3-STO-092917-b-LO-zero-deg-12-8-2020_0119_1":0,
        "STEM_JEOL-ADF1_08-13-20-Wang-1-1-STO-SNO-LSAT-062620-a-LO-45-081320_0011":0,
        "STEM_ADF_09-24-18-50-nm-LMO-STO-081618-LO-091618_0001":4,
        "STEM_ADF_01-15-18-2-uc-LaMnO3-070617B-LO-100-111417-EELS_0004":3,
        "STEM_JEOL-ADF1_03-16-20-Wangoh-LSTO-STO-0.25-TEM-012020-LO-0-031020_0002":4,
        "STEM_JEOL-ADF1_06-29-20-Wangoh-LSTO-STO-0.25-TEM-012020-LO-0-031020-Thinner_0002":4,
        "STEM_ADF_09-24-18-50-nm-LMO-STO-081618-LO-091618_0006":3,
        "STEM_JEOL-ADF1_08-03-20-Wang-1-STO-1-SNO-LSAT-062620-a-LO-45-072720_0002":0,
        "STEM_JEOL-ADF1_12-10-2020-LaFeO3-STO-092917-b-LO-zero-deg-12-8-2020_0114_1":0,
        "STEM_JEOL-HAADF_04-27-17-13-nm-STO-p-Ge-033117-LO-110-042617-HAADF_0005":3,
        "STEM_ADF_09-24-18-50-nm-LMO-STO-081618-LO-091618_0003":3,
        "STEM_JEOL-ADF1_03-16-20-Wangoh-LSTO-STO-0.25-TEM-012020-LO-0-031020_0003":4,
        "STEM_ADF_01-15-18-2-uc-LaMnO3-070617B-LO-100-111417-EELS_0003":3,
        "STEM_JEOL-ADF1_08-03-20-Wang-1-STO-1-SNO-LSAT-062620-a-LO-45-072720_0003":0,
        "STEM_JEOL-ADF1_08-13-20-Wang-1-1-STO-SNO-LSAT-062620-a-LO-45-081320_0007_1":0,
        "STEM_ADF_09-24-18-50-nm-LMO-STO-081618-LO-091618_0005":3,
        "STEM_ADF_01-12-18-4-uc-LaMnO3-072817A-LO-100-01118-EELS_0006":3,
        "STEM_JEOL-ADF1_12-10-2020-LaFeO3-STO-092917-b-LO-zero-deg-12-8-2020_0107_1":0,
        "STEM_JEOL-ADF1_12-10-2020-LaFeO3-STO-092917-b-LO-zero-deg-12-8-2020_0116_1":0,
        "STEM_JEOL-ADF1_08-13-20-Wang-1-1-STO-SNO-LSAT-062620-a-LO-45-081320_0001":0,
        "STEM_JEOL-ADF1_07-23-19-Du-STO-Ge-LO-45-071919_0003":0,
        "STEM_JEOL-ADF1_12-10-2020-LaFeO3-STO-092917-b-LO-zero-deg-12-8-2020_0108_1":0,
        "STEM_JEOL-HAADF_05-02-17-5-uc-STO-p-Ge-033117-LO-110-051116-HAADF_0006":0,
        "STEM_ADF_01-15-18-2-uc-LaMnO3-070617B-LO-100-111417-EELS_0002":3,
        "STEM_JEOL-ADF1_07-23-19-Du-STO-Ge-LO-45-071919_0005":0,
        "STEM_JEOL-ADF1_08-13-20-Wang-1-1-STO-SNO-LSAT-062620-a-LO-45-081320_0004-2":0,
        "STEM_ADF_09-24-18-30-nm-LMO-STO-081317B-LO-091618_0004":3,
        "STEM_JEOL-ADF1_11-20-19-Spurgeon-60-nm-LaMnO3-STO-001-073119-LO-103019_0001":0,
        "STEM_ADF_01-12-18-4-uc-LaMnO3-072817A-LO-100-01118-EELS_0005":3,
        "STEM_ADF_01-12-18-4-uc-LaMnO3-072817A-LO-100-01118-EELS_0003":3,
        "STEM_JEOL-ADF1_12-10-2020-LaFeO3-STO-092917-b-LO-zero-deg-12-8-2020_0110_1":0,
        "STEM_JEOL-ADF1_10-12-20-La0.8Sr0.2FeO3-STO-080317-2-LO-zero-deg_0002":4,
        "STEM_JEOL-ADF1_06-29-20-Wangoh-LSTO-STO-0.25-TEM-012020-LO-0-031020-Thinner_0001":4,
        "STEM_ADF_11-02-18-10-nm-LFO-STO-A-LO-0-103118_0001":3,
        "STEM_JEOL-ADF1_10-12-20-La0.8Sr0.2FeO3-STO-080317-2-LO-zero-deg_0001_1":4,
        "STEM_JEOL-ADF1_10-12-20-La0.8Sr0.2FeO3-STO-080317-2-LO-zero-deg_0006_1":2,
        "STEM_ADF_09-24-18-30-nm-LMO-STO-081317B-LO-091618_0005":3,
        "STEM_JEOL-ADF1_12-10-2020-LaFeO3-STO-092917-b-LO-zero-deg-12-8-2020_0120_1":0,
        "STEM_JEOL-ADF1_08-13-20-Wang-1-1-STO-SNO-LSAT-062620-a-LO-45-081320_0009":0,
        "STEM_JEOL-ADF1_12-10-2020-LaFeO3-STO-092917-b-LO-zero-deg-12-8-2020_0111_1":0,
        "STEM_JEOL-ADF1_10-12-20-La0.8Sr0.2FeO3-STO-080317-2-LO-zero-deg_0003_1":3,
        "STEM_ADF_11-02-18-10-nm-LFO-STO-A-LO-0-103118_0004":3,
        "STEM_JEOL-ADF1_12-10-2020-LaFeO3-STO-092917-b-LO-zero-deg-12-8-2020_0121_1":0,
        "STEM_JEOL-ADF1_08-13-20-Wang-1-1-STO-SNO-LSAT-062620-a-LO-45-081320_0004_1":0,
        "STEM_JEOL-ADF1_11-20-19-Spurgeon-60-nm-LaMnO3-STO-001-073119-LO-103019_0003":0,
        "STEM_ADF_09-24-18-30-nm-LMO-STO-081317B-LO-091618_0008":3,
        "STEM_ADF_01-15-18-2-uc-LaMnO3-070617B-LO-100-111417-EELS_0001":3,
        "STEM_ADF_01-12-18-4-uc-LaMnO3-072817A-LO-100-01118-EELS_0007":3,
        "STEM_JEOL-HAADF_04-27-17-13-nm-STO-p-Ge-033117-LO-110-042617-HAADF_0007":3,
        "STEM_JEOL-ADF1_08-03-20-Wang-1-STO-1-SNO-LSAT-062620-a-LO-45-072720_0004_1":0,
    }
    
    #Collect all performance measures
    #key : IMAGE_ALIAS, values : [iou, recall, precision, f1, fpr]
    results = {}

    # Iterate over the dataloader to access the images and labels
    for image, filename in dataloader:

        print("processing:",filename)
        _, _, width, height = image.shape
        ROWS, COLS = int(height / IDEAL_CHIP_SIZE), int(width / IDEAL_CHIP_SIZE)
       
        IMAGE_NAME = filename[0]
        IMAGE_PATH = DATA_DIR + IMAGE_NAME
        IMAGE_ALIAS = IMAGE_NAME
        if '.' in IMAGE_ALIAS:
            IMAGE_ALIAS = IMAGE_ALIAS.split('.')[0]
        LABEL_PATH =  './CHESS_Labeling_Round_1/12-15-23_Spurgeon_CHESS_Labeling_Round_1/labels_merged/' + IMAGE_ALIAS   # Expert 1
        #LABEL_PATH = './CHESS_Labeling_Round_1/12-17-23_Doty_CHESS_Labeling_Round_1/labels_merged/' + IMAGE_ALIAS  # Expert 2
        #LABEL_PATH = './pychip_labels/' + IMAGE_ALIAS # For ICML

        #Initialize reulsts as none for this image
        results[IMAGE_ALIAS] = [None, None, None, None, None]


        torch.cuda.empty_cache()

        print('IMAGE:',IMAGE_NAME)
        image_name, image_ext = os.path.splitext(IMAGE_NAME)
        CLUSTERS = num_materials[image_name] if image_name in num_materials.keys() else 3
        print(f'{image_name}: clusters {"" if image_name in num_materials.keys() else "not"} found!')

        #Prepare logfile
        log_dir = os.path.abspath(os.getcwd()) + OUTPUT_DIR + IMAGE_ALIAS + '/'
        log_file_name = 'output'
        logger = init_logs(log_file_name, log_dir=log_dir)

        #Copy image from dataset to output file
        if COPY_ORIGINAL_DATA:
            source_file = DATA_DIR + filename[0]
            shutil.copy(source_file, log_dir)


        startTime = datetime.now()
        CHIPS, squares = chip_image(IMAGE_PATH,ROWS, COLS)
        logger.info(f'Image chipping: {datetime.now() - startTime}')

        CHIP_SIZE = squares[0][3]

        #print(squares)

        image = cv2.imread(IMAGE_PATH)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        logger.info(f'{IMAGE_ALIAS}-->{image.shape} = {ROWS} x {COLS} "\tchip_size: {CHIP_SIZE}')

        startTime = datetime.now()
        #mask_generator = SamAutomaticMaskGenerator(sam)

        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=POINTS_PER_SIDE,
            pred_iou_thresh=PRED_IOU_THRESH,
            stability_score_thresh=STABILITY_SCORE_THRESH,
            crop_n_layers=CROP_N_LAYERS,
            crop_n_points_downscale_factor=CROP_N_POINTS_DOWNSCALE_FACTOR,
            min_mask_region_area=IDEAL_CHIP_SIZE ** 2,  # Requires open-cv to run post-processing
        )

        masks = mask_generator.generate(image)
        logger.info(f'SAM {device} mask generation time: {datetime.now() - startTime}')

        #Add 'status' key to all mask dictionaries 
        for mask in masks:
            mask["status"] = "ok"

        #Post-processing mask filtering
        #Remove tiny component masks inside a compund mask if sum of compnent masks < 50% of compund mask
        #Remove compund mask if component masks makeup >= 70% of compund mask
        if args.post:
            startTime = datetime.now()
            mask_tree = {i:{"children":[], "redundant":[]} for i, _ in enumerate(masks)}
            for i, mask_1 in enumerate(masks):
                for j, mask_2 in enumerate(masks):
                    if i != j: 
                        mask_2_overlap, mask_1_overlap, intersection_overlap = mutual_overlap(mask_1["segmentation"], mask_2["segmentation"])
                        #mask2 is child/component of mask1
                        if mask_2_overlap > 90 and mask_1_overlap < 90:
                            mask_tree[i]["children"].append([j,intersection_overlap])
                        #mask2 and mask1 are highly overlapping (redundant)
                        elif mask_2_overlap > 90 and mask_1_overlap > 90:
                            mask_tree[i]["redundant"].append([j,intersection_overlap])
                logger.info(f'MASK-{i} \t Children: {mask_tree[i]["children"]}  \t Redundant: {mask_tree[i]["redundant"]}')
            
            #Remove unnecessary masks
            expendables = set()
            #Remove child if it's less than 5% of parent
            for i,v in mask_tree.items():
                for j,v in v["children"]:
                    if v < 5:
                        expendables.add(j)
                        masks[j]["status"] = "tiny"
            
            #Remove parent if > 70% is madeup of children
            for i,v in mask_tree.items():
                covered = 0
                for j,v in v["children"]:
                    if v > 5:
                        covered += v
                if covered > 70:
                    expendables.add(i)
                    masks[i]["status"] = "compound"
            
            #Remove the samller of redundant masks
            for i,v in mask_tree.items():
                for j,v in v["redundant"]:
                    j_degree = v #Degree to which mask_j overlaps/belongs to mask_i
                    for a,b in mask_tree[j]["redundant"]:
                        if a == i:
                            i_degree = b #Degree to which mask_i overlaps/belongs to with mask_j
                    expendables.add(i if j_degree > i_degree else j)
                    masks[i if j_degree > i_degree else j]["status"] = "redundant"
            
            logger.info(f'Expendables: {expendables}') 

            masks = [mask for i, mask in enumerate(masks) if i not in expendables]  

            logger.info(f"Post-processing time: {datetime.now() - startTime}")




        startTime = datetime.now()
        #Compute all valid chips for the image
        for square in squares:
            for idx,mask in enumerate(masks):
                if mask["status"] == "ok":
                    overlap = chip_mask_overlap(square,mask["segmentation"])
                    if overlap >= OVERLAP:
                        logger.info(f"Chip: {square}  x  Mask-{idx}: {mask['bbox']}   -->  {overlap:.2f}% overlap")
                        #Chips are in form (name,row,column,width,length). Convert to (name,x0,y0,x1,y1) for overlaying.
                        chip = (square[0],square[1],square[2],square[1] + square[3],square[2] + square[4])
                        if "chips" in mask.keys():
                            mask["chips"].append(chip)
                        else:
                            mask["chips"] = [chip]
        logger.info(f'Mask chip matching: {datetime.now() - startTime}')

        startTime = datetime.now()
        #Select k: where k is the sample chip size for each mask
        k = K
        for idx,mask in enumerate(masks):
            if mask["status"] == "ok":
                if 'chips' not in mask.keys():
                    mask["chips"] = []
                k_chips = []
                if len(mask["chips"]) > k:
                    k_chips = random.shuffle(mask["chips"])
                    k_chips = mask["chips"][:k]
                elif len(mask["chips"]) > 0 and len(mask["chips"]) < k:
                    k_chips = mask["chips"]
                mask["k_chips"] = k_chips
                #mask["k_chips"] = mask["chips"]
                logger.info(f"Mask-{idx}: {mask['bbox']}, area {mask['area']}   -->  {len(mask['k_chips'])} chips")
        logger.info(f'Select k chips per mask: {datetime.now() - startTime}')


        startTime = datetime.now()
        #Cut out the k chips for each mask and save them to disk
        for idx,mask in enumerate(masks):
            if mask["status"] == "ok":
                #Chips are saved as (name,x0,y0,x1,y1) for overlaying. Convert back to to (name,row,column,width,length) for chipping. 
                k_chips = {square[0] : CHIPS[square[0]] for square in mask["k_chips"]}
                save_chips(IMAGE_PATH,f"MASK-{idx}",k_chips,savepath=OUTPUT_DIR,imgs_ext='.jpg')
        logger.info(f'Save k-masks to disk: {datetime.now() - startTime}')

        #If no embedding needed, instantiate model in order to compute soft-labels
        if not args.embed:
            if args.model == "res18":
                # For ResNet18 chip classifier
                model = models.resnet18(pretrained=True)
                num_classes = 64
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                model.load_state_dict(torch.load('res18.pth'))
            
            elif args.model == "FENet":
                # For FENet chip classifier
                opt = None
                with open('FENet_DTD_opt.pth', 'rb') as file:
                    opt = pickle.load(file)
                model = FENet18(opt)
                model.load_state_dict(torch.load('FENet_DTD.pth'))

        startTime = datetime.now()
        #Encode masks
        encodings = []
        centroids = []
        encoding_dim = 64
        for idx,mask in enumerate(masks):
            #images_path = "./k_chips/" + IMAGE_PATH.split(".")[-2].split("/")[-1] + f"/MASK-{idx}"
            images_path = log_dir + f"MASK-{idx}"
            if args.embed:
                mask["encodings"] = encode_chips(images_path,args.model,layer_index=4,weights='IMAGENET1K_V1') #'IMAGENET1K_V1',"res18.pth"
                if mask["encodings"]:
                    mask["centroid"] = np.mean(mask["encodings"], axis=0)
                    encodings += mask["encodings"]
                    centroids.append(mask["centroid"])
                logger.info(f"Mask-{idx} \t encoded: {len(mask['encodings'])}")
            else:
                mask["soft_preds"] = soft_predict_chips(images_path,model)
                #Random pred augmentation/repetition for smaller masks
                if mask["soft_preds"]:
                    while len(mask["soft_preds"]) < k:
                        random_pred = random.choice(mask["soft_preds"])
                        mask["soft_preds"].append(random_pred)
                logger.info(f"Mask-{idx} \t preds: {len(mask['soft_preds'])}")
        logger.info(f"{'Encode' if args.embed else 'Soft-pred'} k-masks: {datetime.now() - startTime}")

        n_materials = CLUSTERS #len(masks)
        kmeans = KMeans(n_clusters=n_materials)
        unknown_color = np.concatenate([np.random.random(3), [0.35]])
        color_code = {'unknown':unknown_color}

        if args.post and args.embed:
            #Cluster masks based on encodings
            startTime = datetime.now()
            gmm = GaussianMixture(n_components=CLUSTERS)
            dbscan = DBSCAN(eps=0.5, min_samples=2)
            #labels_train = dbscan.fit_predict(np.array(encodings))
            labels_train = kmeans.fit(np.array(encodings)).labels_
            #gmm.fit(np.array(encodings))
            left, right = 0, 0

            #logger.info(f"Labels: {labels}")
            #Encode masks
            color_code = {}
            label_masks = {label:[] for label in labels_train}
            for idx,mask in enumerate(masks):
                mask["color"] = unknown_color
                if mask["encodings"]:
                    # print("Pred",kmeans.predict(mask["encodings"]))
                    labels = list(kmeans.predict(np.array(mask["encodings"])))
                    # right = left + len(mask["encodings"])
                    # labels = list(labels_train[left:right])
                    # left = right
                    #labels = list(gmm.predict(np.array(mask["encodings"])))
                    mask["label"] = max(set(labels), key=lambda x: labels.count(x))
                    label_masks[mask['label']].append(idx)
                    if not mask["label"] in color_code.keys():
                        color_code[mask["label"]] = np.concatenate([np.random.random(3), [0.35]])
                    
                    mask["color"] = color_code[mask["label"]]
                    logger.info(f"Mask-{idx} \t max: {mask['label']} \t labels: {labels}")
            
            logger.info(f'Labeling time: {datetime.now() - startTime}')
            # Perform dimensionality reduction using PCA
            pca = PCA(n_components=2)
            reduced_vectors = pca.fit_transform(np.array(encodings))

            # Plot the points with colors indicating labels
            plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=labels_train, cmap='viridis')
            plt.colorbar()

            # Add labels and title to the plot
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title('KMeans Clustering')
            encoding_out = log_dir + '_encodings.pdf' if args.saveformat == 'pdf' else log_dir + '_encodings.png'
            plt.draw()
            plt.savefig(encoding_out, bbox_inches='tight', pad_inches=0, format='pdf' if args.saveformat == 'pdf' else 'png')
            plt.close()

        elif args.post and not args.embed:
            startTime = datetime.now()
            kl_adj = {i:[] for i in range(len(masks))}
            for i in range(len(masks)):
                mask_i = masks[i]
                if mask["status"] == "ok" and mask_i["soft_preds"] and len(mask_i["soft_preds"]) == k:
                    for j in range(len(masks)):
                        mask_j = masks[j]
                        if mask_j["soft_preds"] and len(mask_j["soft_preds"]) == k:
                            #logger.info(f"(Mask-{i}, Mask-{j}) \t = {torch.stack(mask_i['soft_preds'])} \t {torch.stack(mask_j['soft_preds'])}")
                            if args.method == 'kl':
                                kl_div = kl(mask_i["soft_preds"],mask_j["soft_preds"])
                            elif args.method == 'emd':
                                kl_div = emd(mask_i["soft_preds"],mask_j["soft_preds"])
                            #logger.info(f"KL-div(Mask-{i}, Mask-{j}) \t = {kl_div}")        
                            kl_adj[i].append(kl_div)
                        else:
                            kl_adj[i].append(-1)

            logger.info(f'KL computation time: {datetime.now() - startTime}')

            data,names,weights = [],[],[]
            for i,v in kl_adj.items():
                if v:
                    data.append(v)
                    names.append(f'Mask-{i}')
                    weights.append(len(masks[i]['k_chips']))
                    #logger.info(f"Mask-{i} => chips {len(masks[i]['k_chips'])}")
            
            if len(data) <  2:
                continue

            #Redefine kmeans n_components
            n_materials = min(CLUSTERS,len(data)) if CLUSTERS else min(3,len(data))
            kmeans = KMeans(n_clusters=n_materials)

            # Perform dimensionality reduction using PCA
            pca = PCA(n_components=2)
            reduced_vectors = pca.fit_transform(np.array(data))
            
            labels = kmeans.fit(reduced_vectors).labels_
            centers = kmeans.cluster_centers_

            # silhouette_values = silhouette_samples(reduced_vectors, labels)
            # #silhouette_avg = silhouette_score(reduced_vectors, labels)
            # weighted_silhouette_avg = np.average(silhouette_values, weights=weights)

            # logger.info(f"Weighted Silhouette score: {weighted_silhouette_avg}")

            unknown_color = np.concatenate([np.random.random(3), [0.35]])
            unknown_mask_idxs = []
            idx = 0
            label_masks = {label:[] for label in labels}
            for i,v in kl_adj.items():
                if v:
                    masks[i]["label"] = labels[idx]
                    idx += 1
                    label_masks[masks[i]['label']].append(i)
                else:
                    masks[i]["label"] = -1
                
                if masks[i]["label"] == -1:
                    masks[i]["color"] = unknown_color
                    unknown_mask_idxs.append(i)
                elif masks[i]["label"] not in color_code.keys():
                    color_code[masks[i]["label"]] = np.concatenate([np.random.random(3), [0.35]])    
                    masks[i]["color"] = color_code[masks[i]["label"]]
                else:
                    masks[i]["color"] = color_code[masks[i]["label"]]
                
                logger.info(f"Mask-{i} \t label: {masks[i]['label']} \t vector = {v}")

            #Filter unknown mask indices
            #masks = [value for index, value in enumerate(masks) if index not in unknown_mask_idxs]

            
            # Perform dimensionality reduction using PCA
            # pca = PCA(n_components=2)
            # reduced_vectors = pca.fit_transform(np.array(data))

            
            # Generate meshgrid to plot cluster regions
            step_size = 0.01
            x_min, x_max = reduced_vectors[:, 0].min() - 0.1, reduced_vectors[:, 0].max() + 0.1
            y_min, y_max = reduced_vectors[:, 1].min() - 0.1, reduced_vectors[:, 1].max() + 0.1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))
            Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            # Plot the points with colors indicating labels
            plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=labels, cmap='viridis')
            plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
            plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100)
            plt.colorbar()

            for i, txt in enumerate(names):
                plt.annotate(txt, (reduced_vectors[i][0], reduced_vectors[i][1]),fontsize=8)

            # Add labels and title to the plot
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title('KMeans Clustering')
            encoding_out = log_dir + '_encodings.pdf' if args.saveformat == 'pdf' else log_dir + '_encodings.png'
            plt.draw()
            plt.savefig(encoding_out, bbox_inches='tight', pad_inches=0, format='pdf' if args.saveformat == 'pdf' else 'png')
            plt.close()

        #Post-processing merge masks with similar labels 
        if args.post and len(masks) > CLUSTERS:
            expendables, merged, merged_labels = [], [], []
            for label,mask_ids in label_masks.items():
                if len(mask_ids) > 1:
                    expendables += mask_ids
                    submasks = [masks[i] for i in mask_ids]
                    merged.append(merge_masks(submasks))
                    merged_labels.append(label)
                    for mask in submasks:
                        mask["status"] = "component"
            
            #masks = [mask for i, mask in enumerate(masks) if i not in expendables]
            #labels = [label for i, label in enumerate(labels) if i not in expendables]
            masks += merged
            labels = np.append(labels,merged_labels)
        elif len(masks) <= CLUSTERS:
            print(f'Avoided merging since masks: {len(masks)} <= clusters: {CLUSTERS}')

        #Performance measure
        file_list = [file for file in os.listdir(LABEL_PATH) if file.endswith(".png")]

        num_gt_labels = len(file_list)

        # Iterate through the list of .png files and load them as PIL Image objects
        avg_iou, avg_pre, avg_rec, avg_f1, total_fpr = 0, 0, 0, 0, 0
        for file_name in file_list:
            file_path = os.path.join(LABEL_PATH, file_name)
            ground_truth_mask = Image.open(file_path)
            # Convert the PIL image to a NumPy array
            mask_array = np.array(ground_truth_mask)

            if len(mask_array.shape) > 2:
                mask_array = mask_array[:,:,1]

            img_area = mask_array.shape[0] * mask_array.shape[1]

            # Convert the grayscale values to binary (0 or 1)
            mask_array = (mask_array > 0).astype(np.uint8)

            label_area = np.sum(mask_array == 1)
            logger.info(f"Label area : {label_area} \t Image area : {img_area}")

            # Optionally, you can display the NumPy array
            logger.info(f'Label {file_name} \t {mask_array.shape}')
            false_positives = []
            max_mask_size, max_mask_id, max_iou, max_rec, max_pre, max_f1 = 0,0,0,0,0,0
            ious, recs, pres, f1s = [], [], [], []
            for i,mask in enumerate(masks):
                if mask["status"] == "ok":
                    mask_pred = mask["segmentation"]
                    mask_area = mask["area"]
                    # Convert the grayscale values to binary (0 or 1)
                    mask_pred = (mask_pred > 0).astype(np.uint8)
                    if  exceeds_iou_threshold(mask_pred,mask_array, IOU_THRESH): #intersects(mask_pred,mask_array):
                        #Convert to boundary mask and recompute metrics if dilation < 1
                        if DILATION < 1:
                            mask_array = mask_to_boundary(mask_array,DILATION)
                            mask_pred = mask_to_boundary(mask_pred,DILATION)
                        #Compute metrics
                        iou, rec, pre, f1 = calculate_metrics(mask_pred,mask_array)
                        logger.info(f'MASK-{i} iou={round(iou, 4):<20} rec={round(rec, 4):<20} pre={round(pre, 4):<20} f1={round(f1,4):<20} size={max_mask_size}')
                        false_positives.append(mask_pred)
                        ious.append(iou)
                        recs.append(rec)
                        pres.append(pre)
                        f1s.append(f1)
                        if iou > max_iou:
                            false_positives.pop()
                            max_iou = iou
                            max_rec = rec
                            max_pre = pre
                            max_f1 = f1
                            max_mask_id = i
                            max_mask_size = mask_area
            mean_iou = sum(ious) / len(ious) if ious else 0
            mean_rec = sum(recs) / len(recs) if recs else 0
            mean_pre = sum(pres) / len(pres) if pres else 0
            mean_f1 = sum(f1s) / len(f1s) if f1s else 0

            fpr = compute_fpr(false_positives,mask_array)

            #logger.info(f'WINNER -> MASK-{max_mask_id} iou = {round(max_iou,4):<20} ')
            logger.info(f'MASK-{max_mask_id} m_iou={round(mean_iou, 4):<20} m_rec={round(mean_rec, 4):<20} m_pre={round(mean_pre, 4):<20} m_f1={round(mean_f1,4):<20} label_fpr={round(fpr,4):<20} size={max_mask_size}')
            if args.par and mixin_coeef[image_name]:
                numerator = mixin_coeef[image_name][file_name]
                denominator = sum(mixin_coeef[image_name].values())
                print(f'Using mixin coeef: {file_name} = {numerator}/{denominator}')
            else:
                numerator = 1
                denominator = num_gt_labels
            avg_iou += (numerator / denominator) * mean_iou * 100
            avg_pre += (numerator / denominator) * mean_pre * 100
            avg_rec += (numerator / denominator) * mean_rec * 100
            avg_f1 += (numerator / denominator) * mean_f1 * 100
            total_fpr += (numerator / denominator) * fpr * 100 

        
        logger.info(f'SAM Avg IOU                 : {round(avg_iou,2)}%')
        logger.info(f'SAM Avg PRECISION           : {round(avg_pre,2)}%')
        logger.info(f'SAM Avg RECALL              : {round(avg_rec,2)}%')
        logger.info(f'SAM Avg F1 SCORE            : {round(avg_f1,2)}%')
        logger.info(f'SAM Avg FALSE POSITIVE RATE : {round(total_fpr,2)}%')

        results[IMAGE_ALIAS] = [round(avg_iou,2), round(avg_rec,2), round(avg_pre,2), round(avg_f1,2), round(total_fpr,2)]

        plt.figure(figsize=(20,20))
        plt.imshow(image)
        show_anns([mask for mask in masks if mask["status"] == "ok"])
        plt.axis('off')
        for idx, mask in enumerate(masks):
            if mask["status"] == "ok":
                bbox = (mask["bbox"][0],mask["bbox"][1],mask["bbox"][0] + mask["bbox"][2],mask["bbox"][1] + mask["bbox"][3])
                show_box(bbox, plt.gca(),f"Mask-{idx}",mask["color"] if "color" in mask else [])
                show_points(np.array(mask["point_coords"]), np.array([1]), plt.gca())
        
        if args.post:
            #Legend
            # Create empty handles and labels lists
            handles = []
            labels = []
            # Iterate over the colors and create proxy artists
            for i, color in color_code.items():
                # Create a rectangle patch as the proxy artist
                rect = plt.Rectangle((0, 0), 1, 1, color=color)
                handles.append(rect)
                labels.append(f'Cluster-{i}')
            plt.legend(handles, labels, loc='upper left')
 
        sam_out = log_dir + '_sam.pdf' if args.saveformat == 'pdf' else log_dir + '_sam.png'
        plt.draw()
        plt.savefig(sam_out, bbox_inches='tight', pad_inches=0, format='pdf' if args.saveformat == 'pdf' else 'png')
        plt.close()

        for i,_ in enumerate(masks):
            if mask["status"] == "ok":
                plt.figure(figsize=(20,20))
                plt.imshow(image)
                show_mask(masks[i]["segmentation"], plt.gca())
                #show_anns(masks)
                plt.axis('off')
                for chip in masks[i]["chips"]:
                    show_box(chip[1:], plt.gca())
                bbox = (masks[i]["bbox"][0],masks[i]["bbox"][1],masks[i]["bbox"][0] + masks[i]["bbox"][2],masks[i]["bbox"][1] + masks[i]["bbox"][3])
                show_box(bbox, plt.gca())
                show_points(np.array(masks[i]["point_coords"]), np.array([1]), plt.gca())
                iname = IMAGE_PATH.split(".")[-2].split("/")[-1] + "/" + f"MASK-{i}" + "/"
                directory_path = os.path.abspath(os.getcwd()) + OUTPUT_DIR + iname
                mask_name = f"MASK-{i}.jpg"
                plt.draw()
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
                plt.savefig(directory_path + mask_name)
                plt.close()
    
    #Summarize results into its own dictionary
    results["SUMMARY"] = summarize_results(results)

    #Write results to file
    titles = ["IMAGE", "IOU", "RECALL", "PRECISION", "F1", "FPR"]
    spacing = [100,10,10,10,10,10]
    write_to_file(results, titles, os.path.abspath(os.getcwd()) + OUTPUT_DIR + 'options.log',spacing)