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
BATCH_SIZE = 1
SUBSET_SIZE = 2
IDEAL_CHIP_SIZE = 60
DATA_DIR = './chessdataset_limited/'
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

    args = get_args()

    OPTIONS = {
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
    }

    os.makedirs(os.path.dirname(os.path.abspath(os.getcwd()) + OUTPUT_DIR), exist_ok=True)

    #Write OPTIONS to file
    titles = ["OPTIONS", "VALUES"]
    spacing = [40,20]
    write_to_file(OPTIONS, titles, os.path.abspath(os.getcwd()) + OUTPUT_DIR + 'options.log',spacing)

    logger = get_logger()

    #sys.path.append(".")
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
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
    num_materials = {'STEM_ADF_03-07-19_20_nm_MnFe2O4_MAO_110618_1_LO_013019_0010.tiff':3,
                     'STEM_ADF_05-08-17_Fe3-xCrxO4_MgO_No_3_041817_LO_100_050417_0002.tiff':2,
                     'STEM_ADF_05-08-17_Fe3-xCrxO4_MgO_No_3_041817_LO_100_050417_0015.tiff':5,
                     'STEM_ADF_07-10-17_30_nm_SrFeOx_LSAT_050517_LO_100_062217_0004.tiff':5,
                     'STEM_ADF_07-10-17_30_nm_SrFeOx_LSAT_050517_LO_100_062217_0005.tiff':4,
                     'STEM_ADF_07-10-17_30_nm_SrFeOx_LSAT_050517_LO_100_062217_0008.tiff':2,
                     'STEM_ADF_09-24-18_50_nm_LMO_STO_081618_LO_091618_0003.tiff':4,
                     'STEM_ADF_11-02-18_10_nm_LFO-STO_A_LO_0_103118_0003.tiff':3,
                     'STEM_ADF_11-02-18_10_nm_LFO-STO_A_LO_0_103118_0004.tiff':3,
                     'STEM_ADF_12-03-18_10_nm_LFO-STO_120717-a_LO_0_103018_0009.tiff':3,
                     'STEM_ADF_12-07-18_10_nm_LFO_STO_120717-A_LO_0_103018_0002.tiff':3,
                     'STEM_ADF_12-07-18_10_nm_LFO_STO_120717-B_LO_0_120518_0008.tiff':3,
                     'STEM_ADF_12-07-18_10_nm_LFO_STO_120717-C_LO_0_110818_0002.tiff':3,
                     'STEM_ADF_12-07-18_10_nm_LFO_STO_120717-C_LO_0_110818_0004.tiff':3,
                     'STEM_ADF_12-07-18_10_nm_LFO_STO_120717-C_LO_0_110818_0012.tiff':3,
                     'STEM_JEOL_ADF1_02-05-20_Wang_5-1_LFO_SNO_LSAT_120718-6_LO_0_020420_0001.tiff':4,
                     'STEM_JEOL_ADF1_02-12-20_Wang_1-1_LFO_SNO_LSAT_011619-a_LO_0_020420_0001.tiff':5,
                     'STEM_JEOL_ADF1_03-08-21_Kaspar_30_Cr2O3_30_Fe3O4_012621B_LO_21F010_0001_1.tiff':6,
                     'STEM_JEOL_ADF1_03-16-20_Wangoh_LSTO_STO_0-25_TEM_012020_LO_0_031020_0004_1.tiff':2,
                     'STEM_JEOL_ADF1_04-12-21_112820_La0-03Sr0-97Zr0-5Ti0-5O3_Ge_12_nm_LO_040521_0009_1.tiff':3,
                     'STEM_JEOL_ADF1_04-15-19_50_nm_Fe2TiO4_MgO_012419_LO_030519_Futher_Polish_0015.tiff':4,
                     'STEM_JEOL_ADF1_04-15-19_50_nm_Fe2TiO4_MgO_012419_LO_030519_Futher_Polish_0027.tiff':4,
                     'STEM_JEOL_ADF1_05-17-21_Le_LNFO_LSAT_375_After_OER_102120-c_LO_051021_0002_1.tiff':4,
                     'STEM_JEOL_ADF1_05-17-21_Le_LNFO_LSAT_375_After_OER_102120-c_LO_051021_0003_1.tiff':3,
                     'STEM_JEOL_ADF1_05-17-21_Le_LNFO_LSAT_375_After_OER_102120-c_LO_051021_0005_1.tiff':4,
                     'STEM_JEOL_ADF1_06-04-19_12_nm_NiMn2O4_MAO_020519_1_LO_030519_0001.tiff':5,
                     'STEM_JEOL_ADF1_06-04-19_12_nm_NiMn2O4_MAO_020519_1_LO_030519_0005.tiff':6,
                     'STEM_JEOL_ADF1_06-11-19_13-7_nm_CoMn2O4_MAO_021519_1_LO_022619_0005.tiff':4,
                     'STEM_JEOL_ADF1_06-11-19_13-7_nm_CoMn2O4_MAO_021519_1_LO_022619_0007.tiff':4,
                     'STEM_JEOL_ADF1_07-06-20_Kaspar_Hematite_1_Unirrad_Uncapped_121719_19F070_0001.tiff':3,
                     'STEM_JEOL_ADF1_08-13-20_Wang_1-1_STO-SNO_LSAT_062620-a_LO_45_081320_0009.tiff':3,
                     'STEM_JEOL_ADF1_08-17-20_Yano_Cr2O3_18O_070820_20F017_0001_1.tiff':5,
                     'STEM_JEOL_ADF1_08-17-20_Yano_Cr2O3_18O_070820_20F017_0005_1.tiff':5,
                     'STEM_JEOL_ADF1_09-04-19_STO-Ge_070919_LO_EMSL_090419_0005.tiff':4,
                     'STEM_JEOL_ADF1_09-05-19_STO-Ge_070919_LO_EMSL_090419_Thinned_0001.tiff':3,
                     'STEM_JEOL_ADF1_09-05-19_STO-Ge_070919_LO_EMSL_090419_Thinned_0003.tiff':5,
                     'STEM_JEOL_ADF1_10-09-19_Fe2TiO4_062619_LO_091819_0011.tiff':3,
                     'STEM_JEOL_ADF1_11-20-19_Scafetta_Fe2CrO4_MAO_050318_600_C_LO__111819_0004.tiff':4,
                     'STEM_JEOL_ADF1_11-20-19_Spurgeon_60_nm_LaMnO3_STO_001_073119_LO_103019_0007.tiff':4,
                     'STEM_JEOL_ADF1_12-10-2020-LaFeO3-STO-092917-b-LO-zero-deg-12-8-2020_0105.tiff':3,
                     'STEM_JEOL_ADF1_12-10-2020-LaFeO3-STO-092917-b-LO-zero-deg-12-8-2020_0110_1.tiff':4,
                     'STEM_JEOL_ADF1_12-10-2020-LaFeO3-STO-092917-b-LO-zero-deg-12-8-2020_0112_1.tiff':3,
                     'STEM_JEOL_ADF1_12-10-2020-LaFeO3-STO-092917-b-LO-zero-deg-12-8-2020_0114_1.tiff':2,
                     'STEM_JEOL_ADF1_12-10-2020-LaFeO3-STO-092917-b-LO-zero-deg-12-8-2020_0117_1.tiff':3,
                     'STEM_JEOL_ADF1_12-10-2020-LaFeO3-STO-092917-b-LO-zero-deg-12-8-2020_0118_1.tiff':2,
                     'STEM_JEOL_ADF1_12-17-20_Hematite_1_Unirrad_Uncapped19F070_LO_121719_Thinned_Followup_0010_1.tiff':2,
                     'STEM_JEOL_ADF2_02-12-20_Wang_1-1_LFO_SNO_LSAT_011619-a_LO_0_020420_0006.tiff':4,
                     'STEM_JEOL_ADF2_02-12-20_Wang_5-1_LFO_SNO_LSAT_120718-6_LO_0_020420_0008.tiff':4,
                     'STEM_JEOL_ADF2_02-12-20_Wang_5-1_LFO_SNO_LSAT_120718-6_LO_0_020420_0010.tiff':5,
                     'STEM_JEOL_ADF2_03-16-20_Wangoh_LSTO_STO_0-25_TEM_012020_LO_0_031020_0004.tiff':2,
                     'STEM_JEOL_ADF2_04-12-21_112820_La0-03Sr0-97Zr0-5Ti0-5O3_Ge_12_nm_LO_040521_0001_1.tiff':3,
                     'STEM_JEOL_ADF2_04-15-19_50_nm_Fe2TiO4_MgO_012419_LO_030519_Futher_Polish_0019.tiff':4,
                     'STEM_JEOL_ADF2_05-17-21_Le_LNFO_LSAT_375_After_OER_102120-c_LO_051021_0001_2.tiff':3,
                     'STEM_JEOL_ADF2_06-03-19_40_nm_Fe2CrO4_MAO_041118_LO_021819_0022.tiff':3,
                     'STEM_JEOL_ADF2_06-06-19_20_nm_MnFe2O4_MAO_110618_1_LO_013019_0006.tiff':5,
                     'STEM_JEOL_ADF2_06-11-19_13-7_nm_CoMn2O4_MAO_021519_1_LO_022619_0010.tiff':3,
                     'STEM_JEOL_ADF2_06-11-19_13-7_nm_CoMn2O4_MAO_021519_1_LO_022619_0012.tiff':3,
                     'STEM_JEOL_ADF2_07-03-19_12_nm_NiMn2O4_MAO_020519_1_LO_030519_Longer_Bake_0006.tiff':4,
                     'STEM_JEOL_ADF2_07-03-19_12_nm_NiMn2O4_MAO_020519_1_LO_030519_Longer_Bake_0012.tiff':3,
                     'STEM_JEOL_ADF2_07-06-20_Kaspar_Hematite_1_Unirrad_Uncapped_121719_19F070_0005_1.tiff':4,
                     'STEM_JEOL_ADF2_07-06-20_Kaspar_Hematite_1_Unirrad_Uncapped_121719_19F070_0006_1.tiff':3,
                     'STEM_JEOL_ADF2_07-06-20_Kaspar_Hematite_Growth_2_LO_90_0004_1.tiff':2,
                     'STEM_JEOL_ADF2_07-23-19_Du_STO-Ge_LO_45_071919_0013.tiff':3,
                     'STEM_JEOL_ADF2_08-03-20_Wang_1-STO_1-SNO_LSAT_062620-a_LO_45_072720_0003_1.tiff':4,
                     'STEM_JEOL_ADF2_08-13-20_Wang_1-1_STO-SNO_LSAT_062620-a_LO_45_081320_0003_2.tiff':4,
                     'STEM_JEOL_ADF2_09-04-19_STO-Ge_070919_LO_EMSL_090419_0006.tiff':3,
                     'STEM_JEOL_ADF2_10-09-19_Fe2TiO4_062619_LO_091819_0008.tiff':5,
                     'STEM_JEOL_ADF2_10-09-19_Fe2TiO4_062619_LO_091819_0010.tiff':3,
                     'STEM_JEOL_ADF2_10-09-19_Fe2TiO4_062619_LO_091819_0012.tiff':3,
                     'STEM_JEOL_ADF2_10-09-19_Fe2TiO4_062619_LO_091819_0016.tiff':3,
                     'STEM_JEOL_ADF2_10-12-20_La0-8Sr0-2FeO3-STO-080317-2-LO-zero-deg_0003_2.tiff':3,
                     'STEM_JEOL_ADF2_10-12-20_La0-8Sr0-2FeO3-STO-080317-2-LO-zero-deg_0005_2.tiff':2,
                     'STEM_JEOL_ADF2_11-20-19_Spurgeon_60_nm_LaMnO3_STO_001_073119_LO_103019_0012.tiff':3,
                     'STEM_JEOL_ADF2_12-04-20_100_nm_Fe3O4_10_nm_Cr2O3_Al2O3_111120B_LO_120120_0004_2.tiff':5,
                     'STEM_JEOL_ADF2_12-04-20_100_nm_Fe3O4_10_nm_Cr2O3_Al2O3_111120B_LO_120120_0006_2.tiff':5,
                     'STEM_JEOL_ADF2_12-04-20_100_nm_Fe3O4_10_nm_Cr2O3_Al2O3_111120B_LO_120120_0009_2.tiff':3,
                     'STEM_JEOL_ADF2_12-10-2020-LaFeO3-STO-092917-b-LO-zero-deg-12-8-2020_0116_2.tiff':3,
                     'STEM_JEOL_ADF2_12-10-2020-LaFeO3-STO-092917-b-LO-zero-deg-12-8-2020_0118_2.tiff':2,
                     'STEM_JEOL_ADF2_12-17-20_Hematite_1_Unirrad_Uncapped19F070_LO_121719_Thinned_Followup_0001_2.tiff':6,
                     'STEM_JEOL_ADF2_12-17-20_Hematite_1_Unirrad_Uncapped19F070_LO_121719_Thinned_Followup_0003_2.tiff':5,
                     'STEM_JEOL_ADF2_12-17-20_Hematite_1_Unirrad_Uncapped19F070_LO_121719_Thinned_Followup_0012_2.tiff':5,
                     'STEM_JEOL_ADF2_12-17-20_Hematite_1_Unirrad_Uncapped19F070_LO_121719_Thinned_Followup_0013_2.tiff':4,
                     'STEM_JEOL_ADF2_12-17-20_Hematite_1_Unirrad_Uncapped19F070_LO_121719_Thinned_Followup_0017_2.tiff':5,
                     'STEM_JEOL_ADF2_12-17-20_Hematite_1_Unirrad_Uncapped19F070_LO_121719_Thinned_Followup_0017_2.jpg':5,
                     'STO_GE_2.jpg':3,
                     'STEM_JEOL_BF_03-08-21_Kaspar_30_Cr2O3_30_Fe3O4_012621A_LO_21F010_0002.tiff':6,
                     'STEM_JEOL_BF_03-08-21_Kaspar_30_Cr2O3_30_Fe3O4_012621A_LO_21F010_0007.tiff':2,
                     'STEM_JEOL_BF_08-13-20_Wang_1-1_STO-SNO_LSAT_062620-a_LO_45_081320_0004_1_2.tiff':3,
                     'STEM_JEOL_BF_08-13-20_Wang_1-1_STO-SNO_LSAT_062620-a_LO_45_081320_0007.tiff':3,
                     'STEM_JEOL_BF_08-13-20_Wang_1-1_STO-SNO_LSAT_062620-a_LO_45_081320_0008_2.tiff':3,
                     'STEM_JEOL_BF_08-13-20_Wang_1-1_STO-SNO_LSAT_062620-a_LO_45_081320_0010.tiff':2,
                     'STEM_JEOL_BF_08-13-20_Wang_1-1_STO-SNO_LSAT_062620-a_LO_45_081320_0011_1.tiff':3,
                     'STEM_JEOL_BF_12-04-20_100_nm_Fe3O4_10_nm_Cr2O3_Al2O3_111120B_LO_120120_0001.tiff':5,
                     'STEM_JEOL_BF_12-04-20_100_nm_Fe3O4_10_nm_Cr2O3_Al2O3_111120B_LO_120120_0004.tiff':5,
                     'STEM_JEOL_BF_12-04-20_100_nm_Fe3O4_10_nm_Cr2O3_Al2O3_111120B_LO_120120_0005.tiff':5,
                     'STEM_JEOL_BF_12-04-20_100_nm_Fe3O4_10_nm_Cr2O3_Al2O3_111120B_LO_120120_0010.tiff':3,
                     'STEM_JEOL_BF_12-10-2020-LaFeO3-STO-092917-b-LO-zero-deg-12-8-2020_0110.tiff':4,
                     'STEM_JEOL_BF_12-10-2020-LaFeO3-STO-092917-b-LO-zero-deg-12-8-2020_0116.tiff':3,
                     'STEM_JEOL_BF_12-17-20_Hematite_1_Unirrad_Uncapped19F070_LO_121719_Thinned_Followup_0007.tiff':5,
                     'STEM_JEOL_BF_12-17-20_Hematite_1_Unirrad_Uncapped19F070_LO_121719_Thinned_Followup_0013.tiff':5,
                     'STEM_JEOL_HAADF_04-27-17_13_nm_STO_p-Ge_033117_LO_110_042617_HAADF_0007.tiff':3,
                     'STEM_JEOL_HAADF_05-02-17_5_uc_STO_p-Ge_033117_LO_110_051116_HAADF_0007.tiff':3
                     }

    #Collect all performance measures
    #key : IMAGE_ALIAS, values : [iou, recall, precision, f1, fpr]
    results = {}

    # Iterate over the dataloader to access the images and labels
    for image, filename in dataloader:
        _, _, width, height = image.shape
        ROWS, COLS = int(height / IDEAL_CHIP_SIZE), int(width / IDEAL_CHIP_SIZE)
       
        IMAGE_NAME = filename[0]
        IMAGE_PATH = DATA_DIR + IMAGE_NAME
        IMAGE_ALIAS = IMAGE_NAME
        if '.' in IMAGE_ALIAS:
            IMAGE_ALIAS = IMAGE_ALIAS.split('.')[0]
        LABEL_PATH = './labels/' + IMAGE_ALIAS

        #Initialize reulsts as none for this image
        results[IMAGE_ALIAS] = [None, None, None, None, None]


        torch.cuda.empty_cache()

        print('IMAGE:',IMAGE_NAME)
        CLUSTERS = num_materials[IMAGE_NAME] if IMAGE_NAME in num_materials.keys() else 0

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
        #Remove compund mask if component masks makeup >= 50% of compund mask
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

            logger.info("Post-processing time:", datetime.now() - startTime)



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

        n_materials = 3 #len(masks)
        kmeans = KMeans(n_clusters=n_materials)
        unknown_color = np.concatenate([np.random.random(3), [0.35]])
        color_code = {'unknown':unknown_color}

        if args.embed:
            #Cluster masks based on encodings
            startTime = datetime.now()
            gmm = GaussianMixture(n_components=n_materials)
            dbscan = DBSCAN(eps=0.5, min_samples=2)
            #labels_train = dbscan.fit_predict(np.array(encodings))
            labels_train = kmeans.fit(np.array(encodings)).labels_
            #gmm.fit(np.array(encodings))
            left, right = 0, 0

            #logger.info(f"Labels: {labels}")
            #Encode masks
            color_code = {}
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
            encoding_out = log_dir + '_encodings.png'
            plt.draw()
            plt.savefig(encoding_out)
            plt.close()

        else:
            startTime = datetime.now()
            kl_adj = {i:[] for i in range(len(masks))}
            for i in range(len(masks)):
                mask_i = masks[i]
                if mask["status"] == "ok" and mask_i["soft_preds"] and len(mask_i["soft_preds"]) == k:
                    for j in range(len(masks)):
                        mask_j = masks[j]
                        if mask_j["soft_preds"] and len(mask_j["soft_preds"]) == k:
                            #logger.info(f"(Mask-{i}, Mask-{j}) \t = {torch.stack(mask_i['soft_preds'])} \t {torch.stack(mask_j['soft_preds'])}")
                            kl_div = kl(mask_i["soft_preds"],mask_j["soft_preds"])
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
            encoding_out = log_dir + '_encodings.png'
            plt.draw()
            plt.savefig(encoding_out)
            plt.close()

            #Post-processing merge masks with similar labels
            if args.post:
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


        #Performance measure
        file_list = [file for file in os.listdir(LABEL_PATH) if file.endswith(".png")]
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
            for i,mask in enumerate(masks):
                if mask["status"] == "ok":
                    mask_pred = mask["segmentation"]
                    mask_area = mask["area"]
                    iou, rec, pre, f1 = calculate_metrics(mask_pred,mask_array)
                    #logger.info(f'\tMASK-{i} iou={round(iou,2)},\t acc={round(acc,2)},\t rec={round(rec,2)},\t pre={round(pre,2)},\t f1={f1}')
                    logger.info(f'MASK-{i} iou={round(iou, 4):<20} rec={round(rec, 4):<20} pre={round(pre, 4):<20} f1={round(f1,4):<20} size={max_mask_size}')
                    if iou > 0:
                        false_positives.append(mask_pred)
                    if iou > max_iou:
                        false_positives.pop()
                        max_iou = iou
                        max_rec = rec
                        max_pre = pre
                        max_f1 = f1
                        max_mask_id = i
                        max_mask_size = mask_area
            fpr = compute_fpr(false_positives,mask_array)
            
            #logger.info(f'WINNER -> MASK-{max_mask_id} iou = {round(max_iou,4):<20} ')
            logger.info(f'MAX MASK-{max_mask_id} iou={round(max_iou, 4):<20} rec={round(max_rec, 4):<20} pre={round(max_pre, 4):<20} f1={round(max_f1,4):<20} label_fpr={round(fpr,4):<20} size={max_mask_size}')
            avg_iou += (label_area / img_area) * max_iou * 100
            avg_pre += (label_area / img_area) * max_pre * 100
            avg_rec += (label_area / img_area) * max_rec * 100
            avg_f1 += (label_area / img_area) * max_f1 * 100
            total_fpr += (label_area / img_area) * fpr * 100 
        
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
                show_box(bbox, plt.gca(),f"Mask-{idx}",mask["color"])
                show_points(np.array(mask["point_coords"]), np.array([1]), plt.gca())
        
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
        
        
        sam_out = log_dir + '_sam.png'
        plt.draw()
        plt.savefig(sam_out)
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