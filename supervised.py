import torch
import matplotlib.pyplot as plt
import cv2
import torch.quantization
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import torchvision.models as models
from models.FENet18 import Net as FENet18
import random
import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from datetime import datetime
import pickle
from utils import *

if __name__ == '__main__':

    args = get_args()

    ROWS, COLS = 50, 50
    IMAGE_PATH = './STO_GE_2.jpg' #"./decompressed/STO_GE_2.f32.cuszx.50.png"
    EXT = '.png'
    SAVE_PATH = '/sample_chips/'

    logger = get_logger()

    startTime = datetime.now()
    CHIPS, squares = chip_image(IMAGE_PATH,ROWS, COLS)
    print("Image chipping:", datetime.now() - startTime)

    CHIP_SIZE = squares[0][3]

    #print(squares)

    torch.cuda.empty_cache()


    image = cv2.imread('STO_GE_2.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sys.path.append(".")
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE:", device)

    startTime = datetime.now()
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    print("SAM",device,"model loading time:", datetime.now() - startTime)

    startTime = datetime.now()
    mask_generator = SamAutomaticMaskGenerator(sam)

    # mask_generator = SamAutomaticMaskGenerator(
    #     model=sam,
    #     points_per_side=2,
    #     pred_iou_thresh=0.86,
    #     stability_score_thresh=0.92,
    #     crop_n_layers=1,
    #     crop_n_points_downscale_factor=2,
    #     min_mask_region_area=100,  # Requires open-cv to run post-processing
    # )

    masks = mask_generator.generate(image)

    print("SAM",device,"mask generation time:", datetime.now() - startTime)

    startTime = datetime.now()
    #Compute all valid chips for the image
    for square in squares:
        for idx,mask in enumerate(masks):
            overlap = chip_mask_overlap(square,mask["segmentation"])
            if overlap >= 90:
                logger.info(f"Chip: {square}  x  Mask-{idx}: {mask['bbox']}   -->  {overlap:.2f}% overlap")
                #Chips are in form (name,row,column,width,length). Convert to (name,x0,y0,x1,y1) for overlaying.
                chip = (square[0],square[1],square[2],square[1] + square[3],square[2] + square[4])
                if "chips" in mask.keys():
                    mask["chips"].append(chip)
                else:
                    mask["chips"] = [chip]
    print("Mask chip matching:", datetime.now() - startTime)

    startTime = datetime.now()
    #Select k: where k is the sample chip size for each mask
    k = 5
    for idx,mask in enumerate(masks):
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
        logger.info(f"Mask-{idx}: {mask['bbox']}   -->  {len(mask['k_chips'])} chips")
    print("Select k chips per mask:", datetime.now() - startTime)


    startTime = datetime.now()
    #Cut out the k chips for each mask and save them to disk
    for idx,mask in enumerate(masks):
        #Chips are saved as (name,x0,y0,x1,y1) for overlaying. Convert back to to (name,row,column,width,length) for chipping. 
        k_chips = {square[0] : CHIPS[square[0]] for square in mask["k_chips"]}
        save_chips(IMAGE_PATH,f"MASK-{idx}",k_chips,savepath='/k_chips/',imgs_ext='.jpg')
    print("Save k-masks to disk:", datetime.now() - startTime)

    torch.cuda.empty_cache()

    startTime = datetime.now()
    #Encode masks
    encodings = []
    centroids = []
    encoding_dim = 64

    if args.model == "res18":
        # For ResNet18 chip classifier
        model = models.resnet18()
        num_classes = 3
        model.fc = torch.nn.Flatten() #nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load('Resnet18_80.0%.pth'))
    
    elif args.model == "res50":
        # For ResNet18 chip classifier
        model = models.resnet50()
        num_classes = 3
        model.fc = torch.nn.Flatten() #nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load('Resnet50_87.5%.pth'))
    
    elif args.model == "FENet":
        # For FENet chip classifier
        opt = None
        with open('opt_pred.pth', 'rb') as file:
            opt = pickle.load(file)
        model = FENet18(opt)
        model.load_state_dict(torch.load('FENet_pred.pth'))
    
    #elif args.model == 

    for idx,mask in enumerate(masks):
        images_path = "./k_chips/" + IMAGE_PATH.split(".")[-2].split("/")[-1] + f"/MASK-{idx}"
        mask["preds"] = predict_chips(images_path,model,trans='FENet' if args.model == "FENet" else 'resnet')
        logger.info(f"Mask-{idx} \t preds: {len(mask['preds'])}")
    print("Predict k-masks:", datetime.now() - startTime)

    #Cluster masks based on encodings
    startTime = datetime.now()
    left, right = 0, 0

    #logger.info(f"Labels: {labels}")
    #Encode masks
    unknown_color = np.concatenate([np.random.random(3), [0.35]])
    color_code = {}
    for idx,mask in enumerate(masks):
        mask["color"] = unknown_color
        if mask["preds"]:
            mask["label"] = max(set(mask["preds"]), key=lambda x: mask["preds"].count(x))
            if not mask["label"] in color_code.keys():
                color_code[mask["label"]] = np.concatenate([np.random.random(3), [0.35]])
            
            mask["color"] = color_code[mask["label"]]
            logger.info(f"Mask-{idx} \t max: {mask['label']} \t labels: {mask['preds']}")

    print("Coloring time:", datetime.now() - startTime)

    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    for idx, mask in enumerate(masks):
        #if idx == 2:
        bbox = (mask["bbox"][0],mask["bbox"][1],mask["bbox"][0] + mask["bbox"][2],mask["bbox"][1] + mask["bbox"][3])
        show_box(bbox, plt.gca(),f"Mask-{idx}",mask["color"])
        show_points(np.array(mask["point_coords"]), np.array([1]), plt.gca())
    plt.savefig("STO_GE_2_sam.png")

    for i,_ in enumerate(masks):
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
        directory_path = os.path.abspath(os.getcwd()) + '/k_chips/' + iname
        mask_name = f"MASK-{i}.jpg"
        plt.savefig(directory_path + mask_name)

