import logging
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import math
import torchvision.transforms as transforms
import pandas as pd
import os
import shutil
import json
import torch
import torch.nn.functional as F

import torchvision.models as models
from models.FENet18 import Net as FENet18
import torch.nn as nn
import argparse
import pickle
from torch.utils.data import Dataset, DataLoader

def get_logger():
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    return logger

def init_logs(log_file_name, log_dir=None):
    
    #mkdirs(log_dir)
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    log_path = log_file_name + '.log'

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)


    logging.basicConfig(
        filename=os.path.join(log_dir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    return logger

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sam', type=str, default='sam_vit_h_4b8939.pth', help='SAM checkpoint.pth file')
    parser.add_argument('--grid', type=int, default=12, help='Prompt grid density')
    parser.add_argument('--chipsize', type=int, default=60, help='Desired chip size')
    parser.add_argument('--model', type=str, default='res18', help='Neural network used for encoding')
    parser.add_argument('--embed', type=int, default=1, help='Embed chips to a latent space vector')
    parser.add_argument('--post', type=int, default=0, help='Apply post processing to remove bad masks')
    parser.add_argument('--par', type=int, default=0, help='Apply mixin_coeef if available for Par computation')
    parser.add_argument('--method', type=str, default='kl', help='Method for comparing chip preds (kl, emd)')
    parser.add_argument('--dilation', type=float, default=1, help='Dilation value for creating boundary masks. Value=1 doesn\'t create a boundary mask. (0,1]')
    parser.add_argument('--iou_thresh', type=float, default=0.05, help='IoU threshold for considering preds as belonging to ground truth label')
    parser.add_argument('--saveformat', type=str, default='png', help='File format for output images  (png, pdf)')

    args = parser.parse_args()
    return args

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = ann["color"] if "color" in ann.keys() else np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax, label="",linecolor=[]):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='black', facecolor=(0,0,0,0), lw=5))
    if label:
        # Add label to the box
        label_x = x0 + 0.5 * w  # x-coordinate of the label position
        label_y = y0 + 0.5 * h  # y-coordinate of the label position
        ax.text(label_x, label_y, label, fontsize=20, color='red',
                ha='center', va='center')

def weighted_average(values, weights):
    weighted_sum = sum(value * weight for value, weight in zip(values, weights))
    total_weight = sum(weights)
    return weighted_sum / total_weight

def merge_bboxs(masks):
    if not masks:
        return None

    min_x = min(mask[0] for mask in masks)
    min_y = min(mask[1] for mask in masks)
    max_x = max(mask[0] + mask[2] for mask in masks)
    max_y = max(mask[1] + mask[3] for mask in masks)

    merged_mask = [min_x, min_y, max_x - min_x, max_y - min_y]
    return merged_mask

def merge_masks(masks):
    mask_segmentations = [mask["segmentation"] for mask in masks]
    mask_predicted_ious = [mask["predicted_iou"] for mask in masks]
    mask_areas = [mask["area"] for mask in masks]
    mask_bboxs = [mask["bbox"] for mask in masks]
    mask_point_coords = [mask["point_coords"] for mask in masks]
    mask_stability_scores = [mask["stability_score"] for mask in masks]
    mask_crop_boxes = [mask["crop_box"] for mask in masks]
    

    
    union_predicted_iou = weighted_average(mask_predicted_ious, mask_areas) 
    union_area = sum(mask_areas)
    union_point_coords = mask_point_coords[0]
    union_stability_score = weighted_average(mask_stability_scores, mask_areas)
    union_bbox = merge_bboxs(mask_bboxs)
    union_crop_box = mask_crop_boxes[0]
    union_color = masks[0]["color"]
    union_chips = [mask["chips"] for mask in masks]
    union_chips = [item for sublist in union_chips for item in sublist]
    union_segmentation = np.logical_or.reduce(mask_segmentations)
    union_mask = {"predicted_iou":union_predicted_iou,
                    "segmentation":union_segmentation,
                    "area":union_area, "bbox":union_bbox,
                    "point_coords":union_point_coords,
                    "stability_score":union_stability_score,
                    "crop_box":union_crop_box,"color":union_color,
                    "chips":union_chips,"status":"ok"}
    return union_mask

def mutual_overlap(parent_mask, child_mask):
    intersection_mask = np.logical_and(parent_mask, child_mask)
    inter_inter_child = np.logical_and(intersection_mask, child_mask)
    inter_inter_parent = np.logical_and(intersection_mask, parent_mask)

    intersection_pixels = np.sum(intersection_mask)

    if not intersection_pixels:
        return 0, 0, 0

    child_pixels = np.sum(child_mask)
    parent_pixels = np.sum(parent_mask)
    
    inter_inter_child_pixels = np.sum(inter_inter_child)
    inter_inter_parent_pixels = np.sum(inter_inter_parent)

    child_overlap = inter_inter_child_pixels / child_pixels * 100
    parent_overlap = inter_inter_parent_pixels / parent_pixels * 100
    intersection_overlap = intersection_pixels / parent_pixels * 100
    
    # False, False -> Unrelated masks
    # True, False -> component mask
    # False, True -> reverse component (i.e. parent is the component of child)
    # True, True -> redundant masks

    return child_overlap, parent_overlap, intersection_overlap

def compute_iou(mask_pred, mask_gt):
    intersection = np.logical_and(mask_pred, mask_gt)
    union = np.logical_or(mask_pred, mask_gt)
    
    iou = np.sum(intersection) / np.sum(union)
    return iou

def exceeds_iou_threshold(predicted_mask, ground_truth_mask, thresh):
    intersection = np.logical_and(predicted_mask, ground_truth_mask)
    union = np.logical_or(predicted_mask, ground_truth_mask)
    iou = np.sum(intersection) / np.sum(union)

    return iou > thresh

def intersects(predicted_mask, ground_truth_mask):
    intersection = np.logical_and(predicted_mask, ground_truth_mask)
    return np.any(intersection)

def calculate_metrics(predicted_mask, ground_truth_mask):
    intersection = np.logical_and(predicted_mask, ground_truth_mask)
    union = np.logical_or(predicted_mask, ground_truth_mask)
    true_positives = np.sum(np.logical_and(predicted_mask, ground_truth_mask))
    #true_negatives = np.sum(np.logical_and(np.logical_not(predicted_mask), np.logical_not(ground_truth_mask)))
    false_positives = np.sum(np.logical_and(predicted_mask, np.logical_not(ground_truth_mask)))
    false_negatives = np.sum(np.logical_and(np.logical_not(predicted_mask), ground_truth_mask))
    
    iou = np.sum(intersection) / np.sum(union)
    recall = true_positives / (true_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    #fpr = false_positives / (false_positives + true_negatives)
    f1 = 2 * (precision * recall) / (precision + recall) if precision * recall != 0 else 0
    
    return iou, recall, precision, f1

def compute_fpr(predicted_masks, ground_truth_mask):
    fpr = 0
    for predicted_mask in predicted_masks:
        intersection = np.logical_and(predicted_mask, ground_truth_mask)
        union = np.logical_or(predicted_mask, ground_truth_mask)
        iou = np.sum(intersection) / np.sum(union)
        fpr += iou
    return fpr

def customize_model(model,layer_index=7):
    #Remove the output layer of the model to directly output feature vectors
    if layer_index == -1:
        new_model = torch.nn.Sequential(*list(model.children())[:-1])
    else:
        new_model = torch.nn.Sequential(*list(model.children())[:layer_index])    
    return new_model

def clean_dataset_filenames(data_dir):
    """Clean a dataset directory by replacing
    all the spaces (' ') in the file names 
    contained in the directory with underscores.

    Args:
        data_dir (_type_): _description_
    """
    for filename in os.listdir(data_dir):
        if filename.endswith('.tiff'):
            new_filename = filename.replace(' ', '_')
            original_file_path = os.path.join(data_dir, filename)
            new_file_path = os.path.join(data_dir, new_filename)
            os.rename(original_file_path, new_file_path)

def cv2_to_PIL(cv2_img):
    """ Auxilliary method to convert the cv2 images into PIL
    images.

    Parameters:
        cv2_img: (a cv2 image) (required)
            This parameter is the cv2 image that wants to be converted
            into a PIL image.
    Return:
        PIL_img: (a PIL image)
            The cv2 image converted into a PIL image."""

    PIL_img = transforms.ToTensor()(Image.fromarray(255 * cv2_img.astype(np.uint8)))

    # to just return the input cv2_img for debugging:
    # cv_img = transforms.ToTensor()(cv2_img)
    return PIL_img

def summarize_results(results):
    summary = {}

    # Calculate averages for each value
    num_values = len(next(iter(results.values())))  # Get the number of values per key
    for values in results.values():
        for index, value in enumerate(values):
            if index not in summary:
                summary[index] = 0
            summary[index] += value

    # Divide the summed values by the number of keys to get averages
    for index in summary:
        summary[index] /= len(results)

    # Create the final summary dictionary
    summary_dict = [round(summary[i],2) for i in range(num_values)]

    return summary_dict

def write_to_file(dictionary, titles, file_path, spacing):
    # Compute the number of stars
    stars = sum(spacing)
    with open(file_path, 'a+') as file:
        # Write the column header
        header = [f"{title:<{space}}" for title,space in zip(titles,spacing)] 
        header = "".join(header) + '\n'
        
        file.write(header)

        # Write the top border
        top_border = '*' * stars + '\n'
        file.write(top_border)

        # Iterate through the key-value pairs in the dictionary
        for key, value in dictionary.items():
            if isinstance(value, list):
                first_line = f"{key:<{spacing[0]}}"
                line = [f"{val:<{space}}" for val,space in zip(value,spacing[1:])] 
                line = "".join(line) + '\n'
                line = first_line + line
                #line = f"{key:<{spacing}}{value}\n"
            else:
                # Format the line as an f-string with key and value aligned in 'spacing=50' characters
                line = f"{key:<{spacing[0]}}{value}\n"
            # Write the line to the file
            file.write(line)

        # Write the bottom border
        bottom_border = '*' * stars + '\n'
        file.write(bottom_border)

def chip_image(img_path,num_rows, num_cols):
    pil_image = Image.open(img_path)
    img_og = np.asarray(pil_image)
    # the torch models we use require the input to have 3 channels. if there is only 1, we make more:
    img = np.asarray(pil_image)
    if len(img_og.shape) == 2:
        for_torch = np.zeros((img_og.shape[0], img_og.shape[1], 3))
        for_torch[:, :, 0] = img_og  # same value in each channel
        for_torch[:, :, 1] = img_og
        for_torch[:, :, 2] = img_og
        img = for_torch
    
    img_shape = img_og.shape
    height = img_shape[0]
    width = img_shape[1]
    chip_size = math.floor(width / num_cols)
    print("Chip size:",chip_size,"x",chip_size)
    num_chips_x = num_cols  # math.floor(width / chip_size)
    num_chips_y = num_rows #math.floor(height / chip_size)
    pixels_ignored_in_x = width % chip_size
    pixels_ignored_in_y = height % math.floor(height / num_rows) #chip_size
    x_coords = list(range(0, width, chip_size))
    y_coords = list(range(0, height, math.floor(height / num_rows))) #chip_size))
    grid_points = []

    # Filling the grid_points. This is used when creating the chips.
    for col_idx, x_coord in enumerate(range(0, width - pixels_ignored_in_x, chip_size)):
        for row_idx, y_coord in enumerate(range(0, height - pixels_ignored_in_y,chip_size)):
            grid_points.append((x_coord, y_coord, row_idx, col_idx))

    chips = {}
    squares = []
    # Creating the chips
    # self.grid_points.append((x_coord, y_coord, row_idx, col_idx))
    for x, y, R_idx, C_idx in grid_points:
        name = f"R{R_idx}C{C_idx}"  # Naming convention for the chips
        chips[name] = img[y:y + chip_size, x:x + chip_size]
        squares.append([name,x,y,chip_size,chip_size])

    return chips, squares

def predict_chips(images_path,model,trans='FENet'):
    model.eval()
    # Define the image transformation pipeline
    # Define the transformation for image preprocessing

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) if trans == 'FENet' else transforms.Compose([
        #transforms.Resize(224),
        transforms.CenterCrop(60),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_files = [os.path.join(images_path, file) for file in os.listdir(images_path) if file.endswith('.jpg') or file.endswith('.png')]

    # Empty list to store the predictions
    predictions = []

    # Iterate over the image files and make predictions
    for image_file in image_files:
        # Load and preprocess the image
        image = Image.open(image_file).convert('RGB')
        image = transform(image)
        image = torch.unsqueeze(image, 0)

        # Make the prediction
        with torch.no_grad():
            output = model(image)
            _, predicted_idx = torch.max(output, 1)
            predictions.append(predicted_idx.item())

    return predictions

class CropDataset(Dataset):
    def __init__(self, directories, transform=None):
        self.files = []
        self.transform = transform

        # Load images starting with 'R' from the given directories
        for directory in directories:
            for file in os.listdir(directory):
                if file.startswith('R') and file.endswith('.jpg'):
                    self.files.append(os.path.join(directory, file))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image

def encode_chips(images_path,model,layer_index=-1,weights='IMAGENET1K_V1'):
    encodings = []
    if model == "res18":
        if weights not in ['IMAGENET1K_V1']:
            model = models.resnet18()
            # Modify the last fully connected layer to match the number of classes in your dataset
            model.fc = nn.Linear(model.fc.in_features, 64)
            model.load_state_dict(torch.load(weights))
        else:
            model = customize_model(models.resnet18(weights=weights),layer_index=7)
    elif model == "res50":
        model = customize_model(models.resnet50(weights=weights),layer_index=7)
    elif model == "FENet":
        opt = None
        with open('opt.pth', 'rb') as file:
            opt = pickle.load(file)
        model = FENet18(opt)
        model.load_state_dict(torch.load('FENet.pth'))
        model.fc = model.fc[:-2]
    else:
        raise Exception(f"Encoder model {model} not supported!")

    # Define the transformation for preprocessing the images
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # dataset = CropDataset([images_path], transform=transform)
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    for filename in os.listdir(images_path):
        if ".jpg" in filename:
            image_path = os.path.join(images_path, filename)
            image = Image.open(image_path).convert('RGB')
            image = transform(image)
            image = torch.unsqueeze(image, 0)
            
            # Load and preprocess the image
            # image = Image.open(image_path).convert("RGB")
            # image_tensor = transform(image).unsqueeze(0)
            
            # Obtain the feature representation of the image
            with torch.no_grad():
                encoding = model(image.to(device))
                encoding = encoding.view(encoding.size(0), -1)

            
            # Append the features to the list
            encodings.append(encoding.squeeze().tolist())
    return encodings

def soft_predict_chips(images_path,model):
    model.eval()
    # Define the image transformation pipeline
    # Define the transformation for image preprocessing
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_files = [os.path.join(images_path, file) for file in os.listdir(images_path) if file.endswith('.jpg') or file.endswith('.png')]

    # Empty list to store the predictions
    soft_labels = []

    # Iterate over the image files and make predictions
    for image_file in image_files:
        # Load and preprocess the image
        image = Image.open(image_file).convert('RGB')
        image = transform(image)
        image = torch.unsqueeze(image, 0)

        # Make the prediction
        with torch.no_grad():
            output = model(image)
            #_, predicted_idx = torch.max(output, 1)
            #predictions.append(predicted_idx.item())
            soft_labels.append(output)

    return soft_labels
    
def kl(pred_a, pred_b):
    assert len(pred_a) == len(pred_b), "Input lists must have the same length."

    # Convert the input lists to tensors
    a_tensor = torch.stack(pred_a) #F.log_softmax(torch.stack(pred_a),dim=1)
    b_tensor = torch.stack(pred_b) #F.log_softmax(torch.stack(pred_b),dim=1)

    # Instantiate the KLDivLoss module
    kl_loss = nn.KLDivLoss(reduction='batchmean',log_target=True)

    # Compute the KL-divergence
    kl_div = kl_loss(a_tensor, b_tensor)
    
    return kl_div.item()

def save_chips(img_path,mask_path,chips,savepath='/k_chips/',imgs_ext='.jpg'):
    # Creating the query_set (the chips to be classified)
    img_chips = []
    for chip_name in chips:
        # print('chips genesis, before cv2_to tensor, size:', self.chips[chip_name].shape)
        image = cv2_to_PIL(chips[chip_name])
        # print('chips genesis, after cv2_to tensor, size:', image.size())
        img_chips.append([image, 'no-label-yet', chip_name])

    # Turning the query_set into a pandas dataframe.
    query_set = pd.DataFrame(img_chips, columns=['images', 'labels', 'filename'])

    count = 0

    # Saving the chips if desired
    if savepath:
        iname = img_path.split(".")[-2].split("/")[-1] + "/" + mask_path + "/"
        directory_path = os.path.abspath(os.getcwd()) + savepath + iname
        # Delete the existing directory
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)
        os.makedirs(directory_path, exist_ok=True)
        for name in chips:
            save_path = os.path.join(directory_path, name + imgs_ext)
            img = np.array(chips[name])
            status = cv2.imwrite(save_path, img)
            count += 1 if status else 0
    return count

def chip_bbox_overlap(square, bounding_box):
    _, square_x, square_y, square_length = square
    bb_x0, bb_y0, w, l  = bounding_box

    # Calculate the coordinates of the bottom right corner of the square
    square_x1 = square_x + square_length
    square_y1 = square_y + square_length
    bb_x1 = bb_x0 + w
    bb_y1 = bb_y0 + l

    #Return False if no overlap between chip and bbox
    if square_x1 < bb_x0 or square_x > bb_x1 or square_y1 < bb_y0 or square_y > bb_y1:
        return False
    
    # Calculate the intersection region coordinates
    x1 = max(square_x, bb_x0)
    y1 = max(square_y, bb_y0)
    x2 = min(square_x1, bb_x1)
    y2 = min(square_y1, bb_y1)

    # Calculate the width and length of the intersection region
    w2 = max(0, x2 - x1)
    l2 = max(0, y2 - y1)

    # Return the intersection region as a tuple
    intersection = (x1, y1, w2, l2)
    return intersection

def chip_mask_overlap(square,mask):
    _, square_y0, square_x0, square_width, square_length = square
    square_x1 = square_x0 + square_width
    square_y1 = square_y0 + square_length
    
    submatrix_data = mask[square_x0:square_x1, square_y0:square_y1]  # Extract the submatrix
    count = np.count_nonzero(submatrix_data)  # Count the number of True values
    return count * 100 / (square_length ** 2)
 

def get_chips(matrix, window_size):
    n = len(matrix)
    m = window_size
    filtered_submatrices = []

    for i in range(n - m + 1):
        for j in range(n - m + 1):
            submatrix = [row[j:j+m] for row in matrix[i:i+m]]  # Extract submatrix as a list
            if all(all(cell for cell in row) for row in submatrix):  # Check if all values are True
                filtered_submatrices.append([i,j,m])

    return filtered_submatrices

def compute_integral_image(matrix):
    n = len(matrix)
    integral_image = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            if i == 0 and j == 0:
                integral_image[i, j] = matrix[i, j]
            elif i == 0:
                integral_image[i, j] = integral_image[i, j-1] + matrix[i, j]
            elif j == 0:
                integral_image[i, j] = integral_image[i-1, j] + matrix[i, j]
            else:
                integral_image[i, j] = (
                    integral_image[i-1, j] + integral_image[i, j-1]
                    - integral_image[i-1, j-1] + matrix[i, j]
                )

    return integral_image

def get_mask_chips(integral_image, window_size):
    n = len(integral_image)
    m = window_size
    chips = []

    for i in range(m-1, n):
        for j in range(m-1, n):
            if i == m-1 and j == m-1:
                submatrix_sum = integral_image[i, j]
            elif i == m-1:
                submatrix_sum = integral_image[i, j] - integral_image[i, j-m]
            elif j == m-1:
                submatrix_sum = integral_image[i, j] - integral_image[i-m, j]
            else:
                submatrix_sum = (
                    integral_image[i, j] - integral_image[i, j-m]
                    - integral_image[i-m, j] + integral_image[i-m, j-m]
                )

            if submatrix_sum == m**2:
                chips.append([i-1,j-1])

    return chips

# General util function to get the boundary of a binary mask.
def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def write_mask(mask):
    # Assuming you have a matrix called 'matrix'
    matrix = np.array(mask["segmentation"])  # Replace with your actual matrix data

    # Specify the file path
    file_path = 'output.txt'

    # Save the matrix to a file with evenly spaced values
    np.savetxt(file_path, matrix, delimiter=' ', fmt='%.2f')