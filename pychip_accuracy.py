from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import csv
from utils import *
from batch_unsupervised import ChessDataset

DATA_DIR = './chessdataset_pychip/'
SUBSET_SIZE = 1
BATCH_SIZE = 1
CHIP_SIZE = 60 #90
THRESHOLD_HIGH = 50
THRESHOLD_LOW = 20

def is_bad(overlap):
    if overlap < THRESHOLD_HIGH:# and overlap > THRESHOLD_LOW:
        return True
    return False


 # Define the transformation(s) you want to apply to the images
transform = transforms.Compose([
    transforms.ToTensor(),
    # Add more transformations as needed
])

# Instantiate the custom dataset
dataset = ChessDataset(DATA_DIR, transform=transform)

# Instantiate the DataLoader with the subset dataset
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

GE_gt = Image.open('pychip_labels/STO_GE_2/GE.png')
# Convert the PIL image to a NumPy array
ge_mask_array = np.array(GE_gt)

if len(ge_mask_array.shape) > 2:
    ge_mask_array = ge_mask_array[:,:,1]

# Convert the grayscale values to binary (0 or 1)
ge_mask_array = (ge_mask_array > 0).astype(np.uint8)

STO_gt = Image.open('pychip_labels/STO_GE_2/STO.png')
# Convert the PIL image to a NumPy array
sto_mask_array = np.array(STO_gt)

if len(sto_mask_array.shape) > 2:
    sto_mask_array = sto_mask_array[:,:,1]

# Convert the grayscale values to binary (0 or 1)
sto_mask_array = (sto_mask_array > 0).astype(np.uint8)

PtC_gt = Image.open('pychip_labels/STO_GE_2/PtC.png')
# Convert the PIL image to a NumPy array
ptc_mask_array = np.array(PtC_gt)

if len(ptc_mask_array.shape) > 2:
    ptc_mask_array = ptc_mask_array[:,:,1]

# Convert the grayscale values to binary (0 or 1)
ptc_mask_array = (ptc_mask_array > 0).astype(np.uint8)

print(f'STO_mask: {sto_mask_array.shape}')
print(f'GE_mask: {ge_mask_array.shape}')
print(f'PtC_mask: {ptc_mask_array.shape}')



# Read ground truth file
#csv_file = 'truth_df_swapped.csv'
gt_csv_file = 'STO_GE_2_truth_df.csv'
pred_csv_file = 'results_ensemble.csv'


square_truth_labels = {}
square_preds = {}
square_counts = {"STO":0,"GE":0,"PtC":0}

# Read the CSV file
with open(gt_csv_file, 'r') as file:
    reader = csv.DictReader(file)

    for row in reader:
        square_truth_labels[row['chips']] = row['class']
        square_counts[row['class']] += 1

with open(pred_csv_file, 'r') as file:
    reader = csv.DictReader(file)

    for row in reader:
        if row['prediction'] == 'set_3_label':
            square_preds[row['name']] = 'PtC'
        elif row['prediction'] == 'particle':
            square_preds[row['name']] = 'STO'
        elif row['prediction'] == 'set_2_label':
            square_preds[row['name']] = 'GE' 

# Iterate over the dataloader to access the images and labels
for image, filename in dataloader:
    IMAGE_NAME = filename[0]
    IMAGE_PATH = DATA_DIR + IMAGE_NAME
    _, _, width, height = image.shape
    ROWS, COLS = int(height / CHIP_SIZE), int(width / CHIP_SIZE)
    CHIPS, squares = chip_image(IMAGE_PATH,ROWS, COLS)

    header = "{:<4} {:<10} {:<40} {:<13} {:<13} {:<13} {:<13}".format("idx", "GT", "chip", "STO_overlap", "GE_overlap", "PtC_overlap","Prediction")
    print(header)

    gt_sto_accuracy, gt_ge_accuracy, gt_ptc_accuracy = [],[],[]
    sto_chip_count, ge_chip_count, ptc_chip_count = 0,0,0
    pred_sto_accuracy, pred_ge_accuracy, pred_ptc_accuracy = [],[],[]
    gt_bad_chips, pred_bad_chips, missed_chip = {}, {}, {}
    for idx,square in enumerate(squares):

        chip = (square[0],square[1],square[2],square[1] + square[3],square[2] + square[4])

        sto_overlap = round(chip_mask_overlap(square,sto_mask_array),2)
        ge_overlap = round(chip_mask_overlap(square,ge_mask_array),2)
        ptc_overlap = round(chip_mask_overlap(square,ptc_mask_array),2)

        square_truth = square_truth_labels[square[0]] if square[0] in square_truth_labels else None
        square_pred = square_preds[square[0]] if square[0] in square_preds else None

        if square_truth != square_pred:
            missed_chip[square[0]] = chip

        if square_truth == 'STO':
            gt_sto_accuracy.append(sto_overlap)
            if is_bad(sto_overlap): gt_bad_chips[square[0]] = chip
            if square_pred == 'STO': sto_chip_count += 1
        elif square_truth == 'GE':
            gt_ge_accuracy.append(ge_overlap)
            if is_bad(ge_overlap): gt_bad_chips[square[0]] = chip
            if square_pred == 'GE': ge_chip_count += 1
        elif square_truth == 'PtC':
            gt_ptc_accuracy.append(ptc_overlap)
            if is_bad(ptc_overlap): gt_bad_chips[square[0]] = chip
            if square_pred == 'PtC': ptc_chip_count += 1
        
        if square_pred == 'STO':
            pred_sto_accuracy.append(sto_overlap)
            if is_bad(sto_overlap): pred_bad_chips[square[0]] = chip
        elif square_pred == 'GE':
            pred_ge_accuracy.append(ge_overlap)
            if is_bad(ge_overlap): pred_bad_chips[square[0]] = chip
        elif square_pred == 'PtC':
            pred_ptc_accuracy.append(ptc_overlap)
            if is_bad(ptc_overlap): pred_bad_chips[square[0]] = chip
        

        print(f'{idx:<4} {square_truth:<10} {str(square):<40} {sto_overlap:<13} {ge_overlap:<13} {ptc_overlap:<13} {square_pred:<13}')

# Calculate mean and standard deviation
sto_mean, sto_sd = np.mean(gt_sto_accuracy), np.std(gt_sto_accuracy)
ge_mean, ge_sd = np.mean(gt_ge_accuracy), np.std(gt_ge_accuracy)
ptc_mean, ptc_sd = np.mean(gt_ptc_accuracy), np.std(gt_ptc_accuracy)

sto_chip_acc = round((sto_chip_count / square_counts['STO']) * 100, 2)
ge_chip_acc = round((ge_chip_count / square_counts['GE']) * 100, 2)
ptc_chip_acc = round((ptc_chip_count / square_counts['PtC']) * 100, 2)



# Format and print the output
output = f"{'Overalp Acc%':<15} {'STO':<15} {'GE':<15} {'PtC':<15}\n" \
         f"{'Mean':<15} {sto_mean:<15.3f} {ge_mean:<15.3f} {ptc_mean:<15.3f}\n" \
         f"{'SD':<15} {sto_sd:<15.3f} {ge_sd:<15.3f} {ptc_sd:<15.3f}"

print("###############################################################")
print("#################### PYCHIP GROUND TRUTH ######################")
print(output)


# Calculate mean and standard deviation
sto_mean, sto_sd = np.mean(pred_sto_accuracy), np.std(pred_sto_accuracy)
ge_mean, ge_sd = np.mean(pred_ge_accuracy), np.std(pred_ge_accuracy)
ptc_mean, ptc_sd = np.mean(pred_ptc_accuracy), np.std(pred_ptc_accuracy)


# Format and print the output
output = f"{'Overlap Acc%':<15} {'STO':<15} {'GE':<15} {'PtC':<15}\n" \
         f"{'Mean':<15} {sto_mean:<15.3f} {ge_mean:<15.3f} {ptc_mean:<15.3f}\n" \
         f"{'SD':<15} {sto_sd:<15.3f} {ge_sd:<15.3f} {ptc_sd:<15.3f}\n" \
         f"{'Correct chips %':<15} {sto_chip_acc:<15.3f} {ge_chip_acc:<15.3f} {ptc_chip_acc:<15.3f}" \
         

print("###############################################################")
print("#################### PYCHIP PREDICTION ########################")
print(output)


image = image.squeeze(0)  # Remove the batch dimension
image = image[0]  # Remove the batch dimension

plt.figure(figsize=(20,20))
plt.imshow(image)
plt.axis('off')
for idx,chip in gt_bad_chips.items():
    show_box(chip[1:], plt.gca())
plt.draw()
plt.savefig(f"gt_overlay.jpg")
plt.close()

plt.figure(figsize=(20,20))
plt.imshow(image)
plt.axis('off')
for idx,chip in pred_bad_chips.items():
    show_box(chip[1:], plt.gca())

for idx,chip in missed_chip.items():
    show_box(chip[1:], plt.gca(),linecolor='red')

plt.draw()
plt.savefig(f"pred_overlay.jpg")
plt.close()

