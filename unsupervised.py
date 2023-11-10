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

if __name__ == '__main__':

    args = get_args()

    # segment = [[False,False,False,False,False,False],
    #            [False,False,True,True,True,False],
    #            [False,False,True,True,True,False],
    #            [False,False,True,True,True,False],
    #            [False,False,True,True,False,False],
    #            [False,False,False,False,False,False]]
    # #print(get_chips(segment,2,2))

    # integral_image = compute_integral_image(np.array(segment))
    # chips = get_mask_chips(integral_image, 2)
    # print(chips)

    ROWS, COLS = 12, 12
    OVERLAP = 80
    K = 5
    N_MATERIALS = 6
    #IMAGE_PATH = './STO_GE_2.jpg' #"./decompressed/STO_GE_2.f32.cuszx.50.png"
    #IMAGE_PATH = './STEM_JEOL_ADF2_08-17-20_Yano_Cr2O3_18O_070820_20F017_0004_2.tiff'
    #IMAGE_PATH = './STEM_JEOL_ADF2_12-17-20_Hematite_1_Unirrad_Uncapped19F070_LO_121719_Thinned_Followup_0017_2.jpg'
    #IMAGE_PATH = './STEM_JEOL_BF_12-17-20_Hematite_1_Unirrad_Uncapped19F070_LO_121719_Thinned_Followup_0007.jpg'
    IMAGE_PATH = './STEM_JEOL_ADF2_12-17-20_Hematite_1_Unirrad_Uncapped19F070_LO_121719_Thinned_Followup_0017_22.jpg'
    #IMAGE_PATH = './STEM_JEOL_BF_08-13-20_Wang_1-1_STO-SNO_LSAT_062620-a_LO_45_081320_0008_2.tiff'
    #IMAGE_PATH = 'STEM_JEOL_ADF1_04-12-21_112820_La0.03Sr0.97Zr0.5Ti0.5O3_Ge_12_nm_LO_040521_0006.tiff'
    IMAGE_ALIAS = IMAGE_PATH.replace('\\','')
    IMAGE_ALIAS = IMAGE_PATH.replace('./','')
    if ' ' in IMAGE_ALIAS:
        IMAGE_ALIAS = IMAGE_ALIAS.split()[0]
    elif '.' in IMAGE_ALIAS:
        IMAGE_ALIAS = IMAGE_ALIAS.split('.')[0]
    EXT = '.png'
    SAVE_PATH = '/sample_chips/'
    LABEL_PATH = './labels/' + IMAGE_ALIAS

    logger = get_logger()

    torch.cuda.empty_cache()


    startTime = datetime.now()
    CHIPS, squares = chip_image(IMAGE_PATH,ROWS, COLS)
    print("Image chipping:", datetime.now() - startTime)

    CHIP_SIZE = squares[0][3]

    #print(squares)

    image = cv2.imread(IMAGE_PATH)
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

        print("Post-processing time:", datetime.now() - startTime)

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
    print("Mask chip matching:", datetime.now() - startTime)

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
    print("Select k chips per mask:", datetime.now() - startTime)


    startTime = datetime.now()
    #Cut out the k chips for each mask and save them to disk
    for idx,mask in enumerate(masks):
        if mask["status"] == "ok":
            #Chips are saved as (name,x0,y0,x1,y1) for overlaying. Convert back to to (name,row,column,width,length) for chipping. 
            k_chips = {square[0] : CHIPS[square[0]] for square in mask["k_chips"]}
            save_chips(IMAGE_PATH,f"MASK-{idx}",k_chips,savepath='/k_chips/',imgs_ext='.jpg')
    print("Save k-chips to disk:", datetime.now() - startTime)

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
        images_path = "./k_chips/" + IMAGE_PATH.split(".")[-2].split("/")[-1] + f"/MASK-{idx}"
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
    print(f"{'Encode' if args.embed else 'Soft-pred'} k-masks: {datetime.now() - startTime}")

    n_materials = N_MATERIALS #len(masks)
    kmeans = KMeans(n_clusters=n_materials)
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
        unknown_color = np.concatenate([np.random.random(3), [0.35]])
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
        
        print("Labeling time:", datetime.now() - startTime)
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
        plt.savefig(IMAGE_ALIAS + "_encodings.png")

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

        print("KL computation time:", datetime.now() - startTime)

        data,names,weights = [],[],[]
        for i,v in kl_adj.items():
            if v:
                data.append(v)
                names.append(f'Mask-{i}')
                weights.append(len(masks[i]['k_chips']))
                #logger.info(f"Mask-{i} => chips {len(masks[i]['k_chips'])}")
        
        n_materials = min(N_MATERIALS,len(data)) #len(masks)
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
        color_code = {}
        label_masks = {label:[] for label in labels}
        for i,v in kl_adj.items():
            if v:
                masks[i]["label"] = labels[idx]
                idx += 1
                #Collect masks belonging to known labels
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
        plt.savefig(IMAGE_ALIAS + "_encodings.png")

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
        
        logger.info(f'\t  = MASK-{max_mask_id} iou = {round(max_iou,4):<20} ')
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


    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns([mask for mask in masks if mask["status"] == "ok"])
    plt.axis('off')
    for idx, mask in enumerate(masks):
        if mask["status"] == "ok":
            bbox = (mask["bbox"][0],mask["bbox"][1],mask["bbox"][0] + mask["bbox"][2],mask["bbox"][1] + mask["bbox"][3])
            show_box(bbox, plt.gca(),f"Mask-{idx}",mask["color"])
            show_points(np.array(mask["point_coords"]), np.array([1]), plt.gca())
    plt.savefig(IMAGE_ALIAS + "_sam.png")

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
            directory_path = os.path.abspath(os.getcwd()) + '/k_chips/' + iname
            mask_name = f"MASK-{i}.jpg"
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            plt.savefig(directory_path + mask_name)

