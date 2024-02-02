

import os

folder_path = "CHESS_Labeling_Round_1/12-17-23_Doty_CHESS_Labeling_Round_1/dataset"

num_materials = {}

# Iterate through the files in the folder
for filename in os.listdir(folder_path):
    # Check if the file is a .jpg image
    if filename.endswith(".jpg"):
        # Replace spaces with hyphens in the filename
        new_filename = filename.replace(".2", "-")
        new_filename = new_filename.replace(".8", "-")
        #new_filename = new_filename.replace(".", "-")
        
        # Construct the full paths for the old and new filenames
        old_filepath = os.path.join(folder_path, filename)
        new_filepath = os.path.join(folder_path, new_filename)

        num_materials[new_filename] = 0
        
        # Rename the file
        os.rename(old_filepath, new_filepath)

# print('{')
# for k,v in num_materials.items():
#     print(f'{k}:{v},')
# print('}')

folder_path = 'CHESS_Labeling_Round_1/12-17-23_Doty_CHESS_Labeling_Round_1/labels_merged'

# Iterate through the subfolders in the main folder
for subfolder in os.listdir(folder_path):
    subfolder_path = os.path.join(folder_path, subfolder)
    
    # Check if the item in the main folder is a subfolder
    if os.path.isdir(subfolder_path):
        # Replace spaces with hyphens in the subfolder name
        new_subfolder_name = subfolder.replace(".2", "-")
        new_subfolder_name = new_subfolder_name.replace(".8", "-")
        
        # Construct the full paths for the old and new subfolder names
        old_subfolder_path = os.path.join(folder_path, subfolder)
        new_subfolder_path = os.path.join(folder_path, new_subfolder_name)
        
        # Rename the subfolder
        os.rename(old_subfolder_path, new_subfolder_path)




# import csv

# effort = {'easy': [], 'medium': [], 'hard': []}

# with open('CHESS_Labeling_Round_1/12-15-23_Spurgeon_CHESS_Labeling_Round_1/12-15-23 Spurgeon Labeling Notes.csv', 'r') as csvfile:
#     csvreader = csv.reader(csvfile)
#     header = next(csvreader)  # Skip the header

#     for row in csvreader:
#         if len(row) >= 5 and row[3]:  # Ensure there are at least 5 columns in the row
#             value_4 = int(row[3])
#             value_5 = int(row[4])

#             if value_4 in [1, 2]:
#                 effort['easy'].append(value_5)
#             elif value_4 == 3:
#                 effort['medium'].append(value_5)
#             elif value_4 in [4, 5]:
#                 effort['hard'].append(value_5)

# # Print or use the 'effort' dictionary as needed
# print('effort = {')
# for k,v in effort.items():
#     print(f'"{k}":{v},')          
# print('}')





# import csv

# """STEM_JEOL_ADF1_10-12-20_La0.8Sr0.2FeO3-STO-080317-2-LO-zero-deg_0004_1"""

# num_materials = {
#                     'STEM_ADF_11-02-18_10_nm_LFO-STO_A_LO_0_103118_0002': 3,
#                     'STEM_JEOL_HAADF_04-27-17_13_nm_STO_p-Ge_033117_LO_110_042617_HAADF_0001': 4,
#                     'STEM_JEOL_ADF1_02-20-20_Yingge_4nm_WO3_-_NbSTO_052617_LO_020620_0005': 3,
#                     'STEM_JEOL_ADF1_02-20-20_Yingge_4nm_WO3_-_NbSTO_052617_LO_020620_0001': 3,
#                     'STEM_JEOL_ADF1_10-12-20_La0.8Sr0.2FeO3-STO-080317-2-LO-zero-deg_0006_1': 6,
#                     'STEM_ADF_11-02-18_10_nm_LFO-STO_A_LO_0_103118_0001': 3,
#                     'STEM_JEOL_ADF1_02-20-20_Yingge_4nm_WO3_-_NbSTO_052617_LO_020620_0002': 3,
#                     'STEM_ADF_09-24-18_30_nm_LMO_STO_081317B_LO_091618_0004': 3,
#                     'STEM_JEOL_ADF1_06-29-20_Wangoh_LSTO_-_STO_0.25_TEM_012020_LO_0_031020_Thinner_0002': 4,
#                     'STEM_JEOL_ADF1_06-29-20_Wangoh_LSTO_-_STO_0.25_TEM_012020_LO_0_031020_Thinner_0004': 3,
#                     'STEM_JEOL_ADF1_03-16-20_Wangoh_LSTO_-_STO_0.25_TEM_012020_LO_0_031020_0001': 5,
#                     'STEM_JEOL_ADF1_10-12-20_La0.8Sr0.2FeO3-STO-080317-2-LO-zero-deg_0004_1': 6,
#                     'STEM_ADF_11-02-18_10_nm_LFO-STO_A_LO_0_103118_0004': 4,
#                     'STEM_JEOL_ADF1_03-16-20_Wangoh_LSTO_-_STO_0.25_TEM_012020_LO_0_031020_0002': 4,
#                     'STEM_JEOL_ADF1_10-12-20_La0.8Sr0.2FeO3-STO-080317-2-LO-zero-deg_0001_1': 5,
#                     'STEM_ADF_09-24-18_30_nm_LMO_STO_081317B_LO_091618_0006': 3,
#                     'STEM_ADF_09-24-18_30_nm_LMO_STO_081317B_LO_091618_0008': 3,
#                     'STEM_JEOL_ADF1_10-12-20_La0.8Sr0.2FeO3-STO-080317-2-LO-zero-deg_0005_1': 3,
#                     'STEM_ADF_09-24-18_30_nm_LMO_STO_081317B_LO_091618_0005': 3,
#                     'STEM_ADF_11-02-18_10_nm_LFO-STO_A_LO_0_103118_0003': 3,
#                     'STEM_JEOL_ADF1_06-29-20_Wangoh_LSTO_-_STO_0.25_TEM_012020_LO_0_031020_Thinner_0003_1': 4,
#                     'STEM_JEOL_HAADF_04-27-17_13_nm_STO_p-Ge_033117_LO_110_042617_HAADF_0005': 3,
#                     'STEM_JEOL_ADF1_03-16-20_Wangoh_LSTO_-_STO_0.25_TEM_012020_LO_0_031020_0004_1': 3,
#                     'STEM_JEOL_ADF1_03-16-20_Wangoh_LSTO_-_STO_0.25_TEM_012020_LO_0_031020_0003': 4,
#                     'STEM_JEOL_HAADF_04-27-17_13_nm_STO_p-Ge_033117_LO_110_042617_HAADF_0006': 3,
#                     'STEM_JEOL_ADF1_10-12-20_La0.8Sr0.2FeO3-STO-080317-2-LO-zero-deg_0002': 5,
#                     'STEM_JEOL_HAADF_04-27-17_13_nm_STO_p-Ge_033117_LO_110_042617_HAADF_0003': 4,
#                     'STEM_JEOL_ADF1_06-29-20_Wangoh_LSTO_-_STO_0.25_TEM_012020_LO_0_031020_Thinner_0001': 5,
#                     'STEM_JEOL_HAADF_04-27-17_13_nm_STO_p-Ge_033117_LO_110_042617_HAADF_0007': 3,
#                     'STEM_JEOL_ADF1_03-16-20_Wangoh_LSTO_-_STO_0.25_TEM_012020_LO_0_031020_0005_1': 4,
#                     'STEM_JEOL_ADF1_10-12-20_La0.8Sr0.2FeO3-STO-080317-2-LO-zero-deg_0003_1': 5,
#                     'STEM_JEOL_ADF1_03-16-20_Wangoh_LSTO_-_STO_0.25_TEM_012020_LO_0_031020_0006': 2,
#                     }

# # Initialize an empty dictionary
# difficulty = {1:[],2:[],3:[],4:[],5:[]}
# vals = set()

# # Path to your CSV file
# csv_file_path = 'CHESS_Labeling_Round_1/12-15-23_Spurgeon_CHESS_Labeling_Round_1/12-15-23 Spurgeon Labeling Notes.csv'

# # Read the CSV file
# with open(csv_file_path, mode='r') as file:
#     # Create a CSV reader object
#     csv_reader = csv.reader(file)
    
#     # Skip the header row
#     next(csv_reader)
    
#     # Iterate through the rows
#     for row in csv_reader:
#         # Use the 3rd column as key and 4th column as value
#         value = row[2].strip()  # Assuming you want to remove leading/trailing whitespaces
#         key = row[3].strip()  # Assuming you want to remove leading/trailing whitespaces
        
#         if key:
#             key = int(key)
#             # Add to the dictionary
#             value = value.replace(".jpg", "")
#             value = value.replace(".", "-")
#             value = value.replace(" ", "_")
#             vals.add(value)
#             difficulty[key].append(value)

# print('difficulty = {')
# for k, v in difficulty.items():
# # Print the resulting dictionary
#     print(k,':', v,',')
# print('}')
# print(f'Total: {len(difficulty)}')

# count = 0
# for k in num_materials.keys():
#     if k in vals:
#         count += 1

# print(f'Matching: {count}')


# ours_log_path = 'logs/2024-01-21--15:57:10/options.log' #'logs/2024-01-21--15:37:48/options.log' # 'logs/2024-01-19--02:05:08/options.log'  #'logs/2024-01-19--01:10:51/options.log'
# default_log_path = 'logs/2024-01-21--15:54:21/options.log' #'logs/2024-01-21--15:19:45/options.log' #'logs/2024-01-19--02:00:55/options.log' #'logs/2024-01-19--01:33:08/options.log'

# # Initialize an empty list to store the data
# ours_data = {}
# default_data = {}

# # Open and read the file line by line
# with open(ours_log_path, 'r') as file:
#     for line in file:
#         # Check if the line starts with 'STEM'
#         if line.startswith('STEM'):
#             # Split the line by spaces and store in a list
#             line_data = line.split()
#             # Append the list to the stem_lines list
#             ours_data[line_data[0]] = [float(v) for v in line_data[1:]] 

# print('ours_data = {')
# for k,v in ours_data.items():
#     print(f'"{k}" : {v},')
# print('}')

# # Open and read the file line by line
# with open(default_log_path, 'r') as file:
#     for line in file:
#         # Check if the line starts with 'STEM'
#         if line.startswith('STEM'):
#             # Split the line by spaces and store in a list
#             line_data = line.split()
#             # Append the list to the stem_lines list
#             default_data[line_data[0]] = [float(v) for v in line_data[1:]] 

# print('default_data = {')
# for k,v in default_data.items():
#     print(f'"{k}" : {v},')
# print('}')