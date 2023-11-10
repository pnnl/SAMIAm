import os

directory = './chessdataset'

for filename in os.listdir(directory):
    if filename.endswith('.tiff'):
        new_filename = filename.replace(' ', '_')
        new_filename = new_filename.replace('_-_', '_')
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)
        os.rename(old_path, new_path)