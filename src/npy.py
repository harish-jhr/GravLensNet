import numpy as np
import os
from tqdm import tqdm  #For Progress bar

def create_single_npy(data_dir, save_path):
    categories = ['no', 'sphere', 'vort']
    all_images = []
    all_labels = []

    '''
      Arguments:
        data_dir : path to the directory containig all .npy files
        save_path: path to save directory 
    '''

    for label, category in enumerate(categories):
        folder_path = os.path.join(data_dir, category)
        files = sorted(os.listdir(folder_path), key=lambda x: int(x.split('.')[0]))  # Sorting all .npy files numerically

        for file in tqdm(files, desc=f"Loading {category}"):
            file_path = os.path.join(folder_path, file)
            img = np.load(file_path)  # Load .npy file

            if img.shape != (1, 150, 150):
                print(f"Skipping {file_path}, shape: {img.shape}")
                continue

            all_images.append(img)
            all_labels.append(label)

    all_images = np.array(all_images)  # Convert to single NumPy array
    all_labels = np.array(all_labels)

    np.save(save_path, (all_images, all_labels), allow_pickle=True) #Saving all .npy files to one file
    print(f"Saved dataset to {save_path}, shape: {all_images.shape}")

