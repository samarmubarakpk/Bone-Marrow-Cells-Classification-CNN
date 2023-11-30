import os
import zipfile
import gdown
from Cnn_Classifier import logger
from Cnn_Classifier.utils.common import get_size
import os
import shutil
import zipfile
import numpy as np
import subprocess
import random
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img
from pathlib import Path
from Cnn_Classifier import logger
from PIL import Image
import random
import os
from Cnn_Classifier.entity.config_entity import DataIngestionConfig
import random
import numpy as np
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image


# Path to the uploaded kaggle.json file
uploaded_kaggle_json_path = 'kaggle.json'

# Create the .kaggle directory in the home folder, if it doesn't exist
kaggle_dir = os.path.expanduser('~/.kaggle')
os.makedirs(kaggle_dir, exist_ok=True)

# Move the kaggle.json to the .kaggle directory
kaggle_json_path = os.path.join(kaggle_dir, 'kaggle.json')
shutil.move(uploaded_kaggle_json_path, kaggle_json_path)

# Set the required permission for the file
os.chmod(kaggle_json_path, 0o600)

print(f"kaggle.json moved to {kaggle_json_path} with proper permissions set.")






class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.root_dir_path = Path(self.config.root_dir)  # Converted Path object
        self.train_dir = self.root_dir_path / 'train_data_set'
        self.val_dir = self.root_dir_path / 'validation_data_set'
        self.split_ratio = 0.3

    def remove_readonly_and_delete(self, file_path):
        try:
            os.chmod(file_path, 0o777)  # Change to allow all operations
            os.remove(file_path)
        except Exception as e:
            print(f"Error removing file {file_path}: {e}")

    def handle_directory(self, directory_path):
        if os.path.exists(directory_path):
            # Change permissions and remove files in the directory
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    self.remove_readonly_and_delete(file_path)
                elif os.path.isdir(file_path):
                    self.handle_directory(file_path)  # Recursively handle subdirectories
            os.rmdir(directory_path)  # Now remove directory


    def download_file(self):
        """
        Download data from Kaggle only if it hasn't been downloaded already.
        """
        try:
            # Check if the dataset file already exists
            if os.path.exists(self.config.local_data_file):
                logger.info(f"Dataset already downloaded: {self.config.local_data_file}")
                return

            # Ensure Kaggle directory and credentials file exist
            assert os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')), "Kaggle API credentials not found"

            # Kaggle dataset command
            dataset_command = ['kaggle', 'datasets', 'download', '-d', self.config.source_URL, '-p', str(self.root_dir_path)]
            subprocess.run(dataset_command, check=True, capture_output=True)

            # Find downloaded zip file
            zip_files = list(self.root_dir_path.glob('*.zip'))
            if zip_files:
                downloaded_zip = zip_files[0]
                shutil.move(str(downloaded_zip), str(self.config.local_data_file))
                logger.info(f"Downloaded data from Kaggle into file {self.config.local_data_file}")
            else:
                logger.error("No zip file downloaded from Kaggle")
                raise FileNotFoundError("Downloaded Kaggle dataset not found")

        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            raise e
     
    
   

    def extract_zip_file(self):
        unzip_path = Path(self.config.unzip_dir)
        os.makedirs(unzip_path, exist_ok=True)

        # Check if extraction is needed by verifying if expected content exists
        # Replace 'expected_content_folder' with an actual folder/file name you expect to find after extraction
        expected_content_path = unzip_path / "artifacts/data_ingestion/bone_marrow_cell_dataset"

        if expected_content_path.exists():
            logger.info(f"Data already extracted in {unzip_path}")
            return

        try:
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
                logger.info(f"Extracted data into {unzip_path}")
        except Exception as e:
            logger.error(f"Error extracting zip file: {e}")
            raise e

    # ... other methods ...


 

    def organize_dataset(self):
        base_path = 'artifacts/data_ingestion/bone_marrow_cell_dataset'  # This can be set as an attribute if it's a fixed value
        for folder in os.listdir(base_path):
            folder_path = os.path.join(base_path, folder)
            if os.path.isdir(folder_path):
                for subfolder in os.listdir(folder_path):
                    subfolder_path = os.path.join(folder_path, subfolder)
                    if os.path.isdir(subfolder_path):
                        # Move files from sub-subfolders to main subfolder
                        for file in os.listdir(subfolder_path):
                            shutil.move(os.path.join(subfolder_path, file), folder_path)
                        # Remove the now empty sub-subfolder
                        os.rmdir(subfolder_path)

    def augment_image(self, image_path, save_path, num_required):
        image = Image.open(image_path)
        augmentations = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
        while num_required > 0:
            aug_image = image.transpose(random.choice(augmentations))
            aug_image.save(os.path.join(save_path, f'aug_{num_required}.jpg'))
            num_required -= 1

    def balance_dataset(self, target_count=700):
        base_path = 'artifacts/data_ingestion/bone_marrow_cell_dataset'  # This can be set as an attribute if it's a fixed value
        for folder in os.listdir(base_path):
            folder_path = os.path.join(base_path, folder)
            if os.path.isdir(folder_path):
                images = [img for img in os.listdir(folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
                current_count = len(images)

                if current_count < target_count:
                    # Augment images
                    for i in range(target_count - current_count):
                        image_path = os.path.join(folder_path, random.choice(images))
                        self.augment_image(image_path, folder_path, target_count - current_count)

                elif current_count > target_count:
                    # Randomly delete images
                    images_to_delete = random.sample(images, current_count - target_count)
                    for img in images_to_delete:
                        os.remove(os.path.join(folder_path, img))

    def split_dataset(self):
            base_path = Path('artifacts/data_ingestion/bone_marrow_cell_dataset')  # Replace with your dataset path
            train_dir = self.root_dir_path / 'train_data_set'
            val_dir = self.root_dir_path / 'validation_data_set'

            # Create train and validation directories if they don't exist
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)

            for folder in os.listdir(base_path):
                folder_path = base_path / folder

                if os.path.isdir(folder_path):
                    images = [img for img in os.listdir(folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
                    random.shuffle(images)  # Shuffle images to randomize the split

                    # Calculate split index
                    split_index = int(len(images) * 0.75)

                    # Split into training and validation images
                    train_images = images[:split_index]
                    val_images = images[split_index:]

                    # Copy training images
                    train_folder_path = train_dir / folder
                    os.makedirs(train_folder_path, exist_ok=True)
                    for img in train_images:
                        shutil.copy(folder_path / img, train_folder_path / img)

                    # Copy validation images
                    val_folder_path = val_dir / folder
                    os.makedirs(val_folder_path, exist_ok=True)
                    for img in val_images:
                        shutil.copy(folder_path / img, val_folder_path / img)    


                            



