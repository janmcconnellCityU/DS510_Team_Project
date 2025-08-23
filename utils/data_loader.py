"""
DataModule Class

This class handles reading downloaded CSV datasets, `normalizing` and `reshaping` the image data,
`splitting` into training, validation, and test sets, and preparing PyTorch `DataLoader` objects 
for use in model training class workflows. It also manages label remapping and provides access to 
data splits, tensors, and class mappings.
"""

import os
import cv2
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

class DataModule:
    def __init__(self, data_dir='./dataset', dataset_dir='animal-classification-150', im_dim_target=280, batch_size=16, test_size=0.2, random_state=42, save_dataset=False):
        """
        Initializes the DataModule with dataset and preprocessing parameters.

        Args:
            data_dir (str): Directory where raw data is located.
            dataset_dir (str): Directory to save the processed dataset.
            im_dim_target (int): Target dimension for image resizing.
            batch_size (int): Batch size for DataLoader.
            test_size (float): Proportion of dataset to use as test set.
            random_state (int): Random seed for reproducibility.
            save_dataset (bool): Whether to save the processed dataset to CSV.
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_size = test_size
        self.dataset_dir = dataset_dir
        self.random_state = random_state
        self.im_dim_target = im_dim_target
        self.save_dataset = save_dataset

        # Initialize loaders and maps
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.class_map = None

        # Initialize data arrays
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_val = None
        self.y_val = None

        # Initialize tensors
        self.x_train_tensor = None
        self.y_train_tensor = None
        self.x_test_tensor = None
        self.y_test_tensor = None
        self.x_val_tensor = None
        self.y_val_tensor = None

        # Set data and preprocess it
        self._set_data()
        self._preprocess()
        
    def _set_data(self):
        """
        Reads and preprocesses raw images from the data directory.
        Resizes images, flattens them, and saves to CSV if required.
        """
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist.")

        csv_target = f"{self.dataset_dir}/mnist-animals.csv"
        data = []

        if not os.path.exists(csv_target):
            if self.save_dataset and not os.path.exists(self.dataset_dir):
                os.makedirs(self.dataset_dir)
            
            for subdir in os.listdir(self.data_dir):
                subdir_path = os.path.join(self.data_dir, subdir)
                for file in os.listdir(subdir_path):
                    if file.endswith('.png') or file.endswith('.jpg'):
                        file_path = os.path.join(subdir_path, file)

                        # Resize image and convert to grayscale
                        im = cv2.imread(file_path)
                        if im.shape[0] != self.im_dim_target or im.shape[1] != self.im_dim_target:
                            gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                            resized_image = cv2.resize(gray_im, (self.im_dim_target, self.im_dim_target)).flatten()

                            # Append class label and image data
                            class_name = [subdir]
                            row = class_name + resized_image.tolist()
                            data.append(row)

            df = pd.DataFrame(data, columns=(['class'] + [f'pixel_{i}' for i in range(self.im_dim_target * self.im_dim_target)]))

            print(f"Preprocessing complete. Data shape: {df.shape}")

            if self.save_dataset:
                df.to_csv(csv_target, index=False) 
                print(f"CSV file saved to {csv_target}")
        else:
            print(f"CSV file already exists at {csv_target}. Skipping preprocessing.")
            df = pd.read_csv(csv_target)
        
        # Set up class mapping
        self.class_map = {i: label for i, label in enumerate(df['class'].unique())}
        X = df.drop(columns=['class']).values
        y = df['class'].apply(lambda x: list(self.class_map.keys())[list(self.class_map.values()).index(x)]).values
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        
    def _preprocess(self):
        """
        Normalizes data and converts it to PyTorch tensors.
        Splits training data into training and validation sets.
        """
        x_train, x_val, y_train, y_val = train_test_split(
            self.x_train, self.y_train, test_size=self.test_size, random_state=self.random_state, stratify=self.y_train, shuffle=True
        )

        # Normalize pixel values
        x_train = x_train.astype('float32') / 255.0
        x_val = x_val.astype('float32') / 255.0
        x_test = self.x_test.astype('float32') / 255.0

        # Reshape for PyTorch Conv2D
        x_train = x_train.reshape(-1, 1, self.im_dim_target, self.im_dim_target)
        x_val = x_val.reshape(-1, 1, self.im_dim_target, self.im_dim_target)
        x_test = x_test.reshape(-1, 1, self.im_dim_target, self.im_dim_target)
        
        # Update instance variables
        self.x_train, self.y_train, self.x_test, self.y_test, self.x_val, self.y_val = (
            x_train, y_train, x_test, self.y_test, x_val, y_val
        )

        # Convert to PyTorch tensors
        self.x_train_tensor = torch.from_numpy(self.x_train)
        self.y_train_tensor = torch.from_numpy(self.y_train).long()
        self.x_val_tensor = torch.from_numpy(self.x_val)
        self.y_val_tensor = torch.from_numpy(self.y_val).long()
        self.x_test_tensor = torch.from_numpy(self.x_test)
        self.y_test_tensor = torch.from_numpy(self.y_test).long()

        # Create DataLoaders
        train_dataset = TensorDataset(self.x_train_tensor, self.y_train_tensor)
        val_dataset = TensorDataset(self.x_val_tensor, self.y_val_tensor)
        test_dataset = TensorDataset(self.x_test_tensor, self.y_test_tensor)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
    def get_class_map(self):
        """Returns class mapping dictionary."""
        return self.class_map

    def get_train_tensors(self):
        """Returns training, validation, and test tensors."""
        return self.x_train_tensor, self.y_train_tensor, self.x_test_tensor, self.y_test_tensor, self.x_val_tensor, self.y_val_tensor

    def get_train_data(self):
        """Returns raw training, validation, and test data."""
        return self.x_train, self.y_train, self.x_test, self.y_test, self.x_val, self.y_val

    def get_dataloaders(self):
        """Returns DataLoader objects for training, validation, and testing."""
        return self.train_loader, self.val_loader, self.test_loader