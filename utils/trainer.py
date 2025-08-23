"""
##### 1.3.2.1 **CTNNTrainer Module**

**CTNNTrainer** is a utility class for training, evaluating, saving, and loading project models using **PyTorch**. 
It manages the entire model training lifecycle, including **optimizer** and **scheduler** setup, 
hyperparameter management, model **versioning**, and **checkpointing**. 

The class also tracks metrics during training and validation, supports reusing existing models with identical 
training parameters, and saves comprehensive training metrics for later analysis.
"""

import os
import re
import glob
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd

class CTNNTrainer:
    def __init__(self, model, model_dir="./models", model_prefix="best_model",
                 params=None, device=None, lr=0.001, weight_decay=1e-4):
        """
        Initializes the CTNNTrainer with model and training parameters.

        Args:
            model: The neural network model to train.
            model_dir (str): Directory to save models.
            model_prefix (str): Prefix for model files.
            params (dict, optional): Training parameters.
            device (torch.device, optional): Device for computation.
            lr (float): Learning rate.
            weight_decay (float): Weight decay for optimizer.

        """
        self.model = model
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), weight_decay=weight_decay, lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_prefix = model_prefix
        self.train_params = params or {"lr": lr, "weight_decay": weight_decay}
        self.version = self._next_model_version()
        self.base_filename = f"{self.model_prefix}_v{self.version}"
        self.best_model_path = os.path.join(self.model_dir, f"{self.base_filename}.pth")
        self.params_path = self.best_model_path + ".params.json"
        self.metrics_path = os.path.join(self.model_dir, f"training_metrics_{self.version}.csv")
        self.metrics_history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def _next_model_version(self):
        """Determines the next model version based on existing files."""
        files = glob.glob(os.path.join(self.model_dir, f"{self.model_prefix}_v*.pth"))
        versions = []
        pat = re.compile(rf"{re.escape(self.model_prefix)}_v(\d+)\.(\d+)\.pth$")
        for f in files:
            m = pat.search(os.path.basename(f))
            if m:
                versions.append((int(m.group(1)), int(m.group(2))))
        if not versions:
            return "1.0"
        versions.sort()
        latest_major, latest_minor = versions[-1]
        return f"{latest_major}.{latest_minor + 1}"

    @staticmethod
    def calculate_accuracy(model, data_loader, device):
        """Calculates accuracy of the model on a given data loader."""
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy

    def find_matching_model(self):
        """
        Checks for an existing model with matching parameters.
        
        Returns:
            Tuple containing model path, params path, metrics path, and version if found, else None.
        """
        files = glob.glob(os.path.join(self.model_dir, f"{self.model_prefix}_v*.pth"))
        pat = re.compile(rf"{re.escape(self.model_prefix)}_v(\d+)\.(\d+)\.pth$")
        for model_path in files:
            m = pat.search(os.path.basename(model_path))
            if m:
                version = f"{m.group(1)}.{m.group(2)}"
                params_path = model_path + ".params.json"
                if os.path.exists(params_path):
                    with open(params_path, 'r') as f:
                        params = json.load(f)
                    if params == self.train_params:
                        metrics_path = os.path.join(self.model_dir, f"training_metrics_{version}.csv")
                        return model_path, params_path, metrics_path, version
        return None

    def save_model_and_params(self):
        """Saves the model state and training parameters."""
        torch.save(self.model.state_dict(), self.best_model_path)
        with open(self.params_path, 'w') as f:
            json.dump(self.train_params, f)

    def load_model(self, model_path):
        """Loads a pre-trained model from the specified path."""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"\n[INFO] Loaded pre-trained model from {model_path}\n")

    def train(self, train_loader, val_loader, num_epochs=20, save_metrics=False, save_model=False):
        """
        Trains the model for a specified number of epochs.

        Args:
            train_loader: DataLoader for training.
            val_loader: DataLoader for validation.
            num_epochs (int): Number of training epochs.
            save_metrics (bool): Flag to save metrics.
            save_model (bool): Flag to save model.
        """
        match = self.find_matching_model()
        if match:
            model_path, params_path, metrics_path, version = match
            self.version = version
            self.base_filename = f"{self.model_prefix}_v{self.version}"
            self.best_model_path = model_path
            self.params_path = params_path
            self.metrics_path = metrics_path
            self.load_model(model_path)
            if os.path.isfile(self.metrics_path):
                print(f"[INFO] Found matching training metrics at {self.metrics_path}")
                df = pd.read_csv(self.metrics_path)
                print(df)
            else:
                print("[INFO] Metrics summary for this pretrained model not found!")
            return

        print(f"Training new model: saving as version v{self.version}")

        best_val_score = 0.0

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
            for inputs, labels in train_pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            epoch_train_loss = running_loss / len(train_loader.dataset)
            epoch_train_acc = self.calculate_accuracy(self.model, train_loader, self.device)

            # Validation
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            self.model.eval()
            running_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_pbar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    running_loss += loss.item() * inputs.size(0)
            
            epoch_val_loss = running_loss / len(val_loader.dataset)
            epoch_val_acc = self.calculate_accuracy(self.model, val_loader, self.device)
            self.scheduler.step(epoch_val_loss)
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
                  f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")

            if save_metrics or save_model:
                self.metrics_history['epoch'].append(epoch + 1)
                self.metrics_history['train_loss'].append(epoch_train_loss)
                self.metrics_history['train_acc'].append(epoch_train_acc)
                self.metrics_history['val_loss'].append(epoch_val_loss)
                self.metrics_history['val_acc'].append(epoch_val_acc)

                epoch_score = round((epoch_val_acc - (epoch_val_loss * 10) - epoch_train_loss), 4)
                if save_model and epoch_score > best_val_score:
                    best_val_score = epoch_score
                    self.save_model_and_params()
                    self.load_model(self.best_model_path)

                if save_metrics:
                    self.save_metrics(self.metrics_path)
                    print(f"\n[INFO] Model and summary saved as: {self.base_filename}\n")

    def print_metrics_summary(self):
        """Prints a summary of the training metrics."""
        df = pd.DataFrame(self.metrics_history)
        print("\nTraining summary by epoch:")
        print(df.to_string(index=False))

    def save_metrics(self, path=None):
        """Saves training metrics to a CSV file."""
        df = pd.DataFrame(self.metrics_history)
        path = path or self.metrics_path
        df.to_csv(path, index=False)
        print(f"Epoch metrics saved to {path}")