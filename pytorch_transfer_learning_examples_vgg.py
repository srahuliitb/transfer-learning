import os
import random
import numpy as np
from PIL import Image # For loading images directly
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models # For VGG16 and image transformations
import matplotlib.pyplot as plt
import logging

# Configure logging
log_filename = 'pytorch_transfer_learning_example_vgg.log'
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Creating an object
logger = logging.getLogger()

# Set a random seed for reproducibility
# Note: For full reproducibility, additional steps are needed for cuDNN and data loaders.
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if using multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42) # Using a fixed seed

# --- PART 1: CPU RAM Requiring Tasks (Setup and Data Preprocessing) ---
# All code in this section primarily uses CPU memory and does not require GPU VRAM.
# This includes defining paths, loading raw data, splitting, pre-processing,
# and defining model architectures as blueprints.

logger.info("--- Part 1: CPU RAM Requiring Tasks (Setup and Data Preprocessing) ---")
#print("\n--- Part 1: CPU RAM Requiring Tasks (Setup and Data Preprocessing) ---\n")

logger.info("1.1 System Configuration and Data Path Setup (CPU RAM).")
# print("1.1 System Configuration and Data Path Setup (CPU RAM).")
root = '101_ObjectCategories'
exclude = ['BACKGROUND_Google', 'Motorbikes', 'airplanes', 'Faces_easy', 'Faces']
train_split, val_split = 0.7, 0.15

categories_paths = [x[0] for x in os.walk(root) if x[0]][1:]
categories_paths = [c for c in categories_paths if c not in [os.path.join(root, e) for e in exclude]]

# Map category paths to simpler names (e.g., 'car' from '101_ObjectCategories/car')
categories = [os.path.basename(p) for p in categories_paths]
category_to_idx = {name: i for i, name in enumerate(categories)}
idx_to_category = {i: name for i, name in enumerate(categories)}

logger.info(f"Image categories: {categories}")
# print(f"  Image categories: {categories}\n")

logger.info("1.2 Custom Dataset Class Definition (CPU RAM - blueprint only).")
# print("1.2 Custom Dataset Class Definition (CPU RAM - blueprint only).")
class Caltech101Dataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path = self.data_list[idx]['path']
        label = self.data_list[idx]['y']

        # Load image using PIL
        img = Image.open(img_path).convert('RGB') # Ensure 3 channels for VGG

        if self.transform:
            img = self.transform(img)

        return img, label

logger.info("1.3 Image transformations setup (CPU RAM).")
# print("1.3 Image transformations setup (CPU RAM).")
# Define transformations. These are applied on the CPU when data is loaded,
# before being potentially moved to GPU.
preprocess_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to VGG16 input size
    transforms.ToTensor(),         # Converts to PyTorch Tensor (HWC to CHW, 0-255 to 0-1)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization
])

logger.info("1.4 Loading and processing all image data paths (CPU RAM).")
# print("1.4 Loading and processing all image data paths (CPU RAM).")
# We only store paths and labels here to avoid loading all images into RAM at once.
# Images will be loaded on-the-fly by the DataLoader.
all_image_paths_and_labels = []
for c, category_path in enumerate(categories_paths):
    images_in_category = [os.path.join(dp, f) for dp, dn, filenames
                          in os.walk(category_path) for f in filenames
                          if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]
    for img_path in images_in_category:
        all_image_paths_and_labels.append({'path': img_path, 'y': c})

num_classes = len(categories)

logger.info(f"Total number of image classes considered = {num_classes}")
logger.info(f"Total number of images found: {len(all_image_paths_and_labels)}")
# print(f"  Total number of image classes considered = {num_classes}\n")
# print(f"  Total number of images found: {len(all_image_paths_and_labels)}\n")

logger.info("1.5 Shuffling and splitting data paths (CPU RAM).")
# print("1.5 Shuffling and splitting data paths (CPU RAM).")
random.shuffle(all_image_paths_and_labels) # Randomise the data order

# create training / validation / test split (70%, 15%, 15%)
idx_val = int(train_split * len(all_image_paths_and_labels))
idx_test = int((train_split + val_split) * len(all_image_paths_and_labels))
train_data_list = all_image_paths_and_labels[:idx_val]
val_data_list = all_image_paths_and_labels[idx_val:idx_test]
test_data_list = all_image_paths_and_labels[idx_test:]

logger.info("Summary:")
logger.info("Finished collecting %d image paths from %d categories"%(len(all_image_paths_and_labels), num_classes))
logger.info("Train / validation / test split (counts): %d, %d, %d"%(len(train_data_list), len(val_data_list), len(test_data_list)))
# print("  Summary:")
# print("  Finished collecting %d image paths from %d categories"%(len(all_image_paths_and_labels), num_classes))
# print("  Train / validation / test split (counts): %d, %d, %d"%(len(train_data_list), len(val_data_list), len(test_data_list)))

logger.info("1.6 Creating Dataset and DataLoader objects (CPU RAM).")
# print("\n1.6 Creating Dataset and DataLoader objects (CPU RAM).")
# These objects are created on CPU and manage the loading of data.
train_dataset = Caltech101Dataset(train_data_list, transform=preprocess_transform)
val_dataset = Caltech101Dataset(val_data_list, transform=preprocess_transform)
test_dataset = Caltech101Dataset(test_data_list, transform=preprocess_transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=os.cpu_count()//2 or 1)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=os.cpu_count()//2 or 1)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=os.cpu_count()//2 or 1)
logger.info("DataLoaders created. Images will be loaded in batches during training/evaluation.")
# print("  DataLoaders created. Images will be loaded in batches during training/evaluation.")


logger.info("1.7 Sequential Model Architecture Definition (CPU RAM - blueprint only).")
# print("\n1.7 Sequential Model Architecture Definition (CPU RAM - blueprint only).")
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # Input: (batch_size, 3, 224, 224)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # (224-3+2*1)/1 + 1 = 224 -> (32, 224, 224)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 224/2 = 112 -> (32, 112, 112)

            nn.Conv2d(32, 32, kernel_size=3, padding=1), # (32, 112, 112)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 112/2 = 56 -> (32, 56, 56)
            nn.Dropout(0.25),

            nn.Conv2d(32, 32, kernel_size=3, padding=1), # (32, 56, 56)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 56/2 = 28 -> (32, 28, 28)

            nn.Conv2d(32, 32, kernel_size=3, padding=1), # (32, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 28/2 = 14 -> (32, 14, 14)
            nn.Dropout(0.25)
        )
        # Calculate output size of features layer: 32 channels * 14 * 14 = 6272
        self.classifier = nn.Sequential(
            nn.Linear(32 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1) # Flatten starting from dimension 1 (channel, height, width)
        x = self.classifier(x)
        return x

# Instantiate the model. Its parameters are on the CPU by default.
model_sequential = SimpleCNN(num_classes)
logger.info(f"Sequential model architecture (CPU RAM):")
logger.info(model_sequential)
# print(f"\n  Sequential model architecture (CPU RAM):\n")
# print(model_sequential)

logger.info("--- Part 2: GPU VRAM and/or CPU RAM Requiring Tasks (Training and Evaluation) ---")
logger.info("This section involves operations that leverage GPU if available, or fall back to CPU.")
# print("\n--- Part 2: GPU VRAM and/or CPU RAM Requiring Tasks (Training and Evaluation) ---\n")
# print("This section involves operations that leverage GPU if available, or fall back to CPU.")

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}. If 'cuda', GPU VRAM will be used for computations.")
# print(f"  Using device: {device}. If 'cuda', GPU VRAM will be used for computations.")

logger.info("\n2.1 Moving Sequential Model to Device (REQUIRES GPU VRAM if 'cuda').")
# print("\n2.1 Moving Sequential Model to Device (REQUIRES GPU VRAM if 'cuda').")
# This is the first point where GPU RAM might be allocated for model_sequential parameters.
model_sequential.to(device)
logger.info(f"  Sequential model moved to {device}.")
# print(f"  Sequential model moved to {device}.")

logger.info("2.2 Defining Loss Function and Optimizer for Sequential Model (CPU RAM, but operates on GPU if model is there).")
# print("2.2 Defining Loss Function and Optimizer for Sequential Model (CPU RAM, but operates on GPU if model is there).")
criterion = nn.CrossEntropyLoss()
optimizer_sequential = optim.Adam(model_sequential.parameters(), lr=0.001)
logger.info("Loss function and optimizer defined.")
# print("  Loss function and optimizer defined.")

logger.info("\n2.3 Training the Sequential Model (REQUIRES GPU VRAM if available, otherwise CPU RAM).")
# print("\n2.3 Training the Sequential Model (REQUIRES GPU VRAM if available, otherwise CPU RAM).")
num_epochs = 10 # Match the 10 epochs from Keras example
history_sequential = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

for epoch in range(num_epochs):
    model_sequential.train() # Set model to training mode
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # Move data to the device. This allocates GPU RAM for data batches.
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer_sequential.zero_grad()
        outputs = model_sequential(inputs) # Forward pass (computations on GPU, activations consume GPU RAM)
        loss = criterion(outputs, labels) # Calculate loss (on GPU)
        loss.backward()                  # Backward pass (gradients computed and stored on GPU RAM)
        optimizer_sequential.step()      # Update weights (on GPU)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        if batch_idx % 50 == 0: # Print less frequently than Keras due to smaller dataset
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
            # print(f"    Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_dataset)
    epoch_accuracy = correct_train / total_train
    history_sequential['loss'].append(epoch_loss)
    history_sequential['accuracy'].append(epoch_accuracy)
    logger.info(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
    # print(f"  Epoch {epoch+1} Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}")

    # Validation phase
    model_sequential.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad(): # Disable gradient calculation for validation
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_sequential(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss_avg = val_loss / len(val_dataset)
    val_accuracy = correct_val / total_val
    history_sequential['val_loss'].append(val_loss_avg)
    history_sequential['val_accuracy'].append(val_accuracy)
    logger.info(f"Epoch {epoch+1} Val Loss: {val_loss_avg:.4f}, Val Acc: {val_accuracy:.4f}")
    # print(f"  Epoch {epoch+1} Val Loss: {val_loss_avg:.4f}, Val Acc: {val_accuracy:.4f}")

logger.info("\nSequential model training complete. GPU VRAM/CPU RAM was utilized for computations.")
# print("\n  Sequential model training complete. GPU VRAM/CPU RAM was utilized for computations.")

logger.info("\n2.4 Evaluating the Sequential Model (REQUIRES GPU VRAM if available, otherwise CPU RAM).")
# print("\n2.4 Evaluating the Sequential Model (REQUIRES GPU VRAM if available, otherwise CPU RAM).")
model_sequential.eval()
test_loss = 0.0
correct_test = 0
total_test = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model_sequential(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

final_test_loss = test_loss / len(test_dataset)
final_test_accuracy = correct_test / total_test

logger.info(f'Sequential Model Test loss: {final_test_loss:.4f}')
logger.info(f'Sequential Model Test accuracy: {final_test_accuracy:.4f}')
# print(f'  Sequential Model Test loss: {final_test_loss:.4f}')
# print(f'  Sequential Model Test accuracy: {final_test_accuracy:.4f}')

# --- Transfer Learning Section ---
logger.info("\n2.5 Loading VGG16 Pre-trained Model (Downloads to CPU, then moved to GPU).")
# print("\n2.5 Loading VGG16 Pre-trained Model (Downloads to CPU, then moved to GPU).")
# PyTorch's VGG16 with pre-trained ImageNet weights
vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
logger.info(f"\nVGG16 Model loaded (initially CPU RAM, parameters transfer to GPU when moved).\n")
logger.info(vgg16)
# print(f"\n  VGG16 Model loaded (initially CPU RAM, parameters transfer to GPU when moved).\n")
# print(vgg16) # Uncomment for full VGG16 summary

logger.info("\n2.6 Building New Transfer Learning Model with VGG16 as backbone (CPU RAM blueprint, GPU VRAM for actual use).")
# print("\n2.6 Building New Transfer Learning Model with VGG16 as backbone (CPU RAM blueprint, GPU VRAM for actual use).")
# Freeze all parameters in the VGG16 feature extractor
for param in vgg16.parameters():
    param.requires_grad = False

# Replace the classifier head with a new one for our number of classes
num_ftrs = vgg16.classifier[6].in_features # Get input features to the last FC layer
vgg16.classifier[6] = nn.Linear(num_ftrs, num_classes) # Replace last layer

# Move the VGG16 model (with new head) to the device
model_transfer = vgg16.to(device)
logger.info(f"Transfer learning model moved to {device}.")
logger.info(f"\n  Transfer learning model architecture (new head added, others frozen):\n")
logger.info(model_transfer)
# print(f"  Transfer learning model moved to {device}.")
# print(f"\n  Transfer learning model architecture (new head added, others frozen):\n")
# print(model_transfer)

logger.info("\n2.7 Defining Optimizer for Transfer Learning Model (only last layer trainable).")
# print("\n2.7 Defining Optimizer for Transfer Learning Model (only last layer trainable).")
# Only parameters of the new classifier head are trainable
optimizer_transfer = optim.Adam(model_transfer.parameters(), lr=0.001)
# Note: criterion (CrossEntropyLoss) remains the same

logger.info("\n2.8 Training the Transfer Learning Model (REQUIRES GPU VRAM if available, otherwise CPU RAM).")
# print("\n2.8 Training the Transfer Learning Model (REQUIRES GPU VRAM if available, otherwise CPU RAM).")
history_transfer = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

for epoch in range(num_epochs):
    model_transfer.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer_transfer.zero_grad()
        outputs = model_transfer(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_transfer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        if batch_idx % 50 == 0:
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
            # print(f"    Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_dataset)
    epoch_accuracy = correct_train / total_train
    history_transfer['loss'].append(epoch_loss)
    history_transfer['accuracy'].append(epoch_accuracy)
    logger.info(f"Epoch {epoch+1} Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}")
    # print(f"  Epoch {epoch+1} Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}")

    # Validation phase for transfer learning model
    model_transfer.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_transfer(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss_avg = val_loss / len(val_dataset)
    val_accuracy = correct_val / total_val
    history_transfer['val_loss'].append(val_loss_avg)
    history_transfer['val_accuracy'].append(val_accuracy)
    logger.info(f"Epoch {epoch+1} Val Loss: {val_loss_avg:.4f}, Val Acc: {val_accuracy:.4f}")
    # print(f"  Epoch {epoch+1} Val Loss: {val_loss_avg:.4f}, Val Acc: {val_accuracy:.4f}")

logger.info("\nTransfer learning model training complete. GPU VRAM/CPU RAM was utilized.")
# print("\n  Transfer learning model training complete. GPU VRAM/CPU RAM was utilized.")

logger.info("\n2.9 Evaluating the Transfer Learning Model (REQUIRES GPU VRAM if available, otherwise CPU RAM).")
# print("\n2.9 Evaluating the Transfer Learning Model (REQUIRES GPU VRAM if available, otherwise CPU RAM).")
model_transfer.eval()
test_loss_transfer = 0.0
correct_test_transfer = 0
total_test_transfer = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model_transfer(inputs)
        loss = criterion(outputs, labels)
        test_loss_transfer += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_test_transfer += labels.size(0)
        correct_test_transfer += (predicted == labels).sum().item()

final_test_loss_transfer = test_loss_transfer / len(test_dataset)
final_test_accuracy_transfer = correct_test_transfer / total_test_transfer
logger.info(f'Transfer Learning Model Test loss: {final_test_loss_transfer:.4f}')
logger.info(f'Transfer Learning Model Test accuracy: {final_test_accuracy_transfer:.4f}')
# print(f'  Transfer Learning Model Test loss: {final_test_loss_transfer:.4f}')
# print(f'  Transfer Learning Model Test accuracy: {final_test_accuracy_transfer:.4f}')

logger.info("\n2.10 Making Predictions (REQUIRES GPU VRAM if available, otherwise CPU RAM).")
# print("\n2.10 Making Predictions (REQUIRES GPU VRAM if available, otherwise CPU RAM).")
# Load a single image for prediction
sample_img_path = os.path.join(root, 'airplanes', 'image_0003.jpg')
sample_img = Image.open(sample_img_path).convert('RGB')
input_tensor = preprocess_transform(sample_img)
input_batch = input_tensor.unsqueeze(0) # Add a batch dimension
input_batch = input_batch.to(device) # Move to the device

model_transfer.eval()
with torch.no_grad():
    output = model_transfer(input_batch)
    probabilities = torch.softmax(output, dim=1)

logger.info(f"Prediction probabilities for sample image: {probabilities.cpu().numpy()}")
# print(f"  Prediction probabilities for sample image: {probabilities.cpu().numpy()}")
predicted_class_idx = torch.argmax(probabilities, dim=1).item()
predicted_category_name = idx_to_category[predicted_class_idx]
logger.info(f"Predicted class: {predicted_category_name} (Index: {predicted_class_idx})")
# print(f"  Predicted class: {predicted_category_name} (Index: {predicted_class_idx})")


# --- Visualisation (Primarily CPU RAM) ---
# This part is about plotting and saving figures, which typically uses CPU RAM.
# It can be placed after all training/evaluation.
logger.info("\n--- Visualisation (Primarily CPU RAM) ---\n")
# print("\n--- Visualisation (Primarily CPU RAM) ---\n")

logger.info("Plotting training history for sequential model and transfer learning model comparison (CPU RAM).")
# print("Plotting training history for sequential model and transfer learning model comparison (CPU RAM).")

plt.figure(figsize=(16, 6))

# Plot Validation Loss
plt.subplot(1, 2, 1)
plt.plot(history_sequential["val_loss"], label="Sequential Loss")
plt.plot(history_transfer["val_loss"], label="Transfer Learning Loss")
plt.title("Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Plot Validation Accuracy
plt.subplot(1, 2, 2)
plt.plot(history_sequential["val_accuracy"], label="Sequential Accuracy")
plt.plot(history_transfer["val_accuracy"], label="Transfer Learning Accuracy")
plt.title("Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.legend()

plt.tight_layout()
plt.savefig("pytorch_model_comparison.png")
logger.info("Plots saved to pytorch_model_comparison.png.")
# print("  Plots saved to pytorch_model_comparison.png.")

logger.info("\nVisualising sample images (CPU RAM).")
# print("\nVisualising sample images (CPU RAM).")
# Visualise a few images from the dataset (CPU RAM)
# Get paths for 8 random images for visualization
sample_image_paths = [img_info['path'] for img_info in random.sample(all_image_paths_and_labels, 8)]

imgs_to_show = []
for p in sample_image_paths:
    try:
        img = Image.open(p).convert('RGB')
        imgs_to_show.append(np.array(img.resize((224, 224)))) # Resize for consistent display
    except Exception as e:
        log.error(f"Could not load image {p}: {e}")
        # print(f"Could not load image {p}: {e}")
        continue

if imgs_to_show:
    concat_image = np.concatenate(imgs_to_show, axis=1)
    plt.figure(figsize=(16, 4))
    plt.imshow(concat_image)
    plt.title("Sample Images from Dataset")
    plt.axis('off') # Hide axes
    plt.savefig('pytorch_caltech_images_sample.png')
    # print("  Sample image visualization saved to pytorch_caltech_images_sample.png.")
    logger.info("Sample image visualization saved to pytorch_caltech_images_sample.png.")
else:
    logger.info("No sample images could be loaded for visualization.")
    # print("  No sample images could be loaded for visualization.")

