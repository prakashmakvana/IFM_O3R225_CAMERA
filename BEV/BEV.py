import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Step 1: Load pre-trained Vision Transformer (ViT)
class ViT_FE(nn.Module):
    def __init__(self):
        super(ViT_FE, self).__init__()
        self.model = models.vit_b_16(weights="IMAGENET1K_V1")  # Vision Transformer

    def forward(self, x):
        print(f"Input tensor shape: {x.shape}")
        x = self.model.conv_proj(x)  # Apply patch embedding
        print(f"After conv_proj shape: {x.shape}")
        x = x.flatten(2).transpose(1, 2)  # Flatten and transpose for ViT (batch_size, seq_length, hidden_dim)
        print(f"After flatten and transpose shape: {x.shape}")

        # Extract features directly from the model's encoder without modifying positional embedding
        features = self.model.encoder(x)  # Extract features
        print(f"Features shape: {features.shape}")
        return features

# Step 2: Function for BEV generation
class BEV_Generation(nn.Module):
    def __init__(self):
        super(BEV_Generation, self).__init__()
        self.conv1 = nn.Conv2d(768, 512, kernel_size=3, padding=1)  # Example Conv for BEV processing
        self.conv2 = nn.Conv2d(512, 1, kernel_size=3, padding=1)  # BEV output with 1 channel (grayscale)

    def forward(self, x):
        print(f"BEV input shape: {x.shape}")
        x = torch.relu(self.conv1(x))
        print(f"After conv1 shape: {x.shape}")
        x = self.conv2(x)
        print(f"After conv2 shape: {x.shape}")
        return x

# Step 3: Image preprocessing and transformations
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to 224x224, matching ViT's expected input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard normalization
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Step 4: Combine ViT feature extraction and BEV generation
def generate_bev(image_path):
    # Load models
    vit_model = ViT_FE()
    bev_model = BEV_Generation()

    # Load and preprocess image
    image_tensor = preprocess_image(image_path)
    print(f"Image tensor shape: {image_tensor.shape}")

    # Feature extraction with ViT
    features = vit_model(image_tensor)

    # Reshape the features to match the BEV model input size
    features = features.view(features.size(0), 768, 14, 14)  # Reshaping to 4D tensor (batch, channels, height, width)
    print(f"Features after reshaping: {features.shape}")

    # Generate BEV representation
    bev_output = bev_model(features)

    # Convert BEV output to numpy for visualization
    bev_image = bev_output.squeeze().detach().numpy()

    # Step 5: Visualization
    plt.imshow(bev_image, cmap='gray')
    plt.title('Bird\'s Eye View Representation')
    plt.show()

# Example usage
image_path = '/home/demo/Desktop/captured_images/front_0.png'
generate_bev(image_path)
