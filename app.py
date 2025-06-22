import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# Define the Decoder class (must match training script)
class Decoder(nn.Module):
    def __init__(self, latent_dim=20):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim + 10, 7*7*64)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z, y):
        z = torch.cat([z, y], dim=1)
        x = self.fc(z)
        x = x.view(x.size(0), 64, 7, 7)
        x = self.relu(self.deconv1(x))
        x = self.sigmoid(self.deconv2(x))
        return x

# Load the trained decoder
decoder = Decoder()
decoder.load_state_dict(torch.load('decoder_weights.pth', map_location=torch.device('cpu'), weights_only=False))
decoder.eval()

# Function to generate images
def generate_images(digit, num_images=5):
    images = []
    for _ in range(num_images):
        z = torch.randn(1, 20)
        y = torch.zeros(1, 10)
        y[0, digit] = 1
        with torch.no_grad():
            generated_image = decoder(z, y).squeeze().numpy()
        images.append(generated_image)
    return images

# Streamlit interface
st.title("Handwritten Digit Generator")
st.write("Select a digit (0-9) and generate 5 images resembling MNIST handwritten digits.")

digit = st.selectbox("Select a digit", list(range(10)))
if st.button("Generate"):
    images = generate_images(digit)
    st.write(f"Generated images for digit {digit}:")
    cols = st.columns(5)
    for i, img in enumerate(images):
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img, mode='L')
        cols[i].image(img, caption=f"Image {i+1}", use_container_width=True)
