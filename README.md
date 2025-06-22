Handwritten Digit Generator
This repository contains a web application that generates images of handwritten digits (0–9) using a Conditional Variational Autoencoder (CVAE) trained on the MNIST dataset. The app is built with Streamlit and deployed on Streamlit Community Cloud for public access.
Features

Digit Selection: Users can select a digit (0–9) via a dropdown menu.
Image Generation: Generates five unique 28x28 grayscale images of the selected digit, resembling the MNIST dataset format.
Model: Uses a custom-trained CVAE (no pre-trained weights) with PyTorch.
Deployment: Hosted on Streamlit Community Cloud, accessible for at least two weeks from deployment.

Repository Structure

app.py: Streamlit web application code for the user interface and image generation.
train_cvae.ipynb: Jupyter notebook with the CVAE model architecture, training script, and loss function.
decoder_weights.pth: Trained weights of the CVAE decoder model.
requirements.txt: Python dependencies required to run the app.

Setup and Deployment
Prerequisites

Python 3.8+
Streamlit Community Cloud account
GitHub repository
Trained model weights (decoder_weights.pth)

Installation

Clone the repository:git clone https://github.com/your-username/your-repo-name.git


Install dependencies:pip install -r requirements.txt



Training the Model

Open train_cvae.ipynb in Google Colab (use a T4 GPU).
Run the notebook to train the CVAE on the MNIST dataset.
Save the decoder_weights.pth file after training.
Upload decoder_weights.pth to the repository.

Running Locally

Ensure all files (app.py, decoder_weights.pth, requirements.txt) are in the same directory.
Run the Streamlit app:streamlit run app.py


Access the app at http://localhost:8501.

Deployment

Push all files to a GitHub repository.
Connect the repository to Streamlit Community Cloud.
Deploy the app and obtain the public URL (e.g., https://your-app-name.streamlit.app).

Usage

Visit the deployed app URL.
Select a digit (0–9) from the dropdown menu.
Click "Generate" to display five generated images of the selected digit.

Model Details

Dataset: MNIST (28x28 grayscale images of handwritten digits).
Framework: PyTorch.
Architecture: Conditional Variational Autoencoder (CVAE) with convolutional encoder and decoder.
Loss Function: Binary Cross-Entropy (reconstruction loss) + KL-Divergence (latent space regularization).
Training: Performed on Google Colab with a single T4 GPU for 10 epochs.

Notes

The app generates diverse images by sampling from a latent space, ensuring the five images are not identical.
The model is trained from scratch, adhering to the requirement of not using pre-trained weights.
The deployed app will remain active for at least two weeks and may enter sleep mode when idle but can be reactivated by users.

License
MIT License
