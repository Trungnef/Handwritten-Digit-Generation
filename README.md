# 🖋️ Handwritten Digit Generator

This repository contains a web application that generates images of handwritten digits (0–9) using a **Conditional Variational Autoencoder (CVAE)** trained on the MNIST dataset. The app is built with **Streamlit** and deployed on **Streamlit Community Cloud** for public access.

---

## ✨ Features

- **Digit Selection**: Users can select a digit (0–9) via a dropdown menu.
- **Image Generation**: Generates five unique 28x28 grayscale images of the selected digit, resembling the MNIST dataset format.
- **Model**: Uses a custom-trained CVAE (no pre-trained weights) with PyTorch.
- **Deployment**: Hosted on Streamlit Community Cloud, accessible for at least two weeks from deployment.

---

## 📁 Repository Structure

| File | Description |
|------|-------------|
| `app.py` | Streamlit web application for user interface and image generation. |
| `train_cvae.ipynb` | Jupyter notebook with CVAE architecture, training code, and loss function. |
| `decoder_weights.pth` | Trained weights of the decoder (CVAE). |
| `requirements.txt` | Python dependencies required to run the app. |

---

## ⚙️ Setup and Deployment

### ✅ Prerequisites
- Python 3.8+
- Streamlit Community Cloud account
- GitHub repository
- Trained model weights (`decoder_weights.pth`)

---

### 🚀 Installation

```bash
git clone https://github.com/Trungnef/Handwritten-Digit-Generation.git
cd Handwritten-Digit-Generation

python -m venv .venv
./.venv/Scripts/activate

pip install -r requirements.txt
```

---

### 🧠 Training the Model

1. Open `train_cvae.ipynb` in **Google Colab** (use T4 GPU).
2. Run the notebook to train the CVAE on the MNIST dataset.
3. Save the file `decoder_weights.pth` after training.
4. Upload `decoder_weights.pth` to this repository.

---

### 🖥️ Running Locally

Make sure all files (`app.py`, `decoder_weights.pth`, `requirements.txt`) are in the same directory.

```bash
streamlit run app.py
```

Then, access the app at [http://localhost:8501](http://localhost:8501).

---

### 🌐 Deployment

1. Push all project files to your GitHub repository.
2. Connect the repository to [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Deploy the app and get a public URL (you can use here `https://handwritten-digit-generation-bmub8latznjrfgjofi9gaw.streamlit.app/`).

---

## 🧪 Usage

1. Visit the deployed app.
2. Select a digit (0–9) from the dropdown menu.
3. Click **"Generate"** to display five images of that digit.

---

## 📊 Model Details

- **Dataset**: MNIST (28x28 grayscale handwritten digits).
- **Framework**: PyTorch.
- **Model Type**: Conditional Variational Autoencoder (CVAE) with convolutional layers.
- **Loss Function**: Binary Cross-Entropy (reconstruction) + KL Divergence (latent space regularization).
- **Training**: Done on Google Colab with a single T4 GPU, trained for 10 epochs.
- **Output**: Diverse digit images sampled from the CVAE latent space.

---

## ℹ️ Notes

- The app **does not** use any pretrained models; it is trained from scratch.
- The five generated images per digit are **not identical**, but exhibit controlled variation.
- The app will remain accessible for **at least 2 weeks**. It may enter sleep mode when idle, but can be reactivated by visiting the URL.

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).
