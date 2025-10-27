# 🪑 XAI-GenArt: Furniture Design Assistant

> Generative + Explainable AI for intelligent furniture design

XAI-GenArt is a hybrid **Generative AI + Explainable AI** system that turns text prompts into realistic furniture concepts **and explains its reasoning**.  
It combines a **Latent Diffusion Model (LDM)** with **Explainable AI (XAI)** techniques like SHAP & Integrated Gradients to highlight which words most influenced each generated design.

---

## 🚀 Highlights
- 🧠 **Text-to-Image Generation** – Create designs from natural-language descriptions  
- 🔍 **Word Attribution Heatmaps** – Visualize which prompt tokens drove the output  
- ⚙️ **Custom VAE Losses** – Reconstruction + Perceptual + SSIM + KL Divergence  
- 🧩 **Latent Fusion Module** – Merges visual & textual embeddings  
- 📊 **Transparent Training** – 29 K images processed via GIT + LLaMA-3 (Groq API)

---

## 🏗️ Architecture
**Pipeline**
1. **VAE Encoder/Decoder** – compress ↔ reconstruct images  
2. **CaptionProjector** – encode text to latent space  
3. **Latent Fusion** – join text + image features  
4. **UNet + Noise Scheduler** – diffusion-based image generation  
5. **Explainability Engine** – compute word-level impact using cosine similarity  

---

## 🧪 Results

![PHOTO-2025-05-08-23-31-01 2](https://github.com/user-attachments/assets/453cc531-bf87-484e-8a08-5646d52ece6f)

![PHOTO-2025-05-08-23-31-01](https://github.com/user-attachments/assets/14c165a1-fc8d-4bae-a3d8-b0e61e5cd8b8)

![PHOTO-2025-05-08-23-32-56](https://github.com/user-attachments/assets/22739593-5061-46e0-a1e2-04fbfba9ba82)
