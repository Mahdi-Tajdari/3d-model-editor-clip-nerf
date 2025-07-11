# 3D Model Editor: CLIP-Guided NeRF

## Project Overview

This project implements a deep learning system for generating and editing 3D scene representations directly from text prompts. It leverages the power of **CLIP (Contrastive Language-Image Pre-training)** for semantic guidance and **NeRF (Neural Radiance Fields)** for 3D scene representation and rendering.

Unlike traditional 3D modeling, this system allows users to "sculpt" or "refine" 3D objects purely through natural language descriptions, offering a novel way to interact with 3D content creation.

## Key Features

* **Text-to-3D Generation:** Create novel 3D models from scratch using a descriptive text prompt (e.g., "A small red car").
* **Text-Guided 3D Editing:** Modify existing 3D scenes or objects by providing new textual instructions (e.g., "Make it blue" for a previously generated red car).
* **CLIP Integration:** Utilizes CLIP's powerful image-text understanding to provide semantic feedback during NeRF's optimization.
* **NeRF Representation:** Employs Neural Radiance Fields for a continuous, high-quality 3D representation that allows for novel view synthesis.
* **Modular Codebase:** Structured for clarity and extensibility with separate modules for CLIP, NeRF, and the optimization loop.

## How It Works (High-Level)

The core idea is an iterative optimization process where CLIP acts as a "semantic critic" guiding a NeRF model:

1.  **Text Prompt:** A user provides a text description of the desired 3D object or modification.
2.  **CLIP Text Encoding:** CLIP's text encoder converts this prompt into a "target semantic code."
3.  **NeRF Rendering:** The NeRF model (initially untrained for generation, or pre-trained for editing) renders 2D images of its current 3D scene representation from various viewpoints.
4.  **CLIP Image Encoding:** CLIP's image encoder converts these rendered 2D images into their "current semantic code."
5.  **CLIP Loss Calculation:** A loss function (CLIP Loss) measures the "distance" or "dissimilarity" between the "target semantic code" (from the text) and the "current semantic code" (from the rendered images).
6.  **NeRF Optimization:** This calculated CLIP Loss is then used to update (optimize) the internal parameters (weights) of the NeRF model. The goal is to minimize this loss, effectively making NeRF's 3D representation more semantically aligned with the text prompt.
7.  **Iteration:** Steps 3-6 repeat for many thousands of iterations, gradually refining NeRF's 3D scene until it accurately reflects the textual description.

## Getting Started

Follow these steps to set up and run the project locally or in Google Colab.

### Prerequisites

* Python 3.8+
* PyTorch (CUDA-enabled recommended for GPU acceleration)
* A compatible NVIDIA GPU (at least 4GB VRAM recommended, 2GB might work with very small parameters).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Mahdi-Tajdari/3d-model-editor-clip-nerf.git](https://github.com/Mahdi-Tajdari/3d-model-editor-clip-nerf.git)
    cd 3d-model-editor-clip-nerf
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The `openai-clip` package will automatically download necessary CLIP model weights (e.g., `ViT-B/32` or `RN50`) to your `~/.cache/clip` directory.*

### Sample Data Setup

This project uses local sample images for the basic CLIP test.

1.  **Create the `assets/sample_images` directory:**
    ```bash
    mkdir -p assets/sample_images
    ```
2.  **Place sample images:** Download a few sample `.jpg` or `.png` images (e.g., of a red car, a cat, a dog) and place them inside the `assets/sample_images/` folder.
    * Make sure their filenames match those expected in `src/main.py` (e.g., `red_car.jpg`, `cute_cat.jpg`, `dog_on_grass.jpg`).

## Usage

### 1. Run Basic CLIP Test

This script verifies that CLIP is loaded correctly and can perform basic image-text similarity calculations.

```bash
python scripts/run_clip_test.py
