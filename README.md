# AI Face Swap Studio Pro 2.0

> A professional guided workflow for creating composite images, from face swapping to complex AI-driven scene generation.

This project was born from the need for a comprehensive tool that goes beyond simple face swapping, integrating the latest image generation and enhancement technologies into a single, simple, and guided web interface. The application allows you to start with a simple photo of a subject, create a photorealistic background from scratch, enhance details and resolution, and finally apply a targeted face swap and final touches.

## 🚀 Key Features

* **Guided 4-Step Workflow**: The interface guides the user step-by-step, from subject preparation to finalization.
* **Smart Background Removal**: Automatically isolates the subject from their original image using `rembg`.
* **AI Scene Generation**: Utilizes **Stable Diffusion XL** in Inpainting mode to create complex and photorealistic backgrounds from a simple text prompt.
* **Prompt Enhancement with Gemini**: Integrates the Google Gemini API to analyze the subject's image and automatically improve the prompt, creating richer and more coherent descriptions.
* **Upscaling & Detailing**: Leverages **Real-ESRGAN** to increase the scene's resolution and a tiling system with ControlNet to add fine details to the upscaled image.
* **Targeted Face Swap**: Thanks to **InsightFace**, it allows for precise selection of the source and target faces for an accurate swap.
* **Face Restoration with GFPGAN**: Automatically improves the quality and coherence of the swapped face, seamlessly integrating it into the scene.
* **Integrated Meme Studio**: A suite of tools for the final touch, including adding text, stickers (static and animated), color filters, and generating witty captions with AI.

## 🖼️ Application Workflow

The process is divided into four main steps:

1.  **Step 1: Subject Preparation**
    Upload the starting image. The application removes the background and prepares the isolated subject for the next step.

2.  **Step 2: Scene Creation**
    Describe the desired background. The AI (SDXL) generates an image to serve as the background and composes it with the previously prepared subject.

3.  **Step 3: Upscale & Detail**
    The composite scene is upscaled and further detailed with a second AI pass to ensure high resolution and photographic quality.

4.  **Step 4: Final Touches**
    Perform the targeted face swap, apply filters, and add text or stickers to finalize the artwork.

## 🛠️ Technology Stack

This project combines several cutting-edge technologies:

* **Backend**:
    * Python 3
    * Flask (as the web framework)
    * Waitress (as the production WSGI server)

* **AI & Machine Learning**:
    * **PyTorch**: Primary framework for the AI models.
    * **Diffusers (Hugging Face)**: For the `StableDiffusionXLControlNetInpaintPipeline`.
    * **InsightFace**: For face analysis (`buffalo_l`) and the swapping model (`inswapper_128.onnx`).
    * **GFPGAN** & **Real-ESRGAN**: For image restoration and upscaling.
    * **rembg**: For background removal.
    * **Google Gemini API**: For automatic prompt enhancement.

* **Frontend**:
    * HTML5
    * Tailwind CSS
    * JavaScript (vanilla)

## ⚙️ Installation and Startup

To run the project locally, follow these steps:

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/lelus78/FaceSwapApp.git](https://github.com/lelus78/FaceSwapApp.git)
    cd FaceSwapApp
    ```

2.  **Create a Virtual Environment and Install Dependencies**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # On Windows
    pip install -r requirements.txt
    ```
    *(Note: a `requirements.txt` file should be created with `pip freeze > requirements.txt`)*

3.  **Download the AI Models**
    Make sure to download the necessary models and place them in the correct folders (`./models/`, `./models/checkpoints/` etc.), as specified in `server.py`. The main models are:
    * An SDXL Checkpoint (e.g., `sdxl-yamers-realistic5-v5Rundiffusion`)
    * `inswapper_128.onnx`
    * `GFPGANv1.4.pth`
    * `RealESRGAN_x2plus.pth`
    * ControlNet models (`diffusers/controlnet-canny-sdxl-1.0`)

4.  **Set Up the API Key**
    Create a file named `api_key.txt` in the root directory and insert your Google Gemini API key to enable the prompt enhancement feature.

5.  **Start the Application**
    ```bash
    python run.py
    ```
    The application will be available at `http://127.0.0.1:8765`.

## 🔧 Configuration
The main settings for image generation can be found in the `GLOBAL CONFIGURATION` section at the top of the `app/server.py` file.

```python
CFG_MODEL_NAME = "sdxl-yamers-realistic5-v5Rundiffusion"
CFG_SAMPLER = "DPM++"
CFG_SCENE_STEPS = 35
CFG_SCENE_GUIDANCE = 12
CFG_UPSCALE_FACTOR = 1.5
CFG_DETAIL_STEPS = 20
CFG_OVERLAP = 128
```

---

Created with passion by **Emanuele Orlandin**.