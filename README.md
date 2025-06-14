🎨 AI Face Swap Studio Pro 2.0
A professional, guided workflow for creating composite images, from face swapping to complex AI-driven scene generation and creative finishing. This project serves as both a powerful tool and a practical example of chaining multiple AI models in a web application.

This project was born from the need for a tool that goes beyond simple face swapping, integrating modern image generation and enhancement technologies into a single, intuitive web interface. It demonstrates how to build a multi-stage AI pipeline, where the output of one model becomes the input for the next, to achieve a complex creative goal.

🚀 Key Features
✨ Guided 4-Step Workflow: A stateful frontend guides the user step-by-step, making the complex process manageable.

✂️ Smart Background Removal: Automatically isolates the subject using rembg, providing a clean slate for composition.

🖼️ AI Scene Generation: Leverages Stable Diffusion XL in Inpainting mode to create photorealistic backgrounds from a text prompt.

🧠 Prompt Enhancement with Gemini: Integrates the Google Gemini API to analyze the subject's image and automatically improve user prompts.

🔍 Hi-Res Upscaling & Detailing: Uses Real-ESRGAN for upscaling and a tiled ControlNet (Canny Edge) pass to add fine details to large images without memory overload.

🎭 Targeted Face Swap: Implements InsightFace for precise, index-based selection of source and target faces.

🌟 Face Restoration with GFPGAN: Automatically improves the quality and coherence of the swapped face.

🎬 Creative Finishing Studio: A full suite of client-side tools for final adjustments:

Text & Meme Tools: Add customizable text with controls for font, size, color, and stroke.

Sticker Gallery: Supports static (PNG), animated (WebM), and vector (Lottie/.tgs) stickers with full manipulation controls.

Multi-Format Export: Save the final artwork as a static PNG or as a video animation in MP4 or GIF format.

🛠️ Technology Stack
This project combines several cutting-edge technologies:

Backend:

Python 3.9+

Flask (as a lightweight web framework)

Waitress (as a production-ready WSGI server)

python-dotenv (for managing environment variables)

imageio-ffmpeg (for server-side video conversion)

AI & Machine Learning:

PyTorch

Hugging Face Diffusers (for Stable Diffusion & ControlNet)

InsightFace (for face analysis and swapping)

GFPGAN & Real-ESRGAN (for image restoration and upscaling)

rembg (for background removal)

Google Gemini API

Frontend:

HTML5 & CSS3

Tailwind CSS (for rapid UI development)

Vanilla JavaScript (ES6 Modules, no build step required)

lottie-web (for rendering Lottie animations)

🧠 Technical Deep Dive & Architectural Choices
This project is a practical case study in building a multi-stage AI pipeline, balancing server-side processing with client-side interactivity.

The AI Pipeline: The application's core is a chain of specialized models where each step builds upon the last: rembg → Gemini → Stable Diffusion XL → Real-ESRGAN → ControlNet → InsightFace → GFPGAN. This sequential process is crucial because no single, monolithic model can perform such a wide array of tasks. The application manages the data flow, for instance, by using the RGBA image from rembg to create a mask for the Stable Diffusion composition. This modularity also means any model in the chain can be updated or swapped (e.g., replacing Real-ESRGAN with a newer upscaler) without rebuilding the entire application logic.

Backend Philosophy: Flask was chosen for its simplicity and minimal boilerplate, allowing for rapid API development. Paired with Waitress, a production-ready WSGI server, it provides a stable environment without the complexity of larger frameworks like Django. The decision to load all models into VRAM at startup is a deliberate trade-off: it leads to higher initial memory consumption but guarantees low-latency responses for each API call, which is critical for a responsive user experience. This architecture is best suited for deployment on a machine with a dedicated, high-VRAM GPU.

Frontend State Management: The choice of vanilla JavaScript avoids a complex build pipeline (like Webpack or Vite) and keeps the project accessible to developers who may not be familiar with modern frontend frameworks. State is managed through a simple dom object and a few global variables that hold the image data (as Blobs) from each step. While this is highly effective for a single-user, linear workflow, it could become challenging to manage in a more complex application with non-linear steps. A developer looking to expand the project might consider this a prime area for refactoring into a lightweight state library or a component-based framework like Svelte or Vue.

Client-Side Finishing: This hybrid approach significantly enhances performance and reduces server costs. By offloading all real-time rendering of text, stickers, and filters to the user's GPU via the <canvas> API, the server is freed from expensive and frequent image manipulation requests. The final static image is exported using canvas.toDataURL(), a purely client-side operation. For animated exports, the frontend uses the MediaRecorder API to capture a high-framerate stream from the canvas, which is then sent to the server as a lightweight WebM blob for efficient conversion to MP4 or GIF using ffmpeg. This strategy optimally balances client and server resources.

⚙️ Installation and Setup
To run the project locally, follow these steps:

Clone the Repository

git clone https://github.com/lelus78/FaceSwapApp.git
cd FaceSwapApp


Create a Virtual Environment & Install Dependencies

# Create the virtual environment
python -m venv venv

# Activate it
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt


Download AI Models
You must download the models manually and place them in the correct folders as referenced in app/server.py.

SDXL Checkpoint: Place your .safetensors model file in ./models/checkpoints/.

InsightFace Swapper: Place inswapper_128.onnx in ./models/.

Enhancement Models: Place GFPGANv1.4.pth and RealESRGAN_x2plus.pth in ./models/.

Other models (ControlNet, InsightFace analysis) will be downloaded automatically on first run.

Set Up Environment Variables
Create a file named .env in the project's root directory. Add your Google Gemini API key to it:

GEMINI_API_KEY="YOUR_API_KEY_HERE"


Note: Remember to add .env to your .gitignore file to prevent accidentally committing your secret keys.

Run the Application

python run.py


The application will be available at http://127.0.0.1:8765.

🔧 Contributing & Future Ideas
Contributions are welcome! This project has a lot of potential for expansion. Fork the repo, create a new branch for your feature, and submit a pull request.

Potential Improvements & Feature Roadmap
[ ] Asynchronous AI Tasks: Move long-running AI jobs to a background worker queue (e.g., Celery) to prevent server timeouts and improve UI responsiveness.

[ ] Advanced State Management: Refactor the frontend's global variables into a more robust state management pattern.

[ ] Backend Refactoring: Split server.py into multiple Flask Blueprints for better organization.

[ ] More Model Choices: Add UI options to allow the user to select different SDXL checkpoints or upscalers.

[ ] Multi-Subject Support: Allow the composition of multiple subjects into a single scene.

[ ] Improved Sticker Controls: Add more advanced manipulation features like perspective distortion.