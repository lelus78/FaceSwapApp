# AGENTS.md

## Overview

This project includes a chain of specialized AI models ("agents") that work together to create a high-quality, user-guided face swap and compositing experience. Each agent is responsible for a specific stage of the workflow, and together they form a cohesive, multi-step pipeline that transforms raw inputs into photorealistic, stylized final outputs.

---

## 🧠 Agent: Background Remover — `rembg`

**Purpose**: Automatically isolates the subject from the background to prepare for scene generation and compositing.

* **Input**: PNG/JPEG image with subject
* **Output**: Transparent PNG with subject only
* **Model**: `rembg` (based on U2Net)
* **Stage**: First

---

## 🧠 Agent: Prompt Enhancer — Google Gemini API

**Purpose**: Analyzes the subject and augments the user-provided text prompt to produce a richer, context-aware prompt for scene generation.

* **Input**: Subject image + raw prompt text
* **Output**: Enhanced prompt
* **Model/API**: Google Gemini (via API key)
* **Stage**: Second

---

## 🧠 Agent: Scene Generator — SDXL + ControlNet

**Purpose**: Generates a detailed, photorealistic scene around the subject using the enhanced prompt and an optional subject mask.

* **Input**: Masked subject + enriched prompt
* **Output**: New background scene image
* **Models**: Stable Diffusion XL (SDXL), ControlNet (Canny for detail guidance)
* **Stage**: Third

---

## 🧠 Agent: Upscaler & Detailer — Real-ESRGAN + ControlNet

**Purpose**: Upscales the generated image and enhances fine textures, maintaining photorealism and preventing artifacts.

* **Input**: Raw AI-generated image
* **Output**: High-resolution detailed image
* **Models**: Real-ESRGAN, ControlNet (tiled Canny)
* **Stage**: Fourth

---

## 🧠 Agent: Face Swapper — InsightFace

**Purpose**: Replaces the target face with a selected source face using facial recognition and vector-based alignment.

* **Input**: Processed image + face database + index selection
* **Output**: Face-swapped image
* **Model**: InsightFace (`inswapper_128.onnx`)
* **Stage**: Fifth

---

## 🧠 Agent: Face Restorer — GFPGAN

**Purpose**: Restores skin clarity and facial features post-swapping to produce a cohesive, natural look.

* **Input**: Face-swapped image
* **Output**: Restored image with enhanced facial fidelity
* **Model**: GFPGAN v1.4
* **Stage**: Sixth

---

## 🧠 Agent: Finishing Toolkit — HTML5 Canvas (Client‑Side)

**Purpose**: Offers creative finishing tools like text overlay, stickers, meme formatting, and export options. Runs in-browser.

* **Input**: Final image
* **Output**: Edited PNG, MP4 or GIF
* **Tech**: JavaScript + `<canvas>` API + `lottie-web` for animated stickers
* **Stage**: Final (Client-side)

---

## Pipeline Summary

```text
rembg → Gemini → SDXL + ControlNet → Real‑ESRGAN → InsightFace → GFPGAN → canvas
(mask)   (prompt)     (scene)          (detail)       (swap)        (restore)    (edit/export)
```

Each agent is designed to accept the output of the previous one, forming a fluid and responsive AI-driven workflow that runs server-side until the final user-controlled editing phase.

---

## Future Enhancements

* Modular agent registration for plug-and-play inference backends
* Support for multi-agent orchestration using task queues
* Multi-subject pipeline support
* Agent status monitoring with async WebSocket updates

---

## Style & Coding Standards

* All Python code follows [PEP-8](https://peps.python.org/pep-0008/) conventions.
* Functions and agents are organized by responsibility and named accordingly.
* Async tasks (planned) will follow snake\_case naming and centralized logging for monitoring.

---

## Testing Notes

This project currently does not include unit or integration tests for the individual agents.

**Planned testing strategy:**

* Snapshot testing for image output of each agent
* API response validation using Flask test client
* Manual visual regression checks for key image stages