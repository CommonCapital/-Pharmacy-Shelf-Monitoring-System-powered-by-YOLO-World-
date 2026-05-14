# 🏥 Pharmacy Shelf Monitoring System (powered by Grounding DINO)

This repository contains a full-stack application for real-time pharmacy shelf monitoring using **Grounding DINO** for zero-shot object detection. It bridges the gap between high-accuracy zero-shot models and real-time performance to detect misplaced drugs and monitor shelf inventory.

## 🚀 Overview

The system is designed to provide an end-to-end workflow:
1. **Setup (Define Ground Truth):** Define Regions of Interest (ROIs) and ground truth mapping for pharmacy shelves.
2. **Scan (Live Comparison):** Run real-time detection using Grounding DINO against the live camera stream.
3. **Alert (Mismatch Detection):** Detect anomalies or misplaced items.

## 🏗️ Architecture

The system consists of three main components orchestrated via Docker:
*   **FastAPI Backend (`/fastapi`):** A Python-based machine learning service that handles Grounding DINO inference and image processing.
*   **Next.js Frontend (`/nextjs`):** A React-based web dashboard for drawing detection zones, configuring the layout, and viewing live detection results.
*   **PostgreSQL Database:** Stores the ground truth configuration, slot coordinates, and application state.

## 🛠️ Getting Started

### Prerequisites
*   [Docker](https://docs.docker.com/get-docker/) and Docker Compose
*   (Optional) Python 3.10+ and Node.js for local development outside Docker

### Running the Application

The easiest way to run the entire system is using Docker Compose.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/CommonCapital/Zero-Shot-Object-Detection-with-Grounding-DINO.git
    cd Zero-Shot-Object-Detection-with-Grounding-DINO
    ```

2.  **Start the services:**
    ```bash
    docker-compose up --build
    ```

3.  **Access the applications:**
    *   **Frontend Dashboard:** [http://localhost:3000](http://localhost:3000)
    *   **FastAPI Backend:** [http://localhost:8000](http://localhost:8000)
    *   **Database:** `localhost:5432`

### Model Configuration
The inference engine uses the `IDEA-Research/grounding-dino-tiny` model from the Transformers library. This allows for lightweight, high-speed zero-shot detection.

## 🧠 Legacy Notebook
The original Jupyter Notebook (`zero-shot-object-detection-with-yolo-world-legacy.ipynb`) used for exploring and prototyping the YOLO-World model is still included in the repository for reference.

## 🤝 Acknowledgments
*   [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
*   [Hugging Face Transformers](https://github.com/huggingface/transformers)
*   [Roboflow Supervision](https://github.com/roboflow/supervision)
