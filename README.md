# 🏥 Pharmacy Shelf Monitoring System (powered by YOLO-World)

This repository contains a full-stack application for real-time pharmacy shelf monitoring using **YOLO-World** for zero-shot object detection. It bridges the gap between high-accuracy zero-shot models and real-time performance to detect misplaced drugs and monitor shelf inventory.

## 🚀 Overview

The system is designed to provide an end-to-end workflow:
1. **Setup (Define Ground Truth):** Define Regions of Interest (ROIs) and ground truth mapping for pharmacy shelves.
2. **Scan (Live Comparison):** Run real-time detection using YOLO-World against the live camera stream.
3. **Alert (Mismatch Detection):** Detect anomalies or misplaced items.

## 🏗️ Architecture

The system consists of three main components orchestrated via Docker:
*   **FastAPI Backend (`/fastapi`):** A Python-based machine learning service that handles YOLO-World inference and image processing.
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
    git clone https://github.com/CommonCapital/Zero-Shot-Object-Detection-with-YOLO-World.git
    cd Zero-Shot-Object-Detection-with-YOLO-World
    ```

2.  **Start the services:**
    ```bash
    docker-compose up --build
    ```

3.  **Access the applications:**
    *   **Frontend Dashboard:** [http://localhost:3000](http://localhost:3000)
    *   **FastAPI Backend:** [http://localhost:8000](http://localhost:8000)
    *   **Database:** `localhost:5432`

### Model Weights
The inference engine uses the YOLO-World-L (Large) weights. Ensure that the `my_yolo_world_model.pt` file is present in your root directory to allow the FastAPI service to load the local model for offline inference.

## 🧠 Legacy Notebook
The original Jupyter Notebook (`zero-shot-object-detection-with-yolo-world.ipynb`) used for exploring and prototyping the YOLO-World model is still included in the repository for reference.

## 🤝 Acknowledgments
*   [YOLO-World](https://github.com/AILab-CVC/YOLO-World)
*   [Roboflow Inference & Supervision](https://github.com/roboflow/supervision)
*   [Ultralytics](https://github.com/ultralytics/ultralytics)
# -Pharmacy-Shelf-Monitoring-System-powered-by-YOLO-World-
