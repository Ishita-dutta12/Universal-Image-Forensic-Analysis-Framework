# 🔍 UIFAF — Universal Image Forensic Analysis Framework

**UIFAF (Universal Image Forensic Analysis Framework)** is a modular AI-powered image forensic framework designed to analyze image authenticity and detect AI-generated or manipulated images using deep learning, metadata analysis, forensic feature extraction, attention mechanisms, and evidence fusion.

The framework generates detailed forensic reports containing confidence scores, manipulation predictions, localization results, metadata analysis, and explainable forensic evidence through a FastAPI-based inference service.

---

## ✨ Features

- Modular image forensic pipeline
- AI-generated image detection
- AI-edited image detection
- Face swap detection
- Deepfake detection
- Copy-Move forgery detection
- Image splicing detection
- Object removal detection
- Metadata tampering analysis
- Screenshot/Re-photographed image detection
- Computer Generated Image (CGI) detection
- Explainable forensic reports
- Heatmap and overlay localization
- FastAPI REST API
- Image upload support
- External image URL support
- JSON forensic report generation

---

## 📂 Project Structure

```text
UIFAF
│
├── api/
├── attention/
├── backbone/
├── configs/
├── cross_representation/
├── decision/
├── evaluation/
├── integrity/
├── metadata/
├── pipeline/
├── preprocessing/
├── report_generator/
├── representation/
├── training/
├── tests/

---

## 🔬 Detection Categories

The framework is designed to classify images into the following forensic categories:

- Authentic
- AI Generated
- AI Edited
- Face Swap
- Deepfake
- Copy-Move Forgery
- Image Splicing
- Object Removal
- Metadata Tampering
- Screenshot / Re-photographed
- Computer Generated Image (CGI)
- Unknown

---

## ⚙ Current Pipeline

The current pipeline performs:

1. Image Validation
2. Image Decoding
3. Metadata Extraction
4. Backbone Feature Extraction
5. Attention Enhancement
6. Multi-Representation Learning
7. Forensic Reasoning
8. Decision Making
9. Localization
10. Report Generation

---

## 📑 Example JSON Report

The API returns a comprehensive forensic report including:

- File Information
- Evidence Scores
- Evidence Summary
- Risk Assessment
- Confidence Assessment
- Final Verdict
- Primary & Secondary Classification
- Localization Results
- Metadata Summary
- Raw Model Outputs
- Threshold Information
- System Information
- Performance Statistics

---

## 🚀 API

Launch the FastAPI server:

```bash
uvicorn api.app:app --reload

Open Swagger UI:

- **Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

The API supports:

- Upload image directly from local storage
- Analyze images from external URLs
- Returns structured JSON forensic reports

---

## 🛠 Technology Stack

| Category | Technologies |
| :--- | :--- |
| **Programming** | Python 3.11 |
| **Deep Learning** | PyTorch, TorchVision |
| **API** | FastAPI, Uvicorn |
| **Image Processing** | Pillow, NumPy |
| **Visualization** | Matplotlib |
