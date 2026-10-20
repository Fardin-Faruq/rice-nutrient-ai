# AI-Enabled Rice Nutrient Deficiency Detection

An image-based machine learning project exploring the classification of rice leaves into four nutrient-status categories: Healthy, Nitrogen, Phosphorus, and Potassium.

## Project Overview

The project includes dataset inspection, image preprocessing, duplicate analysis, dataset splitting, model training, and prediction utilities.

The intended long-term goal is to explore edge-AI deployment for rice leaf nutrient deficiency detection using an ESP32-compatible platform.

## Classification Categories

- Healthy
- Nitrogen deficiency
- Phosphorus deficiency
- Potassium deficiency

## Configuration

The current configuration specifies:

- Image input: 96 x 96 pixels
- Image channels: 3 (RGB)
- Random seed: 42
- Dataset split: 70% training, 15% validation, 15% testing
- Confidence threshold: 0.65

These are configured project settings; they do not by themselves establish model performance or deployment readiness.

## Repository Structure

- `src/` - Python scripts for dataset preparation, inspection, preprocessing comparison, training, evaluation, and prediction.
- `outputs/` - Dataset audit reports and visualization outputs.
- `outputs_4class_v1/reports/` - Reports associated with the four-class dataset workflow.
- `config.json` - Project configuration.
- `requirements.txt` - Python dependencies for the currently listed data-processing workflow.
- `.gitignore` - Excludes local datasets, virtual environments, and model artifacts.

## Setup

Use Python 3.11 in a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place the dataset in the expected local directory structure before running dataset-processing scripts. The raw dataset and generated image splits are intentionally excluded from Git.

## Model and Evaluation

Model artifacts are not included in this repository at present. Training results, evaluation metrics, and deployment compatibility should be documented here after the corresponding artifacts and results are added and verified.

## Hardware Deployment Status

ESP32 edge deployment is a project goal. Hardware compatibility, inference latency, memory requirements, and on-device accuracy still need to be validated before claiming successful deployment.

## Reproducibility

The configuration records the class order, input dimensions, random seed, split ratios, and confidence threshold. Exact reproducibility also depends on the dataset version, installed package versions, and training configuration.

## License

Add an appropriate license after confirming the permissions and licensing terms for the source dataset and project code.
