# Data Poisoning Defense System

## Overview

A machine learning security framework designed to detect and mitigate data poisoning attacks in neural networks. This project implements defense mechanisms against adversarial manipulation of training datasets, helping to maintain model integrity and robustness.

## Features

- **Poisoning Detection**: Identifies potentially malicious samples in training data using statistical analysis and anomaly detection techniques
- **Data Sanitization**: Filters and cleanses datasets to remove poisoned samples before model training
- **Robustness Evaluation**: Measures model resilience against various data poisoning attack strategies
- **Multiple Defense Strategies**: Implements several state-of-the-art defense mechanisms including gradient-based detection and outlier removal
- **Attack Simulation**: Built-in tools to simulate common poisoning attacks for testing defense effectiveness
- **Visualization Tools**: Generates plots and metrics to analyze dataset quality and model performance

## Usage Examples

### Basic Poisoning Detection

Run poisoning detection on a dataset:
```
python detect_poisoning.py --dataset ./data/training_data.csv --method statistical
```

### Training with Defense Mechanism

Train a model with active defense:
```
python train_model.py --data ./data/clean_data.csv --defense gradient_filter --epochs 50
```

### Evaluating Model Robustness

Test model against poisoning attacks:
```
python evaluate_robustness.py --model ./models/trained_model.pt --attack_type label_flip
```

### Simulating an Attack

Generate poisoned dataset for testing:
```
python simulate_attack.py --input ./data/original.csv --output ./data/poisoned.csv --poison_rate 0.1
```

### Data Sanitization

Clean a potentially poisoned dataset:
```
python sanitize_data.py --input ./data/suspicious_data.csv --output ./data/cleaned_data.csv --threshold 0.95
```

## License

This project is available for educational and research purposes.
