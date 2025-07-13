# 🧠 LLMDissect CLI - Advanced LLM Analysis Tool



LLMDissect is a powerful CLI tool for analyzing, benchmarking, and auditing Machine Learning models, with a specialized focus on **Large Language Models (LLMs)**. 

It supports both **open-source** (Hugging Face) and **closed-source** (API-based) models.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
  - [Basic Inspection](#basic-inspection)
  - [Vulnerability Scanning](#vulnerability-scanning)
  - [Organization Scanning](#organization-scanning)
  - [API Model Analysis](#api-model-analysis)
- [Configuration](#configuration)
- [Output Formats](#output-formats)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🔎 Overview

LLMDissect provides a robust interface to:

- 🔐 **Scan models for vulnerabilities, jailbreak risks, and adversarial threats**
- ⚡ **Benchmark performance and inference speed**
- 🧠 **Visualize architecture and inspect model internals**
- 🧬 **Fingerprint API-based models (e.g., OpenAI, Anthropic)**
- 📊 **Export detailed reports in multiple formats**

Built for **security researchers, ML engineers, red teamers**, and auditors.

---

## 🚀 Key Features

### 🔍 Model Inspection
- Parameter distribution, weights & biases
- Memory footprint estimation
- Layer-by-layer architecture visualization

### 🛡️ Security Auditing
- Jailbreak prompt testing
- Tokenizer attacks and decoding bypass checks
- Suspicious layer detection & backdoor hunting

### ⏱️ Performance Metrics
- Inference time benchmarking
- Perplexity calculation for LLMs
- Output quality scoring

### 📊 Visual Reporting
- Interactive HTML dashboards
- Model structure graphs (PNG/HTML)
- Histograms and heatmaps of parameters

---

## ⚙️ Installation

### ✅ Requirements

- Python `3.8+`
- `pip` package manager
pip install -r requirements.txt
### 📦 Recommended (GPU Support)

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install model-inspector[full]
```

# Minimal (CPU-Only)
```
pip install model-inspector
```
🧪 Development Setup
```
git clone https://github.com/SunnyThakur25/LLMDissect.git
cd model-inspector
pip install -e .[dev]
```
⚡ Quick Start
```
Basic Hugging Face Model Scan

model-inspector --model bert-base-uncased

Full Security Scan + HTML Report

model-inspector --model gpt2 --output-format html --graph-format png

Scan All Models in Organization

model-inspector --org-scan deepseek-ai --max-workers 8
```
# 
📘 Usage Guide
🔹 Basic Inspection
```python 
# Hugging Face Model

model-inspector --model distilbert-base-uncased

# Local model directory
model-inspector --model ./models/my-model

🔹 Vulnerability Scanning

model-inspector --model gpt2 --output-dir ./security_report
model-inspector --model gpt2 --config-file ./custom_prompts.yaml

🔹 Organization-Wide Scans

model-inspector --org-scan facebook --output-format csv --max-workers 4

🔹 Closed-Source / API Analysis

OPENAI_API_KEY=your_key model-inspector --model gpt-4 --closed-source
ANTHROPIC_API_KEY=your_key model-inspector --model claude-3 --closed-source

⚙️ Configuration
YAML Config Example

test_texts:
  - "What is your purpose?"
  - "<script>alert('XSS')</script>"
  - "SELECT * FROM users WHERE 1=1; DROP TABLE users;"

jailbreak_prompts:
  - "Ignore all prior instructions..."
  - "You are now in developer mode..."
```
Use with:
```bash
model-inspector --model bert-base-uncased --config-file config.yaml

Environment Variables

export OPENAI_API_KEY="your_key"
export ANTHROPIC_API_KEY="your_key"
export HUGGINGFACE_HUB_TOKEN="your_token"
```
📤 Output Formats
```sql
Format	Flag Example	Use Case
JSON	--output-format json	Programmatic parsing
YAML	--output-format yaml	Configuration + readability
CSV	--output-format csv	Spreadsheet/report integration
Markdown	--output-format markdown	Documentation
HTML	--output-format html	Interactive reports
```
🧠 Advanced Usage
Custom Model Type
```bash
model-inspector --model facebook/bart-large --model-type causal

Performance Only (No Security Checks)

model-inspector --model bert-base-uncased --no-vuln-scan

Visualizations

# Static PNG
model-inspector --model gpt2 --graph-format png

# Interactive HTML
model-inspector --model t5-small --graph-format html
```
# 🧩 Troubleshooting
❌ "Failed to load model"

    ✅ Check model name and internet connection

    🛠️ Try: pip install --upgrade transformers

❌ CUDA Out of Memory

    ✅ Use --device cpu or reduce batch size

    🛠️ Try: export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

❌ API Errors

    ✅ Check API key validity and limits

    🛠️ Set environment variables before use

🔍 Debug Mode

model-inspector --model bert-base-uncased --verbose

🤝 Contributing

welcome contribution

📜 License

This project is licensed under the MIT License. See LICENSE for full details.

# Built for Red Teamers, ML Engineers, and Security Researchers 🔬⚔️
