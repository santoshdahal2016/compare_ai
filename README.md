# Compare_AI

The `compare_ai` package is a modular Python library for managing AI model comparisons. It supports dataset loading, running predictions, annotating results, and generating reports. The package is structured into different modules, each with a specific purpose.

---

## **Table of Contents**
1. [Installation](#installation)
2. [Package Overview](#package-overview)
3. [Modules and Usage](#modules-and-usage)
    - [Repository](#repository)
    - [Actions](#actions)
    - [Data](#data)
    - [Utils](#utils)
    - [Results Manager](#results-manager)
4. [Examples](#examples)
5. [Extending the Package](#extending-the-package)
6. [Troubleshooting](#troubleshooting)

---

## **Installation**
### Updated **Installation** Section

The `compare_ai` package is also available on PyPI, making it easy to install using `pip`.

---

## **Installation**

### Prerequisites
- Python 3.8+

### Installation Methods

#### **Install via `pip`**
The package is hosted on PyPI, so you can install it with:

```bash
pip install compare_ai
```

#### **Install from Source**
1. Clone the repository:
   ```bash
   git clone https://github.com/santoshdahal2016/compare_ai.git
   cd compare_ai
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

3. Activate the environment:
   ```bash
   poetry shell
   ```

---

### Verifying the Installation
Once installed, you can verify the package by running the following command in Python:

```python
import compare_ai

print(compare_ai.__version__)  # Outputs the current version of the package
```

This ensures that the package and its dependencies are correctly installed.

--- 

## **Package Overview**

The package structure is as follows:

```
compare_ai/
├── __init__.py              # Package initializer
├── repository/              # Handles datasets, matrices, and model management
│   ├── __init__.py
│   ├── datasets.py          # Dataset loading and management
│   ├── matrix.py            # Prediction matrix management
│   ├── models.py            # AI model definitions and loading
├── actions/                 # Executes primary workflows
│   ├── __init__.py
│   ├── run_predictions.py   # Runs predictions for datasets and models
│   ├── build_argilla.py     # Annotates results
│   ├── generate_report.py   # Generates reports
├── data/                    # Stores JSON files for datasets and results
│   ├── __init__.py
│   ├── srmh.json            # Default SRMH dataset
│   ├── mmlu.json            # Example dataset
│   └── argilla_final.json   # Annotated data for reports
├── utils.py                 # Helper functions
└── results_manager.py       # Handles result saving and loading
```

---

## **Modules and Usage**

### **Repository**
This module manages datasets, prediction matrices, and AI models.

#### **`datasets.py`**
- **`Dataset.load(name: str)`**: Loads a default dataset (`srmh` or other pre-defined datasets).
- **`Dataset.load_json(filepath: str)`**: Loads a dataset from a JSON file.

#### **`matrix.py`**
- **`Matrix.load(name: str)`**: Loads a predefined prediction matrix.

#### **`models.py`**
- **`Model.load(name: str)`**: Loads a predefined model configuration.
- **`Model.load_json(filepath: str)`**: Loads a model from a JSON file.
- **`Model()`**: Allows dynamic creation of custom models with attributes like `endpoint` and `token`.

---

### **Actions**
This module provides functions to execute key workflows like running predictions, annotating results, and generating reports.

#### **`run_predictions.py`**
- **`run_predictions(matrix: Matrix, test_set: Dataset, models: list[Model])`**:
  Runs predictions using the given matrix, dataset, and list of models. Returns the prediction results.

#### **`build_argilla.py`**
- **`build_argilla(results: ResultsManager)`**:
  Processes prediction results and prepares them for annotation.

#### **`generate_report.py`**
- **`generate_report(annotated_data: dict)`**:
  Generates a report from annotated data.

---

### **Data**
The `data` directory contains JSON files for datasets and results. 

#### Default Files
- **`srmh.json`**: Default SRMH dataset.
- **`mmlu.json`**: Example MMLU dataset.
- **`argilla_final.json`**: Example annotated data for generating reports.

You can add your own JSON files for datasets or results.

---

### **Utils**
The `utils.py` module provides helper functions that simplify common operations like file handling or data validation.

---

### **Results Manager**
The `results_manager.py` module is used to save, load, and manage prediction results.

#### **Methods**
- **`ResultsManager.save(filepath: str)`**: Saves results to a JSON file.
- **`ResultsManager.load(filepath: str)`**: Loads results from a JSON file.

---

## **Examples**

### **Full Workflow Example**
The following example demonstrates loading datasets, running predictions, annotating results, and generating reports.

```python
from compare_ai.repository import datasets, matrix, models
from compare_ai.actions.run_predictions import run_predictions
from compare_ai.actions.build_argilla import build_argilla
from compare_ai.actions.generate_report import generate_report
from compare_ai.results_manager import ResultsManager

# Load datasets
srmh_dataset = datasets.Dataset.load("srmh")
mmlu_dataset = datasets.Dataset.load_json("compare_ai/data/mmlu.json")

# Load prediction matrix
bmgf_matrix = matrix.Matrix.load("bmgf")

# Load models
gpt4_model = models.Model.load("gpt")
llama_model = models.Model.load_json("compare_ai/data/llama.json")

# Run predictions
results = run_predictions(matrix=bmgf_matrix, test_set=srmh_dataset, models=[gpt4_model, llama_model])

# Save and load results
results.save("compare_ai/data/results.json")
loaded_results = ResultsManager.load("compare_ai/data/results.json")

# Build Argilla and generate report
build_argilla(loaded_results)
annotated_data = loaded_results.data
generate_report(annotated_data)
```

---

## **Extending the Package**

1. **Add New Datasets**:
   - Place the JSON file in the `data` directory.
   - Load it using `Dataset.load_json(filepath)`.

2. **Add New Models**:
   - Define a model configuration in a JSON file.
   - Load it using `Model.load_json(filepath)`.

3. **Customize Predictions**:
   - Modify the `run_predictions` function in `run_predictions.py` to include custom logic.

---

## **Troubleshooting**

### Common Issues
1. **File Not Found**: Ensure JSON files are placed in the correct `data` directory.
2. **Invalid JSON Format**: Validate JSON files using an online validator or Python's `json` module.
3. **Missing Dependencies**: Ensure all dependencies are installed via `poetry install`.

### Debugging Tips
- Use logging to debug workflows:
  ```python
  import logging
  logging.basicConfig(level=logging.DEBUG)
  ```

---

## **Contributing**

Contributions are welcome! If you encounter issues or have feature suggestions:
1. Create a GitHub issue.
2. Submit a pull request.

---

Let me know if you need enhancements or additional sections!