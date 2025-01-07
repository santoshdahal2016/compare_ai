# Compare_AI

The `compare_ai` package is a modular Python library for managing AI model comparisons. It supports dataset loading, running predictions, annotating results, and generating reports. The package is structured into different modules, each with a specific purpose.

---

## **Table of Contents**
1. [Installation](#installation)
2. [Package Overview](#package-overview)
3. [Modules and Usage](#modules-and-usage)
5. [Extending the Package](#extending-the-package)
6. [Troubleshooting](#troubleshooting)
7. [Providers](#providers)

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


## **Modules and Usage**

## **Examples**

### **Full Workflow Example**
The following example demonstrates loading datasets, running predictions, annotating results, and generating reports.

```python
from compare_ai.models import  ModelRegistry, TaskType
from compare_ai.datasets import Evaluationdataset , DatasetRegistry


from compare_ai.actions import run_prediction

registry = ModelRegistry()

text_models = registry.find_models(
    task=TaskType.TEXT_GENERATION)


dataset_registry = DatasetRegistry()
translation_dataset = dataset_registry.find_model("nepali_translation")


result = run_prediction(models=text_models, dataset=translation_dataset)


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

## **Troubleshooting**


## **Contributing**

Contributions are welcome! If you encounter issues or have feature suggestions:
1. Create a GitHub issue.
2. Submit a pull request.

---

## **Providers**

The `compare_ai` package uses a provider-based architecture to support multiple AI model providers. Each provider is implemented as a separate module and can be installed independently.

### **Available Providers**

- OpenAI (`compare_ai[openai]`)

### **Installing Providers**

Install specific providers using Poetry:

```bash
# Install with OpenAI support
poetry install --extras openai

# Install multiple providers
poetry install --extras "openai"

# Install all providers
poetry install --extras all
```

### **Using Providers**

```python

from compare_ai.repository.evaluation_datasets import  MMLUDataset

from compare_ai.repository.models import ModelRegistry
from compare_ai.repository.models.definitions import TaskType, Modality


from compare_ai.repository.actions import run_prediction

# Initialize with provider configurations
config = {
    "openai": {
        "api_key": "your-openai-key"
    },
    "anthropic": {
        "api_key": "your-anthropic-key"
    }
}

# Create registry with configurations
registry = ModelRegistry(config)

# Find models for specific tasks
text_models = registry.find_models(
    task=TaskType.TEXT_GENERATION,
    modality=Modality.TEXT
)


eval_dataset = MMLUDataset(system_prompts=["","You are an expert academic assistant who answers questions precisely and accurately"])




result = run_prediction(models=text_models,eval_dataset=eval_dataset)
```

### **Creating New Providers**
