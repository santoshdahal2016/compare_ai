import pytest
from compare_ai.repository.evaluation_datasets.dataset_mmlu import MMLUDataset
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_dataset():
    return {
        "data": [
            {
                "question": "What is the capital of France?",
                "options": ["London", "Paris", "Berlin", "Madrid"]
            },
            {
                "question": "Which planet is closest to the Sun?",
                "options": ["Venus", "Mercury", "Mars", "Earth"]
            }
        ]
    }

@pytest.fixture
def mmlu_dataset():
    return MMLUDataset()

def test_mmlu_dataset_initialization():
    dataset = MMLUDataset()
    assert dataset.name == "MMLU"
    assert dataset.description == "Massive Multitask Language Understanding benchmark"
    assert len(dataset.system_prompts) == 2
    assert isinstance(dataset.entries, list)
    assert isinstance(dataset.inputs, list)

@patch('compare_ai.repository.datasets.dataset_mmlu.load_dataset')
def test_load_dataset(mock_load_dataset, mock_dataset):
    mock_load_dataset.return_value = mock_dataset
    dataset = MMLUDataset()
    assert dataset.entries == mock_dataset["data"]

@patch('compare_ai.repository.datasets.dataset_mmlu.load_dataset')
def test_preprocess_entries(mock_load_dataset, mock_dataset):
    mock_load_dataset.return_value = mock_dataset
    dataset = MMLUDataset()
    
    processed_entries = dataset._preprocess_entries()
    assert len(processed_entries) == 2
    
    first_entry = processed_entries[0]
    assert "prompt" in first_entry
    assert "system_prompt" in first_entry
    assert "Question: What is the capital of France?" in first_entry["prompt"]
    assert "Options:\nLondon\nParis\nBerlin\nMadrid" in first_entry["prompt"]

def test_load_dataset_error():
    with patch('compare_ai.repository.datasets.dataset_mmlu.load_dataset', 
               side_effect=Exception("Test error")):
        with pytest.raises(RuntimeError) as exc_info:
            MMLUDataset()
        assert "Failed to load MMLU dataset" in str(exc_info.value)
