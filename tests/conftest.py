import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_model():
    """Create a mock PyTorch model for testing."""
    model = Mock()
    model.eval = Mock(return_value=model)
    model.train = Mock(return_value=model)
    model.parameters = Mock(return_value=[torch.zeros(1)])
    model.state_dict = Mock(return_value={'layer': torch.zeros(1)})
    return model


@pytest.fixture
def sample_data():
    """Provide sample data for testing."""
    return {
        'features': torch.randn(10, 5),
        'labels': torch.randint(0, 2, (10,)),
        'texts': ['sample text'] * 10
    }


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    dataset = Mock()
    dataset.__len__ = Mock(return_value=100)
    dataset.__getitem__ = Mock(return_value=(torch.randn(5), 0))
    return dataset


@pytest.fixture
def csv_data(temp_dir):
    """Create temporary CSV files for testing."""
    csv_content = "text,label\nSample text 1,0\nSample text 2,1\n"
    
    # Create test CSV files
    for subdir in ['training_data', 'validation_data', 'evaluation_data']:
        dir_path = temp_dir / subdir
        dir_path.mkdir()
        
        for filename in ['related.csv', 'not_related.csv']:
            file_path = dir_path / filename
            file_path.write_text(csv_content)
    
    # Create unlabeled data
    unlabeled_dir = temp_dir / 'unlabeled_data'
    unlabeled_dir.mkdir()
    unlabeled_content = "text\nUnlabeled sample 1\nUnlabeled sample 2\n"
    (unlabeled_dir / 'unlabeled_data.csv').write_text(unlabeled_content)
    
    return temp_dir


@pytest.fixture
def mock_config():
    """Provide a mock configuration object."""
    return {
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 10,
        'device': 'cpu',
        'model_path': 'models/',
        'random_seed': 42
    }


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    import random
    import numpy as np
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def capture_stdout():
    """Capture stdout for testing print statements."""
    from io import StringIO
    
    captured_output = StringIO()
    with patch('sys.stdout', new=captured_output):
        yield captured_output


@pytest.fixture
def mock_file_operations():
    """Mock file operations for testing."""
    with patch('builtins.open', create=True) as mock_open:
        mock_file = Mock()
        mock_open.return_value.__enter__ = Mock(return_value=mock_file)
        mock_open.return_value.__exit__ = Mock(return_value=None)
        yield mock_open, mock_file


@pytest.fixture
def device():
    """Provide the appropriate device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Markers for different test types
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow running tests")