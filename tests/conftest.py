"""Shared pytest fixtures and configuration for all tests."""

import os
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any
import pytest
import torch
from unittest.mock import MagicMock, Mock


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide a mock configuration dictionary for testing."""
    return {
        "model_name": "test_model",
        "block_size": 128,
        "vocab_size": 50257,
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 128,
        "rotary_percentage": 0.25,
        "parallel_residual": True,
        "bias": False,
        "lm_head_bias": False,
        "n_query_groups": 1,
        "shared_attention_norm": False,
        "norm_eps": 1e-5,
        "intermediate_size": None,
        "condense_ratio": 1,
    }


@pytest.fixture
def mock_model_config(mock_config):
    """Create a mock model configuration object."""
    from lit_gpt.config import Config
    return Config(**mock_config)


@pytest.fixture
def device() -> torch.device:
    """Return the appropriate device for testing (CPU or CUDA if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_tensor(device) -> torch.Tensor:
    """Create a sample tensor for testing."""
    return torch.randn(2, 10, 128, device=device)


@pytest.fixture
def mock_tokenizer() -> Mock:
    """Create a mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
    tokenizer.decode = Mock(return_value="Hello world")
    tokenizer.eos_id = 2
    tokenizer.bos_id = 1
    return tokenizer


@pytest.fixture
def mock_checkpoint(temp_dir: Path) -> Path:
    """Create a mock checkpoint file for testing."""
    checkpoint_path = temp_dir / "mock_checkpoint.pt"
    checkpoint_data = {
        "model_state_dict": {"layer1.weight": torch.randn(10, 10)},
        "optimizer_state_dict": {"param_groups": [{"lr": 0.001}]},
        "epoch": 5,
        "global_step": 1000,
    }
    torch.save(checkpoint_data, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def sample_text_data() -> str:
    """Provide sample text data for testing."""
    return """The quick brown fox jumps over the lazy dog.
    This is a sample text for testing purposes.
    Machine learning models need diverse training data."""


@pytest.fixture
def mock_wandb(monkeypatch):
    """Mock wandb for testing without actual logging."""
    mock_wandb_module = MagicMock()
    mock_wandb_module.init = MagicMock()
    mock_wandb_module.log = MagicMock()
    mock_wandb_module.finish = MagicMock()
    monkeypatch.setattr("wandb", mock_wandb_module)
    return mock_wandb_module


@pytest.fixture
def mock_dataset(temp_dir: Path) -> Path:
    """Create a mock dataset file for testing."""
    dataset_path = temp_dir / "mock_dataset.bin"
    # Create a simple binary file with some data
    data = torch.randint(0, 50257, (1000,), dtype=torch.uint16)
    data.numpy().tofile(dataset_path)
    return dataset_path


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables before each test."""
    # Store original environment
    original_env = os.environ.copy()
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def capture_stdout(monkeypatch):
    """Capture stdout for testing print statements."""
    from io import StringIO
    buffer = StringIO()
    monkeypatch.setattr("sys.stdout", buffer)
    return buffer


# Markers for different test types
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests")


# Skip slow tests by default unless --runslow is provided
def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle slow tests."""
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)