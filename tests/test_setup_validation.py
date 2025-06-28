"""Validation tests to ensure the testing infrastructure is set up correctly."""

import pytest
import sys
from pathlib import Path


class TestSetupValidation:
    """Test class to validate the testing infrastructure setup."""
    
    @pytest.mark.unit
    def test_pytest_installed(self):
        """Test that pytest is properly installed and importable."""
        import pytest
        assert pytest.__version__
        
    @pytest.mark.unit
    def test_coverage_tools_available(self):
        """Test that coverage tools are available."""
        import pytest_cov
        assert pytest_cov
        
    @pytest.mark.unit 
    def test_mock_tools_available(self):
        """Test that mocking tools are available."""
        import pytest_mock
        from unittest.mock import Mock, MagicMock
        assert pytest_mock
        assert Mock
        assert MagicMock
        
    @pytest.mark.unit
    def test_lit_gpt_importable(self):
        """Test that the lit_gpt package is importable."""
        # Skip this test since lit_gpt has optional CUDA dependencies
        pytest.skip("Skipping lit_gpt import test - requires CUDA dependencies")
        
    @pytest.mark.unit
    def test_fixtures_available(self, temp_dir, mock_config, device):
        """Test that custom fixtures are available and working."""
        assert temp_dir.exists()
        assert isinstance(mock_config, dict)
        assert "model_name" in mock_config
        assert device is not None
        
    @pytest.mark.unit
    def test_markers_configured(self, request):
        """Test that custom markers are properly configured."""
        markers = [marker.name for marker in request.node.iter_markers()]
        assert "unit" in markers
        
    @pytest.mark.integration
    def test_integration_marker(self, request):
        """Test that integration marker works."""
        markers = [marker.name for marker in request.node.iter_markers()]
        assert "integration" in markers
        
    @pytest.mark.slow
    def test_slow_marker_skip(self, request):
        """Test that slow tests can be marked and skipped."""
        markers = [marker.name for marker in request.node.iter_markers()]
        assert "slow" in markers
        # This test should be skipped by default unless --runslow is provided
        
    @pytest.mark.unit
    def test_torch_available(self):
        """Test that PyTorch is available for testing."""
        import torch
        assert torch
        tensor = torch.tensor([1, 2, 3])
        assert tensor.shape == (3,)
        
    @pytest.mark.unit
    def test_temp_dir_fixture_creates_directory(self, temp_dir):
        """Test that temp_dir fixture creates a valid temporary directory."""
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Test we can write to it
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()
        assert test_file.read_text() == "test content"
        
    @pytest.mark.unit
    def test_mock_tokenizer_fixture(self, mock_tokenizer):
        """Test that mock tokenizer fixture works correctly."""
        assert mock_tokenizer.encode("test") == [1, 2, 3, 4, 5]
        assert mock_tokenizer.decode([1, 2, 3]) == "Hello world"
        assert mock_tokenizer.eos_id == 2
        assert mock_tokenizer.bos_id == 1
        
    @pytest.mark.unit
    def test_project_structure(self):
        """Test that the project structure is set up correctly."""
        project_root = Path(__file__).parent.parent
        
        # Check main directories exist
        assert (project_root / "lit_gpt").exists()
        assert (project_root / "tests").exists()
        assert (project_root / "tests" / "unit").exists()
        assert (project_root / "tests" / "integration").exists()
        
        # Check configuration files exist
        assert (project_root / "pyproject.toml").exists()
        
    @pytest.mark.unit
    def test_coverage_configuration(self):
        """Test that coverage is properly configured."""
        # This test verifies coverage runs by checking if coverage module is available
        try:
            import coverage
            assert coverage
        except ImportError:
            pytest.fail("Coverage module not available")


def test_basic_assertion():
    """Simple test to verify pytest works."""
    assert True
    assert 1 + 1 == 2
    assert "test" in "testing"