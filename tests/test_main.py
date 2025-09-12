"""
Basic tests for the main module.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for testing
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def test_main_import():
    """Test that main module can be imported."""
    import main
    assert hasattr(main, 'main')


def test_src_package():
    """Test that src package can be imported."""
    import src
    assert hasattr(src, '__version__')


class TestMainFunction:
    """Test cases for main function."""
    
    def test_main_function_exists(self):
        """Test that main function exists."""
        import main
        assert callable(main.main)
        
    def test_main_function_runs(self, capsys):
        """Test that main function runs without error."""
        import main
        main.main()
        captured = capsys.readouterr()
        assert "Welcome to LMU_S4 Project!" in captured.out
