"""
Tests for Model Loading Parameter Selection
===========================================

This suite verifies the rules used to derive model-loading arguments for local
inference, especially around hardware and compatibility constraints.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.core import huggingface_utils

class TestModelLoadingParams(unittest.TestCase):
    @patch('src.core.huggingface_utils.os.listdir')
    @patch('src.core.huggingface_utils.pipeline')
    @patch('src.core.huggingface_utils.snapshot_download')
    @patch('src.core.huggingface_utils.is_model_downloaded')
    @patch('src.core.huggingface_utils.os.path.exists')
    def test_load_model_optimizations(self, mock_exists, mock_downloaded, mock_snapshot, mock_pipeline, mock_listdir):
        mock_exists.return_value = False
        mock_downloaded.return_value = True
        mock_listdir.return_value = ["snapshot-1"]
        
        huggingface_utils.load_model("test-model", "image-to-text", device=0)
        
        # Verify pipeline was called with optimizations
        args, kwargs = mock_pipeline.call_args
        self.assertEqual(kwargs.get('model_kwargs', {}).get('low_cpu_mem_usage'), True)
        self.assertEqual(kwargs.get('torch_dtype'), "auto")
        self.assertEqual(kwargs.get('device_map'), "auto")
        self.assertIsNone(kwargs.get('device'))

    @patch('src.core.huggingface_utils.os.listdir')
    @patch('src.core.huggingface_utils.pipeline')
    @patch('src.core.huggingface_utils.snapshot_download')
    @patch('src.core.huggingface_utils.is_model_downloaded')
    @patch('src.core.huggingface_utils.os.path.exists')
    def test_load_model_cpu_params(self, mock_exists, mock_downloaded, mock_snapshot, mock_pipeline, mock_listdir):
        mock_exists.return_value = False
        mock_downloaded.return_value = True
        mock_listdir.return_value = ["snapshot-1"]
        
        huggingface_utils.load_model("test-model", "image-to-text", device=-1)
        
        # Verify pipeline was called with optimizations but NO device_map for CPU
        args, kwargs = mock_pipeline.call_args
        self.assertEqual(kwargs.get('model_kwargs', {}).get('low_cpu_mem_usage'), True)
        self.assertEqual(kwargs.get('torch_dtype'), "auto")
        self.assertIsNone(kwargs.get('device_map'))
        self.assertEqual(kwargs.get('device'), -1)

if __name__ == '__main__':
    unittest.main()
