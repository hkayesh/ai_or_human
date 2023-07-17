import unittest
import tempfile
import shutil
import os
from unittest.mock import patch, MagicMock

import torch
import numpy as np
from transformers import AutoModelForSequenceClassification

from code.model import _get_train_data_loader, _get_test_data_loader, train, flat_accuracy, test

class ModelTests(unittest.TestCase):
    def setUp(self):
        self.batch_size = 16
        self.test_batch_size = 8
        self.num_labels = 2
        self.data_dir = "data/train"
        self.test_dir = "data/test"
        self.epochs = 2
        self.log_interval = 50

        # Create temporary directories for data
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        # Remove temporary directories and files
        shutil.rmtree(self.data_dir)
        shutil.rmtree(self.test_dir)

    def test_get_train_data_loader(self):
        # Create a temporary train.csv file
        train_data = [
            {"text": "This is sentence 1", "label": 0},
            {"text": "This is sentence 2", "label": 1},
            {"text": "This is sentence 3", "label": 0},
        ]
        train_csv_path = os.path.join(self.data_dir, "train.csv")
        with open(train_csv_path, "w") as f:
            f.write("text,label\n")
            for item in train_data:
                f.write(f"{item['text']},{item['label']}\n")

        # Test _get_train_data_loader function
        train_dataloader = _get_train_data_loader(self.batch_size, self.data_dir)

        # Assertions
        self.assertEqual(len(train_dataloader.dataset), len(train_data))
        self.assertEqual(train_dataloader.batch_size, self.batch_size)

    def test_get_test_data_loader(self):
        # Create a temporary test.csv file
        test_data = [
            {"text": "This is sentence 1", "label": 1},
            {"text": "This is sentence 2", "label": 0},
        ]
        test_csv_path = os.path.join(self.test_dir, "test.csv")
        with open(test_csv_path, "w") as f:
            f.write("text,label\n")
            for item in test_data:
                f.write(f"{item['text']},{item['label']}\n")

        # Test _get_test_data_loader function
        test_dataloader = _get_test_data_loader(self.test_batch_size, self.test_dir)

        # Assertions
        self.assertEqual(len(test_dataloader.dataset), len(test_data))
        self.assertEqual(test_dataloader.batch_size, self.test_batch_size)

    @patch("torch.cuda.is_available")
    @patch("torch.manual_seed")
    @patch("torch.cuda.manual_seed")
    def test_train(self, mock_cuda_available, mock_manual_seed, mock_cuda_manual_seed):
        mock_cuda_available.return_value = False
        mock_manual_seed.return_value = None
        mock_cuda_manual_seed.return_value = None

        # Create a temporary train.csv file
        train_data = [
            {"text": "This is sentence 1", "label": 0},
            {"text": "This is sentence 2", "label": 1},
            {"text": "This is sentence 3", "label": 0},
        ]
        train_csv_path = os.path.join(self.data_dir, "train.csv")
        with open(train_csv_path, "w") as f:
            f.write("text,label\n")
            for item in train_data:
                f.write(f"{item['text']},{item['label']}\n")

        # Create a temporary test.csv file
        test_data = [
            {"text": "This is sentence 4", "label": 1},
            {"text": "This is sentence 5", "label": 0},
        ]
        test_csv_path = os.path.join(self.test_dir, "test.csv")
        with open(test_csv_path, "w") as f:
            f.write("text,label\n")
            for item in test_data:
                f.write(f"{item['text']},{item['label']}\n")

        # Create a temporary directory for the model
        temp_model_dir = tempfile.mkdtemp()

        args = MagicMock()
        args.num_gpus = 0
        args.seed = 123
        args.batch_size = self.batch_size
        args.test_batch_size = self.test_batch_size
        args.num_labels = self.num_labels
        args.epochs = self.epochs
        args.log_interval = self.log_interval
        args.data_dir = self.data_dir
        args.test = self.test_dir
        args.model_dir = temp_model_dir

        # Mock the model and optimizer
        mock_model = MagicMock(spec=AutoModelForSequenceClassification)
        mock_optimizer = MagicMock()
        mock_model.parameters.return_value = []
        mock_optimizer.param_groups.return_value = []
        train_args = {
            "token_type_ids": None,
            "attention_mask": MagicMock(),
            "labels": MagicMock(),
        }
        mock_model.return_value = (MagicMock(),) + (train_args,)
        torch.nn.utils.clip_grad_norm_.side_effect = lambda *args, **kwargs: None

        # Test train function
        with patch("model.AutoModelForSequenceClassification.from_pretrained", return_value=mock_model):
            train(args)

        # Assertions
        self.assertEqual(mock_model.call_count, self.epochs)
        self.assertEqual(mock_model.save_pretrained.call_count, 1)

    def test_flat_accuracy(self):
        # Test flat_accuracy function
        preds = np.array([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
        labels = np.array([1, 0, 0])
        accuracy = flat_accuracy(preds, labels)

        # Assertions
        self.assertAlmostEqual(accuracy, 2 / 3, places=6)

    def test_test(self):
        # Create a temporary test.csv file
        test_data = [
            {"text": "This is sentence 4", "label": 1},
            {"text": "This is sentence 5", "label": 0},
        ]
        test_csv_path = os.path.join(self.test_dir, "test.csv")
        with open(test_csv_path, "w") as f:
            f.write("text,label\n")
            for item in test_data:
                f.write(f"{item['text']},{item['label']}\n")

        # Mock the model and device
        mock_model = MagicMock()
        mock_device = MagicMock()

        # Mock the DataLoader and Batch instances
        mock_dataloader = MagicMock()
        mock_batch = MagicMock()
        mock_dataloader.__iter__.return_value = [mock_batch]

        # Mock the model evaluation
        with patch("test", wraps=test) as mock_test:
            test(mock_model, mock_dataloader, mock_device)

            # Assertions
            mock_model.eval.assert_called_once()
            mock_test.assert_called_once_with(mock_model, mock_dataloader, mock_device)

