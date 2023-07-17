import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import AdamW, AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

MAX_LEN = 64

def _get_train_data_loader(batch_size, training_dir):
    """
    Get the train data loader.
    Args:
        batch_size (int): Batch size for training.
        training_dir (str): Directory path of the training data.
    Returns:
        train_dataloader (DataLoader): Train data loader.
    """
    logger.info("Get train data loader")

    dataset = pd.read_csv(os.path.join(training_dir, "train.csv"))
    texts = dataset.text.values
    labels = dataset.label.values

    tokenizer.pad_token = tokenizer.eos_token
    input_ids = []
    for sent in texts:
        encoded_sent = tokenizer.encode(sent, add_special_tokens=True)
        input_ids.append(encoded_sent)

    # Pad shorter sentences
    input_ids_padded = []
    for i in input_ids:
        while len(i) < MAX_LEN:
            i.append(0)
        input_ids_padded.append(i)
    input_ids = input_ids_padded

    # Mask; 0: added, 1: otherwise
    attention_masks = []
    # For each sentence...
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)

    # Convert to PyTorch data types.
    train_inputs = torch.tensor(input_ids)
    train_labels = torch.tensor(labels)
    train_masks = torch.tensor(attention_masks)

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    return train_dataloader


def _get_test_data_loader(test_batch_size, training_dir):
    """
    Get the test data loader.
    Args:
        test_batch_size (int): Batch size for testing.
        training_dir (str): Directory path of the testing data.
    Returns:
        test_dataloader (DataLoader): Test data loader.
    """
    dataset = pd.read_csv(os.path.join(training_dir, "test.csv"))
    texts = dataset.text.values
    labels = dataset.label.values

    input_ids = []
    for sent in texts:
        encoded_sent = tokenizer.encode(sent, add_special_tokens=True)
        input_ids.append(encoded_sent)

    # Pad shorter sentences
    input_ids_padded = []
    for i in input_ids:
        while len(i) < MAX_LEN:
            i.append(0)
        input_ids_padded.append(i)
    input_ids = input_ids_padded

    # Mask; 0: added, 1: otherwise
    attention_masks = []
    # For each sentence...
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)

    # Convert to PyTorch data types.
    train_inputs = torch.tensor(input_ids)
    train_labels = torch.tensor(labels)
    train_masks = torch.tensor(attention_masks)

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=test_batch_size)

    return train_dataloader


def train(args):
    """
    Train the model.
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    use_cuda = args.num_gpus > 0
    logger.debug("Number of GPUs available - %d", args.num_gpus)
    device = torch.device("cuda" if use_cuda else "cpu")

    # Set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)
    test_loader = _get_test_data_loader(args.test_batch_size, args.test)

    model = AutoModelForSequenceClassification.from_pretrained(
        "gpt2",
        num_labels=args.num_labels,
        output_attentions=False,
        output_hidden_states=False,
    )

    model.config.pad_token_id = model.config.eos_token_id
    model = model.to(device)

    model = torch.nn.DataParallel(model)
    optimizer = AdamW(
        model.parameters(),
        lr=1e-5,
        eps=1e-8,
    )

    logger.info("Starting the training...\n")
    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        model.train()
        for step, batch in enumerate(train_loader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]

            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % args.log_interval == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        step * len(batch[0]),
                        len(train_loader.sampler),
                        100.0 * step / len(train_loader),
                        loss.item(),
                    )
                )

        logger.info("Average training loss: %f\n", total_loss / len(train_loader))

        test(model, test_loader, device)

    logger.info("Saving the trained model...")
    trained_model = model.module if hasattr(model, "module") else model
    trained_model.save_pretrained(save_directory=args.model_dir)


def flat_accuracy(preds, labels):
    """
    Calculate the accuracy of predictions.
    Args:
        preds (np.ndarray): Predicted labels.
        labels (np.ndarray): True labels.
    Returns:
        accuracy (float): Accuracy of predictions.
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def test(model, test_loader, device):
    """
    Perform testing and calculate accuracy.
    Args:
        model: Model for testing.
        test_loader (DataLoader): Test data loader.
        device: Device (cuda or cpu) for testing.
    """
    model.eval()
    _, eval_accuracy = 0, 0

    with torch.no_grad():
        for batch in test_loader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0]
            logits = logits.detach()
