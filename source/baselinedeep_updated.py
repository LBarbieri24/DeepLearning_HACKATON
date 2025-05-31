#!/usr/bin/env python3
"""
Updated BaselineDeep - Graph Neural Network Training Pipeline
Handles .json.gz input and proper train/validation splitting
"""

import os
import gc
import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt
import logging
import json
import gzip
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data import random_split
import argparse
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Add source directory to path
sys.path.insert(0, 'source')

try:
    from source.preprocessor import MultiDatasetLoader
    from source.utils import set_seed
    from source.models_EDandBatch_norm import GNN

    print("Successfully imported modules.")
except ImportError as e:
    print(f"ERROR importing module: {e}")
    sys.exit(1)


def load_json_gz(file_path):
    """Load data from .json.gz or .json file"""
    if file_path.endswith('.gz'):
        with gzip.open(file_path, 'rt') as f:
            return json.load(f)
    else:
        with open(file_path, 'r') as f:
            return json.load(f)


def json_to_torch_geometric(json_data, has_labels=True):
    """Convert JSON data to PyTorch Geometric Data objects"""
    data_list = []

    for i, item in enumerate(json_data):
        # Handle the actual JSON structure: edge_index, edge_attr, num_nodes, y
        if 'edge_index' in item and 'num_nodes' in item:
            # Use provided structure
            edge_index = torch.tensor(item['edge_index'], dtype=torch.long)
            num_nodes = item['num_nodes']

            # Handle edge attributes
            edge_attr = None
            if 'edge_attr' in item and item['edge_attr'] is not None:
                edge_attr = torch.tensor(item['edge_attr'], dtype=torch.float)

            # Create node features (zeros as expected by the model)
            x = torch.zeros(num_nodes, dtype=torch.long)

        else:
            # Fallback to old parsing logic for different formats
            nodes = item.get('nodes', [])
            edges = item.get('edges', [])

            num_nodes = len(nodes) if nodes else 0

            if edges and num_nodes == 0:
                if isinstance(edges[0], list) and len(edges[0]) == 2:
                    flat_edges = [node for edge in edges for node in edge]
                    num_nodes = max(flat_edges) + 1 if flat_edges else 0
                elif len(edges) == 2 and isinstance(edges[0], list):
                    all_nodes = edges[0] + edges[1]
                    num_nodes = max(all_nodes) + 1 if all_nodes else 0

            if num_nodes == 0 and not edges:
                continue

            x = torch.zeros(max(1, num_nodes), dtype=torch.long)

            if edges:
                if isinstance(edges[0], list) and len(edges[0]) == 2:
                    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                elif len(edges) == 2 and isinstance(edges[0], list):
                    edge_index = torch.tensor(edges, dtype=torch.long)
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)

            edge_attr = None

        # Create data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.num_nodes = num_nodes

        # Add label if available
        if has_labels:
            if 'y' in item:
                y_data = item['y']
                # Handle nested list format [[4]] -> 4
                if isinstance(y_data, list) and len(y_data) > 0:
                    if isinstance(y_data[0], list):
                        y_data = y_data[0][0] if y_data[0] else 0
                    else:
                        y_data = y_data[0]
                data.y = torch.tensor(y_data, dtype=torch.long)
            elif 'label' in item:
                data.y = torch.tensor(item['label'], dtype=torch.long)
            elif 'target' in item:
                data.y = torch.tensor(item['target'], dtype=torch.long)

        # Store original index for GCOD if needed
        data.original_idx = i

        data_list.append(data)

    print(f"Loaded {len(data_list)} graphs from {len(json_data)} total items")
    return data_list


def add_zeros(data):
    """Ensure node features are zeros"""
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data


def train(data_loader, model, optimizer, criterion, device, save_checkpoints, checkpoint_path, current_epoch,
          scheduler=None, args=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, data in enumerate(tqdm(data_loader, desc="Training", unit="batch")):
        data = data.to(device)
        optimizer.zero_grad()

        try:
            output = model(data)
        except IndexError as e:
            edge_max_val = data.edge_index.max().item() if data.edge_index.numel() > 0 else 'N/A'
            print(f"Error in batch with {data.num_nodes} nodes, edge_max={edge_max_val}")
            print(f"Batch info: x.shape={data.x.shape}, edge_index.shape={data.edge_index.shape}")
            raise e

        loss = criterion(output, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None and args.scheduler_type == 'OneCycleLR':
            scheduler.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.y.size(0)

    if save_checkpoints and (current_epoch + 1) % 10 == 0:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        checkpoint_file = f"{checkpoint_path}_{current_epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")

    return total_loss / len(data_loader), correct / total


def evaluate(data_loader, model, criterion, device, calculate_accuracy=False):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    total_loss = 0

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluating", unit="batch"):
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())

            if calculate_accuracy:
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
                total_loss += criterion(output, data.y).item()
                true_labels.extend(data.y.cpu().numpy())

    if calculate_accuracy:
        accuracy = correct / total
        f1_macro = f1_score(true_labels, predictions, average='macro')
        return total_loss / len(data_loader), accuracy, f1_macro
    return predictions


def save_predictions(predictions, dataset_name):
    """Save predictions to CSV file"""
    os.makedirs('submission', exist_ok=True)
    output_csv_path = f"submission/testset_{dataset_name}.csv"

    test_graph_ids = list(range(len(predictions)))
    output_df = pd.DataFrame({
        "id": test_graph_ids,
        "pred": predictions
    })

    output_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")


def plot_training_progress(train_losses, train_accuracies, val_losses, val_accuracies, output_dir):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss", color='blue', marker='o')
    plt.plot(epochs, val_losses, label="Validation Loss", color='red', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Training Accuracy", color='green', marker='o')
    plt.plot(epochs, val_accuracies, label="Validation Accuracy", color='orange', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy per Epoch')
    plt.legend()
    plt.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_progress.png"))
    plt.show()
    plt.close()


def get_arguments(dataset_name):
    """Set training configuration directly"""
    args = {
        'dataset': dataset_name,
        'train_mode': 1,
        'num_layer': 2,
        'emb_dim': 128,
        'drop_ratio': 0.3,
        'virtual_node': True,
        'residual': True,
        'JK': "last",
        'edge_drop_ratio': 0.15,
        'batch_norm': True,
        'graph_pooling': "mean",
        'batch_size': 64,
        'epochs': 200,
        'baseline_mode': 3,  # 1=CE, 2=Noisy CE, 3=GCE
        'noise_prob': 0.2,
        'gce_q': 0.9,
        'initial_lr': 5e-3,
        'best_model_criteria': 'f1',
        'use_scheduler': True,
        'scheduler_type': 'ReduceLROnPlateau',
        'step_size': 30,
        'gamma': 0.5,
        'patience_lr': 12,
        'factor': 0.5,
        'min_lr': 1e-7,
        'T_max': 50,
        'eta_min': 1e-6,
        'gamma_exp': 0.95,
        'max_lr': 1e-3,
        'pct_start': 0.3,
        'early_stopping': True,
        'patience': 25,
        'device': 0,
        'num_checkpoints': 1,
        'val_split': 0.2,  # Validation split ratio
    }

    # Dataset-specific optimizations
    if dataset_name == 'B':
        args.update({
            'num_layer': 3,
            'gce_q': 0.9,
            'emb_dim': 128,
        })

    if dataset_name == 'C':
        args.update({
            'num_layer': 3,
            'gce_q': 0.9,
            'emb_dim': 256,
            'edge_drop_ratio' : 0.1,
            'drop_ratio': 0.4,
        })

    if dataset_name == 'D':
        args.update({
            'num_layer': 3,
            'gce_q': 0.9,
            'emb_dim': 256,
            'edge_drop_ratio' : 0.15,
            'drop_ratio': 0.7,
        })

    return argparse.Namespace(**args)


class NoisyCrossEntropyLoss(torch.nn.Module):
    def __init__(self, p_noisy):
        super().__init__()
        self.p = p_noisy
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        losses = self.ce(logits, targets)
        weights = (1 - self.p) + self.p * (
                1 - torch.nn.functional.one_hot(targets, num_classes=logits.size(1)).float().sum(dim=1))
        return (losses * weights).mean()


class GeneralizedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, q=0.7):
        super(GeneralizedCrossEntropyLoss, self).__init__()
        if not (0 < q <= 1):
            raise ValueError("q should be in (0, 1]")
        self.q = q

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        target_probs = probs[torch.arange(targets.size(0)), targets]
        target_probs = target_probs.clamp(min=1e-7, max=1 - 1e-7)
        loss = (1 - (target_probs ** self.q)) / self.q
        return loss.mean()


def run_baseline_deep(dataset, train_path=None, test_path=None, baseline_choice='gce'):
    """Main training and testing function"""
    set_seed()

    args = get_arguments(dataset)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Dataset: {dataset}, Baseline: {baseline_choice}")

    if train_path:
        # Load and split training data
        print(f"Loading training data from {train_path}")
        train_json = load_json_gz(train_path)
        all_train_data = json_to_torch_geometric(train_json, has_labels=True)

        # Split into train and validation
        total_size = len(all_train_data)
        if total_size == 0:
            raise ValueError("No valid training samples found after filtering")

        train_size = max(1, int((1 - args.val_split) * total_size))
        val_size = max(1, total_size - train_size)

        # Ensure we don't exceed total size
        if train_size + val_size > total_size:
            train_size = total_size - 1
            val_size = 1

        train_dataset, val_dataset = random_split(all_train_data, [train_size, val_size])

        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        # Initialize model
        model = GNN(num_class=6,
                    num_layer=args.num_layer,
                    emb_dim=args.emb_dim,
                    drop_ratio=args.drop_ratio,
                    virtual_node=args.virtual_node,
                    residual=args.residual,
                    JK=args.JK,
                    graph_pooling=args.graph_pooling,
                    edge_drop_ratio=args.edge_drop_ratio,
                    batch_norm=args.batch_norm)

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr)

        # Setup loss function based on baseline choice
        baseline_map = {'ce': 1, 'noisy': 2, 'gce': 3}
        args.baseline_mode = baseline_map.get(baseline_choice, 3)

        if args.baseline_mode == 2:
            criterion = NoisyCrossEntropyLoss(args.noise_prob)
            print(f"Using Noisy Cross Entropy Loss (p={args.noise_prob})")
        elif args.baseline_mode == 3:
            criterion = GeneralizedCrossEntropyLoss(q=args.gce_q)
            print(f"Using Generalized Cross Entropy Loss (q={args.gce_q})")
        else:
            criterion = torch.nn.CrossEntropyLoss()
            print("Using standard Cross Entropy Loss")

        # Setup scheduler
        scheduler = None
        if args.use_scheduler and args.scheduler_type == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=args.factor,
                patience=args.patience_lr, min_lr=args.min_lr)

        # Training loop with early stopping
        best_val_metric = 0.0
        patience_counter = 0
        os.makedirs('checkpoints', exist_ok=True)
        best_model_path = f'checkpoints/best_model_{dataset}.pth'

        train_losses, train_accuracies = [], []
        val_losses, val_accuracies = [], []

        for epoch in range(args.epochs):
            train_loss, train_acc = train(
                train_loader, model, optimizer, criterion, device,
                save_checkpoints=True, checkpoint_path=f"checkpoints/{dataset}_epoch", current_epoch=epoch,
                scheduler=scheduler, args=args)

            val_loss, val_acc, val_f1 = evaluate(
                val_loader, model, criterion, device, calculate_accuracy=True)

            current_metric = val_f1 if args.best_model_criteria == 'f1' else val_acc

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            # Early stopping logic
            if current_metric > best_val_metric:
                best_val_metric = current_metric
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"★ New best {args.best_model_criteria.upper()}! {current_metric:.4f}")
            else:
                patience_counter += 1

            # Learning rate scheduling with logging
            if scheduler is not None:
                current_lr = optimizer.param_groups[0]['lr']
                scheduler.step(val_loss)
                new_lr = optimizer.param_groups[0]['lr']

                if new_lr != current_lr:
                    print(f"Learning rate reduced: {current_lr:.2e} → {new_lr:.2e}")

            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}, LR={current_lr:.2e}, Patience={patience_counter}")

            # Early stopping check
            if args.early_stopping and patience_counter >= args.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # Plot training progress
        os.makedirs('logs', exist_ok=True)
        plot_training_progress(train_losses, train_accuracies, val_losses, val_accuracies, 'logs')

    else:
        # Testing only mode
        model = GNN(num_class=6,
                    num_layer=args.num_layer,
                    emb_dim=args.emb_dim,
                    drop_ratio=args.drop_ratio,
                    virtual_node=args.virtual_node,
                    residual=args.residual,
                    JK=args.JK,
                    graph_pooling=args.graph_pooling,
                    edge_drop_ratio=args.edge_drop_ratio,
                    batch_norm=args.batch_norm)

        model = model.to(device)
        best_model_path = f'checkpoints/best_model_{dataset}.pth'

        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
            print(f"Loaded model from {best_model_path}")
        else:
            raise FileNotFoundError(f"No pre-trained model found at {best_model_path}")

    # Test phase
    print(f"Loading test data from {test_path}")
    test_json = load_json_gz(test_path)
    test_dataset = json_to_torch_geometric(test_json, has_labels=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    criterion = torch.nn.CrossEntropyLoss()  # For evaluation only
    predictions = evaluate(test_loader, model, criterion, device, calculate_accuracy=False)

    save_predictions(predictions, dataset)
    print(f"Predictions saved for dataset {dataset}")


if __name__ == "__main__":
    # This would be called from main.py
    pass