#!/usr/bin/env python3
"""
Dataset-Optimized GCOD Baselines
Handles .json.gz input with dataset-specific hyperparameters
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
import torch.nn.functional as F
import torch.nn as nn

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
        # Try different possible keys for nodes and edges
        nodes = item.get('nodes', [])
        edges = item.get('edges', [])

        # Alternative keys that might be used
        if not nodes and 'node_features' in item:
            nodes = item['node_features']
        if not edges and 'edge_index' in item:
            edges = item['edge_index']
        if not edges and 'adjacency' in item:
            edges = item['adjacency']

        # If nodes is still empty, try to infer from edges or use default
        num_nodes = len(nodes) if nodes else 0

        # If we have edges but no explicit nodes, infer node count
        if edges and num_nodes == 0:
            if isinstance(edges[0], list) and len(edges[0]) == 2:
                # Edges in format [[src, dst], ...]
                flat_edges = [node for edge in edges for node in edge]
                num_nodes = max(flat_edges) + 1 if flat_edges else 0
            elif len(edges) == 2 and isinstance(edges[0], list):
                # Edges in format [[src1, src2, ...], [dst1, dst2, ...]]
                all_nodes = edges[0] + edges[1]
                num_nodes = max(all_nodes) + 1 if all_nodes else 0

        # Skip only if we truly have no nodes AND no edges
        if num_nodes == 0 and not edges:
            print(f"Skipping empty graph at index {i}")
            continue

        # Create node features (using zeros as in original code)
        x = torch.zeros(max(1, num_nodes), dtype=torch.long)

        # Create edge index
        if edges:
            if isinstance(edges[0], list) and len(edges[0]) == 2:
                # Format: [[src, dst], [src, dst], ...]
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            elif len(edges) == 2 and isinstance(edges[0], list):
                # Format: [[src1, src2, ...], [dst1, dst2, ...]]
                edge_index = torch.tensor(edges, dtype=torch.long)
            else:
                # Fallback
                edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # Create data object
        data = Data(x=x, edge_index=edge_index)
        data.num_nodes = max(1, num_nodes)

        # Add edge attributes (required for model)
        if edge_index.shape[1] > 0:
            # Create dummy edge attributes if not provided
            data.edge_attr = torch.zeros(edge_index.shape[1], 7, dtype=torch.float)
        else:
            data.edge_attr = torch.empty(0, 7, dtype=torch.float)

        # Add label if available
        if has_labels:
            y_val = None
            if 'y' in item:
                y_data = item['y']
                # Handle nested list format [[4]] -> 4
                if isinstance(y_data, list) and len(y_data) > 0:
                    if isinstance(y_data[0], list):
                        y_val = y_data[0][0] if y_data[0] else 0
                    else:
                        y_val = y_data[0]
                else:
                    y_val = y_data
            elif 'label' in item:
                y_val = item['label']
            elif 'target' in item:
                y_val = item['target']

            # Validate and clamp label to valid range [0, 5]
            if y_val is not None:
                y_val = max(0, min(5, int(y_val)))
                data.y = torch.tensor(y_val, dtype=torch.long)
            else:
                data.y = torch.tensor(0, dtype=torch.long)

        # Store original index for GCOD if needed
        data.original_idx = i

        data_list.append(data)

    print(f"Loaded {len(data_list)} graphs from {len(json_data)} total items")
    return data_list


def add_zeros(data):
    """Ensure node features are zeros"""
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data


class GCODLoss(nn.Module):
    """Graph Centroid Outlier Discounting (GCOD) Loss Function"""

    def __init__(self, num_classes, alpha_train=0.01, lambda_r=0.1):
        super(GCODLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha_train = alpha_train
        self.lambda_r = lambda_r
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def _ensure_u_shape(self, u_params, batch_size, target_ndim):
        """Helper to ensure u_params has the correct shape for operations."""
        if u_params.shape[0] != batch_size:
            raise ValueError(
                f"u_params batch dimension {u_params.shape[0]} does not match expected batch_size {batch_size}")

        if target_ndim == 1:
            return u_params.squeeze() if u_params.ndim > 1 else u_params
        elif target_ndim == 2:
            return u_params.unsqueeze(1) if u_params.ndim == 1 else u_params
        return u_params

    def compute_L1(self, logits, targets, u_params):
        """Computes L1 = CE(f_θ(Z_B)) + α_train * u_B * (y_B ⋅ ỹ_B)"""
        batch_size = logits.size(0)
        if batch_size == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=logits.requires_grad)

        # Validate and clamp targets to [0, num_classes-1]
        targets = torch.clamp(targets, 0, self.num_classes - 1)

        y_onehot = F.one_hot(targets, num_classes=self.num_classes).float()
        y_soft = F.softmax(logits, dim=1)
        ce_loss_values = self.ce_loss(logits, targets)
        current_u_params = self._ensure_u_shape(u_params, batch_size, target_ndim=1)
        feedback_term_values = self.alpha_train * current_u_params * (y_onehot * y_soft).sum(dim=1)
        L1 = ce_loss_values + feedback_term_values
        return L1.mean()

    def compute_L2(self, logits, targets, u_params):
        """Computes L2 = (1/|C|) * ||ỹ_B + u_B * y_B - y_B||²_F + λ_r * ||u_B||²_2"""
        batch_size = logits.size(0)
        if batch_size == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=logits.requires_grad)

        # Validate and clamp targets to [0, num_classes-1]
        targets = torch.clamp(targets, 0, self.num_classes - 1)

        y_onehot = F.one_hot(targets, num_classes=self.num_classes).float()
        y_soft = F.softmax(logits, dim=1)
        current_u_params_unsqueezed = self._ensure_u_shape(u_params, batch_size, target_ndim=2)
        term = y_soft + current_u_params_unsqueezed * y_onehot - y_onehot
        L2_reconstruction = (1.0 / self.num_classes) * torch.norm(term, p='fro').pow(2)

        # Only add regularization if lambda_r > 0
        if self.lambda_r > 0:
            current_u_params_1d = self._ensure_u_shape(u_params, batch_size, target_ndim=1)
            u_reg = self.lambda_r * torch.norm(current_u_params_1d, p=2).pow(2)
            L2 = L2_reconstruction + u_reg
        else:
            L2 = L2_reconstruction
        return L2

    def compute_L3(self, logits, targets, u_params, l3_coeff):
        """Computes L3 = l3_coeff * D_KL(L || σ(-log(u_B)))"""
        batch_size = logits.size(0)
        if batch_size == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=logits.requires_grad)

        # Validate and clamp targets to [0, num_classes-1]
        targets = torch.clamp(targets, 0, self.num_classes - 1)

        y_onehot = F.one_hot(targets, num_classes=self.num_classes).float()
        diag_elements = (logits * y_onehot).sum(dim=1)
        L_log_probs = F.logsigmoid(diag_elements)
        current_u_params = self._ensure_u_shape(u_params, batch_size, target_ndim=1)
        target_probs_for_kl = torch.sigmoid(-torch.log(current_u_params + 1e-8))
        kl_div = F.kl_div(L_log_probs, target_probs_for_kl, reduction='mean', log_target=False)
        L3 = l3_coeff * kl_div
        return L3

    def forward(self, logits, targets, u_params, training_accuracy):
        """Calculates the GCOD loss components"""
        calculated_L1 = self.compute_L1(logits, targets, u_params)
        calculated_L2 = self.compute_L2(logits, targets, u_params)
        l3_coefficient = (1.0 - training_accuracy)
        calculated_L3 = self.compute_L3(logits, targets, u_params, l3_coefficient)
        total_loss_for_theta = calculated_L1 + calculated_L3
        return total_loss_for_theta, calculated_L1, calculated_L2, calculated_L3


def train(data_loader, model, optimizer, criterion, device, save_checkpoints, checkpoint_path, current_epoch,
          args_namespace, u_values_global, current_baseline_mode, scheduler=None):
    model.train()
    total_loss_accum = 0.0
    correct_preds = 0
    total_samples_processed = 0

    for data in tqdm(data_loader, desc="Training", unit="batch"):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)

        if current_baseline_mode == 4:  # GCOD specific logic
            batch_indices = data.original_idx.to(device=device, dtype=torch.long)

            if u_values_global.device != device:
                u_batch_cpu = u_values_global[batch_indices.cpu()].clone().detach()
                u_batch = u_batch_cpu.to(device).requires_grad_(True)
            else:
                u_batch = u_values_global[batch_indices].clone().detach().requires_grad_(True)

            output_for_u_optim = output.detach()

            for _ in range(args_namespace.gcod_T_u):
                if u_batch.grad is not None:
                    u_batch.grad.zero_()

                L2_for_u = criterion.compute_L2(output_for_u_optim, data.y, u_batch)
                L2_for_u.backward()
                with torch.no_grad():
                    u_batch.data -= args_namespace.gcod_lr_u * u_batch.grad.data
                    u_batch.data.clamp_(0, 1)

            u_batch_optimized = u_batch.detach()
            pred_for_acc = output.argmax(dim=1)
            if data.y.size(0) > 0:
                batch_accuracy = (pred_for_acc == data.y).sum().item() / data.y.size(0)
            else:
                batch_accuracy = 0.0

            loss_theta_components = criterion(output, data.y, u_batch_optimized, batch_accuracy)
            actual_loss_for_bp = loss_theta_components[0]

            with torch.no_grad():
                if u_values_global.device != device:
                    u_values_global[batch_indices.cpu()] = u_batch_optimized.cpu()
                else:
                    u_values_global[batch_indices] = u_batch_optimized
        else:
            actual_loss_for_bp = criterion(output, data.y)

        try:
            actual_loss_for_bp.backward()
            optimizer.step()
            if scheduler is not None and args_namespace.scheduler_type == 'OneCycleLR':
                scheduler.step()
        except IndexError as e:
            edge_max_val = data.edge_index.max().item() if data.edge_index.numel() > 0 else 'N/A'
            print(f"Error in batch with {data.num_nodes} nodes, edge_max={edge_max_val}")
            print(f"Batch info: x.shape={data.x.shape}, edge_index.shape={data.edge_index.shape}")
            raise e

        total_loss_accum += actual_loss_for_bp.item()
        pred_final = output.argmax(dim=1)
        correct_preds += (pred_final == data.y).sum().item()
        total_samples_processed += data.y.size(0)

    if save_checkpoints and (current_epoch + 1) % 10 == 0:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        checkpoint_file = f"{checkpoint_path}_{current_epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")

    avg_loss = total_loss_accum / len(data_loader) if len(data_loader) > 0 else 0.0
    accuracy = correct_preds / total_samples_processed if total_samples_processed > 0 else 0.0
    return avg_loss, accuracy


def evaluate(data_loader, model, criterion, device, calculate_accuracy=False, args_namespace=None,
             u_values_global_eval=None):
    model.eval()
    correct = 0
    total = 0
    predictions_list = []
    total_loss_val = 0

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluating", unit="batch"):
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)

            if calculate_accuracy:
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)

                if args_namespace and args_namespace.baseline_mode == 4:
                    u_eval_dummy = torch.zeros(data.y.size(0), device=device, dtype=torch.float)
                    loss_value = criterion.compute_L1(output, data.y, u_eval_dummy)
                else:
                    loss_value = criterion(output, data.y)
                total_loss_val += loss_value.item()
            else:
                predictions_list.extend(pred.cpu().numpy())

    if calculate_accuracy:
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss_val / len(data_loader) if len(data_loader) > 0 else 0.0
        return avg_loss, accuracy
    return predictions_list


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


def get_optimized_arguments(dataset_name):
    """Get dataset-specific optimized hyperparameters"""
    base_args = {
        'dataset': dataset_name,
        'train_mode': 1,
        'num_layer': 3,
        'emb_dim': 218,
        'drop_ratio': 0.7,
        'virtual_node': True,
        'residual': True,
        'JK': "last",
        'graph_pooling': "mean",
        'batch_norm': True,
        'layer_norm': False,
        'batch_size': 64,
        'epochs': 250,
        'baseline_mode': 4,  # GCOD
        'noise_prob': 0.2,
        'gce_q': 0.4,
        'initial_lr': 5e-3,
        'early_stopping': True,
        'patience': 25,
        'gcod_T_u': 15,
        'gcod_lr_u': 0.1,
        'use_scheduler': True,
        'scheduler_type': 'ReduceLROnPlateau',
        'step_size': 30,
        'gamma': 0.5,
        'patience_lr': 10,
        'factor': 0.5,
        'min_lr': 1e-7,
        'device': 0,
        'num_checkpoints': 10,
        'val_split': 0.2,
    }

    # Dataset-specific optimizations
    if dataset_name == 'C':
        base_args.update({
            'gcod_lambda_p': 5.0,
            'gcod_lambda_r': 0.5,
            'edge_drop_ratio': 0.3,
        })
    elif dataset_name == 'D':
        base_args.update({
            'gcod_lambda_p': 2.0,
            'gcod_lambda_r': 0.0,  # No regularization for D
            'edge_drop_ratio': 0.3,
        })
    else:
        # Default values for A, B
        base_args.update({
            'gcod_lambda_p': 2.0,
            'gcod_lambda_r': 0.1,
            'edge_drop_ratio': 0.1,
        })

    return argparse.Namespace(**base_args)


def run_optimized_gcod(dataset, train_path=None, test_path=None):
    """Main optimized GCOD training and testing function"""
    set_seed()

    args = get_optimized_arguments(dataset)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Dataset: {dataset}, Optimized GCOD")
    print(f"λ_p: {args.gcod_lambda_p}, λ_r: {args.gcod_lambda_r}, edge_drop: {args.edge_drop_ratio}")

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

        # Setup GCOD loss with dataset-specific parameters
        criterion = GCODLoss(
            num_classes=6,
            alpha_train=args.gcod_lambda_p,
            lambda_r=args.gcod_lambda_r
        )

        # Setup scheduler
        scheduler = None
        if args.use_scheduler and args.scheduler_type == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=args.factor,
                patience=args.patience_lr, min_lr=args.min_lr)

        # Initialize u_values for GCOD
        u_values_for_train = torch.zeros(len(train_dataset), device=device, requires_grad=False)
        print(f"Initialized u_values_for_train with size: {u_values_for_train.size()}")

        # Training loop with early stopping
        best_val_accuracy = 0.0
        patience_counter = 0
        os.makedirs('checkpoints', exist_ok=True)
        best_model_path = f'checkpoints/best_model_{dataset}_optimized.pth'

        train_losses, train_accuracies = [], []
        val_losses, val_accuracies = [], []

        for epoch in range(args.epochs):
            train_loss, train_acc = train(
                train_loader, model, optimizer, criterion, device,
                save_checkpoints=True, checkpoint_path=f"checkpoints/{dataset}_gcod_epoch", current_epoch=epoch,
                args_namespace=args, u_values_global=u_values_for_train,
                current_baseline_mode=args.baseline_mode, scheduler=scheduler)

            val_loss, val_acc = evaluate(
                val_loader, model, criterion, device, calculate_accuracy=True,
                args_namespace=args, u_values_global_eval=None)

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            # Early stopping logic
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"★ New best accuracy! {val_acc:.4f}")
            else:
                patience_counter += 1

            # Learning rate scheduling with logging
            if scheduler is not None:
                current_lr = optimizer.param_groups[0]['lr']
                scheduler.step(val_acc)
                new_lr = optimizer.param_groups[0]['lr']

                if new_lr != current_lr:
                    print(f"Learning rate reduced: {current_lr:.2e} → {new_lr:.2e}")

            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Acc={val_acc:.4f}, LR={current_lr:.2e}, Patience={patience_counter}")

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
        best_model_path = f'checkpoints/best_model_{dataset}_optimized.pth'

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

    criterion = GCODLoss(num_classes=6, alpha_train=args.gcod_lambda_p, lambda_r=args.gcod_lambda_r)
    predictions = evaluate(test_loader, model, criterion, device, calculate_accuracy=False,
                           args_namespace=args, u_values_global_eval=None)

    save_predictions(predictions, dataset)
    print(f"Predictions saved for dataset {dataset}")


if __name__ == "__main__":
    # This would be called from main.py
    pass