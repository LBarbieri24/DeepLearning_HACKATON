#!/usr/bin/env python3
"""
Deep Learning Exam Hackathon - Main Script
Handles training and testing of graph neural networks with noisy labels
"""

import argparse
import os
import re


def extract_dataset_from_path(test_path):
    """Extract dataset letter (A, B, C, D) from test path"""
    # Look for pattern like /datasets/A/, /A/, A_test, etc.
    patterns = [
        r'/([ABCD])/',  # /datasets/A/
        r'([ABCD])_test',  # A_test.json.gz
        r'/([ABCD])_',  # /A_test.json.gz
        r'_([ABCD])\.',  # test_A.json.gz
    ]

    for pattern in patterns:
        match = re.search(pattern, test_path)
        if match:
            return match.group(1)

    # Fallback: check if path contains A, B, C, or D
    for dataset in ['A', 'B', 'C', 'D']:
        if dataset in test_path.upper():
            return dataset

    raise ValueError(f"Could not extract dataset from path: {test_path}")


def get_train_path_from_test(test_path, train_path=None):
    """Generate train path from test path if not provided"""
    if train_path:
        return train_path

    # Replace 'test' with 'train' in the path
    train_path = test_path.replace('test', 'train')
    return train_path


def run_baseline_deep(dataset, train_path=None, test_path=None, baseline_choice='gce'):
    """Run the standard baseline (CE, Noisy CE, GCE)"""
    from source.baselinedeep_updated import run_baseline_deep as baseline_func
    baseline_func(dataset, train_path, test_path, baseline_choice)


# def run_baseline_gcod(dataset, train_path=None, test_path=None):
#     """Run the GCOD baseline with dataset-specific optimizations"""
#     from source.gcod_optimized_updated import run_optimized_gcod as gcod_func
#     gcod_func(dataset, train_path, test_path)


def main():
    parser = argparse.ArgumentParser(description='Deep Learning Exam Hackathon - Graph Classification')
    parser.add_argument('--test_path', required=True, help='Path to test.json.gz file')
    parser.add_argument('--train_path', help='Path to train.json.gz file (optional)')
    parser.add_argument('--baseline', choices=['ce', 'noisy', 'gce'], # , 'gcod'
                        default='gce', help='Baseline choice (default: gce)')

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.test_path):
        raise FileNotFoundError(f"Test path not found: {args.test_path}")

    if args.train_path and not os.path.exists(args.train_path):
        raise FileNotFoundError(f"Train path not found: {args.train_path}")

    # Extract dataset from test path
    dataset = extract_dataset_from_path(args.test_path)
    print(f"Detected dataset: {dataset}")

    # Determine train path if not provided
    if not args.train_path:
        print("No train path provided - running in test-only mode")
        train_path = None
    else:
        train_path = args.train_path
        print(f"Training with: {train_path}")

    print(f"Testing with: {args.test_path}")
    print(f"Using model: {args.baseline}")

    # Run appropriate baseline
    if args.baseline == 'gcod':
        # run_baseline_gcod(dataset, train_path, args.test_path)
        print("no longer available gcod")
    else:
        run_baseline_deep(dataset, train_path, args.test_path, args.baseline)

    print("Execution completed!")


if __name__ == "__main__":
    main()