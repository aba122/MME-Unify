#!/usr/bin/env python3
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

def calculate_accuracy(predictions: List[Dict]) -> Tuple[float, Dict[str, float]]:
    """
    Calculate overall accuracy and accuracy for each category
    
    Args:
        predictions: List containing prediction results
    
    Returns:
        Overall accuracy and dictionary of accuracy per category
    """
    total_correct = 0
    total_count = 0
    
    # Store statistics for each category
    category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for pred in predictions:
        category = pred['category']
        
        # If output is empty, count as incorrect instead of skipping
        if not pred.get('output'):
            is_correct = False
        else:
            is_correct = pred['output'] == pred['answer']
        
        # Update overall statistics
        total_correct += int(is_correct)
        total_count += 1
        
        # Update category statistics
        category_stats[category]['correct'] += int(is_correct)
        category_stats[category]['total'] += 1
    
    # Calculate overall accuracy
    overall_accuracy = total_correct / total_count if total_count > 0 else 0
    
    # Calculate accuracy for each category
    category_accuracy = {}
    for category, stats in category_stats.items():
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        category_accuracy[category] = accuracy
    
    return overall_accuracy, category_accuracy

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Evaluate model prediction results')
    parser.add_argument('input_file', help='Path to input JSON file')
    args = parser.parse_args()
    
    # Read JSON file
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found")
        return
    except json.JSONDecodeError:
        print(f"Error: '{args.input_file}' is not a valid JSON file")
        return
    
    # Calculate accuracy
    overall_acc, category_acc = calculate_accuracy(predictions)
    
    # Print results
    print("\n=== Evaluation Results ===")
    print(f"\nOverall Accuracy: {overall_acc:.2%}")
    
    print("\nAccuracy by Category:")
    for category, acc in sorted(category_acc.items()):
        print(f"{category}: {acc:.2%}")

if __name__ == "__main__":
    main()