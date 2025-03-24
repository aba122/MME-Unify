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
    
    category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for pred in predictions:
        category = pred['category']
        
        if not pred.get('output'):
            is_correct = False
        else:
            is_correct = pred['output'] == pred['answer']
        
        total_correct += int(is_correct)
        total_count += 1
        
        category_stats[category]['correct'] += int(is_correct)
        category_stats[category]['total'] += 1
    
    overall_accuracy = total_correct / total_count if total_count > 0 else 0
    
    category_accuracy = {}
    for category, stats in category_stats.items():
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        category_accuracy[category] = accuracy
    
    return overall_accuracy, category_accuracy

def main():
    parser = argparse.ArgumentParser(description='Evaluate model prediction results')
    parser.add_argument('input_file', help='Path to input JSON file')
    args = parser.parse_args()
    
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found")
        return
    except json.JSONDecodeError:
        print(f"Error: '{args.input_file}' is not a valid JSON file")
        return
    
    overall_acc, category_acc = calculate_accuracy(predictions)
    
    print("\n=== Evaluation Results ===")
    print(f"\nOverall Accuracy: {overall_acc:.2%}")
    
    print("\nAccuracy by Category:")
    for category, acc in sorted(category_acc.items()):
        print(f"{category}: {acc:.2%}")

if __name__ == "__main__":
    main()