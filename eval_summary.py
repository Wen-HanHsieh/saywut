import json
import os
import math

def calculate_metrics(data):
    """
    Calculate success rate and false premise detection rates from JSON data.
    
    Args:
        data: A list of episode data dictionaries
        
    Returns:
        Dictionary containing calculated metrics, with NaN for missing FP types
    """
    # Calculate success rate
    total_episodes = len(data)
    successful_episodes = sum(1 for episode in data if episode.get('success') == 100.0)
    success_rate = (successful_episodes / total_episodes) * 100 if total_episodes > 0 else 0
    
    # Check if any episodes have the is_fp field at all
    has_fp_field = any('is_fp' in episode for episode in data)
    
    # Calculate FP detection rates for easy cases
    if has_fp_field:
        easy_fp_episodes = [episode for episode in data if episode.get('is_fp') == True and episode.get('is_fp_easy') == True]
        if easy_fp_episodes:
            easy_fp_detected = sum(1 for episode in easy_fp_episodes if episode.get('hard_false_premise_detacted') == 100.0)
            easy_fp_detection_rate = (easy_fp_detected / len(easy_fp_episodes)) * 100
        else:
            easy_fp_detection_rate = float('nan')  # No easy FP episodes exist
    else:
        easy_fp_detection_rate = float('nan')  # No FP field in the data
    
    # Calculate FP detection rates for hard cases
    if has_fp_field:
        hard_fp_episodes = [episode for episode in data if episode.get('is_fp') == True and episode.get('is_fp_easy') != True]
        if hard_fp_episodes:
            hard_fp_detected = sum(1 for episode in hard_fp_episodes if episode.get('hard_false_premise_detacted') == 100.0)
            hard_fp_detection_rate = (hard_fp_detected / len(hard_fp_episodes)) * 100
        else:
            hard_fp_detection_rate = float('nan')  # No hard FP episodes exist
    else:
        hard_fp_detection_rate = float('nan')  # No FP field in the data
    
    return {
        'success_rate': success_rate,
        'easy_fp_detection_rate': easy_fp_detection_rate,
        'hard_fp_detection_rate': hard_fp_detection_rate,
        'total_episodes': total_episodes,
        'has_fp_field': has_fp_field
    }

def calculate_metrics_from_file(file_path):
    """
    Load JSON data from a file and calculate metrics.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing calculated metrics
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return calculate_metrics(data)

def calculate_metrics_for_all_files(directory):
    """
    Calculate metrics for all JSON files in a directory.
    
    Args:
        directory: Path to the directory containing JSON files
        
    Returns:
        Dictionary mapping filenames to their metrics
    """
    results = {}
    for filename in os.listdir(directory):
        if filename.endswith('_eval.json'):
            file_path = os.path.join(directory, filename)
            try:
                results[filename] = calculate_metrics_from_file(file_path)
            except Exception as e:
                results[filename] = f"Error: {str(e)}"
    return results

# Example usage
if __name__ == "__main__":
    # Option 1: Process a specific JSON file
    # file_path = "open_drawer_FPmodel_FPdata_eval.json"
    # metrics = calculate_metrics_from_file(file_path)
    # print(f"Metrics for {file_path}:")
    # print(f"Success Rate: {metrics['success_rate']:.2f}%")
    # print(f"Easy FP Detection Rate: {metrics['easy_fp_detection_rate'] if not math.isnan(metrics['easy_fp_detection_rate']) else 'NaN (No easy FP episodes)'}")
    # print(f"Hard FP Detection Rate: {metrics['hard_fp_detection_rate'] if not math.isnan(metrics['hard_fp_detection_rate']) else 'NaN (No hard FP episodes)'}")
    
    # Option 2: Process all JSON files in the directory
    directory = "eval_data/result/."  # Current directory with all the JSON files
    results = calculate_metrics_for_all_files(directory)
    for filename, metrics in results.items():
        if isinstance(metrics, dict):
            print(f"Metrics for {filename}:")
            print(f"Success Rate: {metrics['success_rate']:.2f}%")
            print(f"Total Episodes: {metrics['total_episodes']}")
            
            if not metrics['has_fp_field']:
                print("Note: This file does not contain any false premise episodes")
                
            if math.isnan(metrics['easy_fp_detection_rate']):
                print("Easy FP Detection Rate: NaN (No easy FP episodes)")
            else:
                print(f"Easy FP Detection Rate: {metrics['easy_fp_detection_rate']:.2f}%")
                
            if math.isnan(metrics['hard_fp_detection_rate']):
                print("Hard FP Detection Rate: NaN (No hard FP episodes)")
            else:
                print(f"Hard FP Detection Rate: {metrics['hard_fp_detection_rate']:.2f}%")
            print()
        else:
            print(f"Error processing {filename}: {metrics}")
            print()