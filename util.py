import json
import argparse
from collections import defaultdict
# TASKS: close_jar       push_buttons       slide_block_to_color_target
# meat_off_grill  put_money_in_safe  sweep_to_dustpan_of_size
# open_drawer     reach_and_drag     turn_tap

def restructure_json(task):
    task = 'turn_tap'
    input_path = f'/home/wenhan/Projects/saywut/sim/data/anns_eval/{task}/no_trace/eval.json'
    # output_path = f'/home/wenhan/Projects/saywut/sim/data/anns_eval/{task}/no_trace/eval.json'
    output_path = f'eval_data/{task}_eval.json'

    # Load the original JSON
    with open(input_path, 'r') as fin:
        data = json.load(fin)

    # Write it back with 2‑space indentation
    with open(output_path, 'w') as fout:
        json.dump(data, fout, indent=2)

    print(f"Re‑indented JSON written to {output_path}")

def json_fp_stats(json_file):
    try:
        # Load the JSON data
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Initialize counters
        easy_false_premises = defaultdict(int)
        hard_false_premises = defaultdict(int)
        
        # Analyze each entry
        for entry in data:
            # Skip entries without false premise tags
            if 'is_false_premise' not in entry:
                continue
            
            episode = entry.get('episode')
            
            # Count false premises
            if entry.get('is_false_premise', False):
                if entry.get('is_easy_false_premise', False):
                    easy_false_premises[episode] += 1
                else:
                    # This is a hard false premise
                    hard_false_premises[episode] += 1
        
        # Generate summary
        print("Summary of False Premises by Episode:\n")
        print("Easy False Premises:")
        for episode, count in sorted(easy_false_premises.items()):
            print(f"  Episode {episode}: {count} easy false premises")
        
        print("\nHard False Premises:")
        for episode, count in sorted(hard_false_premises.items()):
            print(f"  Episode {episode}: {count} hard false premises")
        
        print("\nTotal Easy False Premises:", sum(easy_false_premises.values()))
        print("Total Hard False Premises:", sum(hard_false_premises.values()))
        
    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file '{json_file}'")
    except Exception as e:
        print(f"Error: {str(e)}")

import numpy as np
import random

def bootstrap_standard_error(num_ones, total_eps=25, subsample_size=20, num_iterations=500):
    """
    Perform bootstrap sampling to compute standard error of the mean.
    
    Parameters:
    - num_ones: number of episodes that score 1
    - total_eps: total number of episodes (default 25)
    - subsample_size: size of each subsample (default 20)
    - num_iterations: number of bootstrap iterations (default 500)
    
    Returns:
    - standard_error: standard error of the mean (std dev of bootstrap means)
    - means: array of all 500 means
    """
    
    # Create the original dataset: num_ones episodes with value 1, rest with value 0
    original_data = [100] * num_ones + [0] * (total_eps - num_ones)
    
    # Store means from each iteration
    means = []
    
    # Perform bootstrap sampling
    for i in range(num_iterations):
        # Sample with replacement
        subsample = random.choices(original_data, k=subsample_size)
        
        # Compute mean of this subsample
        mean_subsample = np.mean(subsample)
        means.append(mean_subsample)
    
    # Compute standard error (standard deviation of the means)
    standard_error = np.std(means, ddof=1)
    
    return standard_error, np.array(means)

def multi_task_bootstrap_analysis(num_ones_list, total_eps=25, subsample_size=20, num_iterations=500):
    """
    Perform bootstrap analysis across multiple tasks and compute average standard error.
    
    Parameters:
    - num_ones_list: list of number of episodes that score 1 for each task
    - total_eps: total number of episodes per task (default 25)
    - subsample_size: size of each subsample (default 20)
    - num_iterations: number of bootstrap iterations (default 500)
    
    Returns:
    - average_standard_error: average standard error across all tasks
    - task_standard_errors: list of standard errors for each task
    - task_results: detailed results for each task
    """
    
    if len(num_ones_list) != 9:
        raise ValueError("Input list must contain exactly 9 values (one for each task)")
    
    task_standard_errors = []
    task_results = []
    
    print("Processing each task:")
    print("-" * 50)
    
    for i, num_ones in enumerate(num_ones_list):
        if num_ones > total_eps:
            raise ValueError(f"Task {i+1}: num_ones ({num_ones}) cannot exceed total_eps ({total_eps})")
        
        standard_error, means = bootstrap_standard_error(num_ones, total_eps, subsample_size, num_iterations)
        
        task_standard_errors.append(standard_error)
        task_results.append({
            'task': i+1,
            'num_ones': num_ones,
            'standard_error': standard_error,
            'mean_of_means': np.mean(means),
            'variance_of_means': np.var(means, ddof=1)
        })
        
        print(f"Task {i+1}: {num_ones}/25 episodes scored 1, standard error = {standard_error:.6f}")
    
    # Compute average standard error across all tasks
    average_standard_error = np.mean(task_standard_errors)
    
    return average_standard_error, task_standard_errors, task_results

# Example usage
if __name__ == "__main__":
    # Input: List of number of episodes that scored 1 for each of the 9 tasks
    # Example values - replace with your actual data
    num_ones_per_task_og = [16, 32, 36, 28, 60, 92, 88, 32, 0] # IVA
    # num_ones_per_task_og = [4, 40, 32, 40, 44, 88, 60, 40, 0] # LLaRVA
    num_ones_per_task = [int(n / 4) for n in num_ones_per_task_og]
    print(f"Average success rate {sum(num_ones_per_task_og)/len(num_ones_per_task_og) :.2f}" )
    # Perform the analysis
    avg_variance, variances, detailed_results = multi_task_bootstrap_analysis(num_ones_per_task)
    
    print("\n" + "="*60)
    print("SUMMARY RESULTS")
    print("="*60)
    print(f"Average variance across 9 tasks: {avg_variance:.2f}")
    print(f"Standard deviation of variances: {np.std(variances, ddof=1):.2f}")


# if __name__ == "__main__":
    # for task in ["close_jar", "push_buttons", "slide_block_to_color_target", "meat_off_grill", "put_money_in_safe", "sweep_to_dustpan_of_size", "open_drawer", "reach_and_drag", "turn_tap"]:
    #     json_fp_stats(f"/home/elvis/saywut/eval_data/{task}_fp_eval.json")
    # task = "meat_off_grill"
    # json_fp_stats(f"/home/elvis/saywut/eval_data/{task}_fp_eval.json")