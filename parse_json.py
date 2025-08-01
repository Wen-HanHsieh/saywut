import json
import re
import os

def parse_robot_data(json_data, eval=False, eval_task=None):
    """Parse robot data to extract task, episode, and step while preserving original structure."""
    result = []
    
    # Process each item in the JSON data
    for item in json_data:
        try:
            # Create a new dictionary with all original data
            new_item = {
                "id": item["id"],
                "image": item["image"],
                "conversations": item["conversations"]
            }
            
            # Extract the image path
            image_path = item["image"]
            if eval:
                # Extract task name (first segment of the path)
                task = image_path.split('/')[0]
                
                # Extract episode number
                episode_match = re.search(r'episode(\d+)', image_path)
                episode = int(episode_match.group(1)) if episode_match else None

                # Extract step number (the filename number before .png)
                step_match = re.search(r'/(\d+)\.png$', image_path)
                step = int(step_match.group(1)) if step_match else None
                
                # Add these extracted fields
                new_item["task"] = eval_task
                new_item["episode"] = episode
                new_item["step"] = step

            else:
                # Extract task name (first segment of the path)
                task = image_path.split('/')[0]
                
                # Extract episode number
                episode_match = re.search(r'episode(\d+)', image_path)
                episode = int(episode_match.group(1)) if episode_match else None
                
                # Extract step number (the filename number before .png)
                step_match = re.search(r'/(\d+)\.png$', image_path)
                step = int(step_match.group(1)) if step_match else None
                
                # Add these extracted fields
                new_item["task"] = task
                new_item["episode"] = episode
                new_item["step"] = step
            
            # Add the item to the result
            result.append(new_item)
        except Exception as e:
            print(f"Error processing item: {e}")
    
    return result

# Main execution
if __name__ == "__main__":
    try:
        tasks = ['close_jar', 'push_buttons', 'slide_block_to_color_target', 'meat_off_grill', 'put_money_in_safe', 'sweep_to_dustpan_of_size', 'open_drawer', 'reach_and_drag', 'turn_tap']
        for task in tasks:
            input_path = f'/home/wenhan/Projects/saywut/sim/data/anns_eval/{task}/no_trace/eval.json'
            # input_path = f'/home/wenhan/Projects/saywut/multi_task_output/multi_task_200.json'
            # Read the file content
            with open(input_path, 'r') as file:
                json_data = json.load(file)
            
            # Parse the data
            parsed_data = parse_robot_data(json_data, eval=True, eval_task=task)
            
            # Save the output to a new JSON file
            output_path = f'eval_data/{task}_eval.json'
            with open(output_path, 'w') as output_file:
                json.dump(parsed_data, output_file, indent=2)
            
            print(f"Successfully parsed {len(parsed_data)} entries. Output written to parsed_data.json")
    except Exception as e:
        print(f"Error: {e}")