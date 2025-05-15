import json
import random
import copy
import re
from collections import defaultdict

# Define false premises for different tasks
dict_false_premise_easy = {
        "open_drawer": [
            "open the top elephant",
            "open the top car",
            "open the middle tree",
            "open the middle bicycle",
            "open the bottom couch",
            "open the bottom book",
        ],
        "meat_off_grill": [
            "take the elephant off the grill",
            "take the car off the grill",
            "take the tree off the grill",
            "take the bicycle off the grill",
            "take the couch off the grill",
            "take the book off the grill",
        ],
        "turn_tap": [
            "turn left elephant",
            "turn right elephant",
            "turn left car",
            "turn right car",
            "turn left tree",
            "turn right tree",
        ],
        "put_money_in_safe": [
            "put the money away in the elephant on the bottom shelf",
            "put the money away in the car on the bottom shelf",
            "put the money away in the tree on the middle shelf",
            "put the money away in the bicycle on the middle shelf",
            "put the money away in the couch on the top shelf",
            "put the money away in the book on the top shelf",
        ],
        "push_buttons": [
            "push the maroon elephant",
            "push the maroon car",
            "push the maroon tree",
            "push the maroon bicycle",
            "push the maroon couch",
            "push the maroon book",
        ],
        "slide_block_to_color_target": [
            "slide the elephant to blue target",
            "slide the car to green target",
            "slide the tree to pink target",
            "slide the bicycle to yellow target",
            "slide the couch to blue target",
            "slide the book to green target",
        ],
        "close_jar": [
            "close the blue elephant",
            "close the teal elephant",
            "close the blue car",
            "close the teal car",
            "close the blue tree",
            "close the teal tree",
        ],
        "sweep_to_dustpan_of_size": [
            "sweep elephant to the short dustpan",
            "sweep car to the short dustpan",
            "sweep tree to the tall dustpan",
            "sweep bicycle to the tall dustpan",
            "sweep couch to the short dustpan",
            "sweep book to the tall dustpan",
        ],
        "reach_and_drag": [
            "use the stick to drag the elephant onto the magenta target",
            "use the stick to drag the car onto the magenta target",
            "use the stick to drag the tree onto the maroon target",
            "use the stick to drag the bicycle onto the maroon target",
            "use the stick to drag the couch onto the teal target",
            "use the stick to drag the book onto the teal target",
        ],
        # Legacy mappings for backward compatibility
        "sweep_dirt_to_dustpan": ["sweep dirt to the ceiling fan", "sweep dirt to the trash bin", 
                                  "sweep dust to the tall dustpan", "sweep dirt to the floor"],
    }
    
dict_false_premise_hard = {
        "open_drawer": [
            "open the top chicken",
            "open the top tap",
            "open the middle safe",
            "open the middle block",
            "open the bottom jar",
            "open the bottom cube",
        ],
        "meat_off_grill": [
            "take the drawer off the grill",
            "take the tap off the grill",
            "take the safe off the grill",
            "take the block off the grill",
            "take the jar off the grill",
            "take the cube off the grill",
        ],
        "turn_tap": [
            "turn left drawer",
            "turn right drawer",
            "turn left chicken",
            "turn right chicken",
            "turn left safe",
            "turn right block",
        ],
        "put_money_in_safe": [
            "put the money away in the drawer on the bottom shelf",
            "put the money away in the chicken on the bottom shelf",
            "put the money away in the tap on the middle shelf",
            "put the money away in the block on the middle shelf",
            "put the money away in the jar on the top shelf",
            "put the money away in the cube on the top shelf",
        ],
        "push_buttons": [
            "push the maroon drawer",
            "push the maroon chicken",
            "push the maroon tap",
            "push the maroon safe",
            "push the maroon block",
            "push the maroon jar",
        ],
        "slide_block_to_color_target": [
            "slide the drawer to blue target",
            "slide the chicken to green target",
            "slide the tap to pink target",
            "slide the safe to yellow target",
            "slide the jar to blue target",
            "slide the cube to green target",
        ],
        "close_jar": [
            "close the blue drawer",
            "close the teal drawer",
            "close the blue chicken",
            "close the teal tap",
            "close the blue safe",
            "close the teal block",
        ],
        "sweep_to_dustpan_of_size": [
            "sweep drawer to the short dustpan",
            "sweep chicken to the short dustpan",
            "sweep tap to the tall dustpan",
            "sweep safe to the tall dustpan",
            "sweep block to the short dustpan",
            "sweep jar to the tall dustpan",
        ],
        "reach_and_drag": [
            "use the stick to drag the drawer onto the magenta target",
            "use the stick to drag the chicken onto the maroon target",
            "use the stick to drag the tap onto the teal target",
            "use the stick to drag the safe onto the magenta target",
            "use the stick to drag the block onto the maroon target",
            "use the stick to drag the jar onto the teal target",
        ],
        # Legacy mappings for backward compatibility
        "sweep_dirt_to_dustpan": ["use the broom as a baseball bat", "clean the entire kitchen", 
                                  "sweep dirt under the rug", "use the broom to knock items off the table"],
    }

remapping_tasks = ["open_drawer","turn_tap","put_money_in_safe","slide_block_to_color_target","close_jar","sweep_to_dustpan_of_size","reach_and_drag"]

remapping_dict = {
    # close_jar
    "close the blue car":    ("car",    "jar"),
    "close the blue chicken":("chicken","jar"),
    "close the blue drawer": ("drawer", "jar"),
    "close the blue elephant":("elephant","jar"),
    "close the blue safe":   ("safe",   "jar"),
    "close the blue tree":   ("tree",   "jar"),
    "close the teal block":  ("block",  "jar"),
    "close the teal car":    ("car",    "jar"),
    "close the teal drawer": ("drawer", "jar"),
    "close the teal elephant":("elephant","jar"),
    "close the teal tap":    ("tap",    "jar"),
    "close the teal tree":   ("tree",   "jar"),
    # open_drawer
    "open the bottom book":  ("book",   "drawer"),
    "open the bottom couch": ("couch",  "drawer"),
    "open the bottom cube":  ("cube",   "drawer"),
    "open the bottom jar":   ("jar",    "drawer"),
    "open the middle bicycle":("bicycle","drawer"),
    "open the middle block": ("block",  "drawer"),
    "open the middle safe":  ("safe",   "drawer"),
    "open the middle tree":  ("tree",   "drawer"),
    "open the top car":      ("car",    "drawer"),
    "open the top chicken":  ("chicken","drawer"),
    "open the top elephant": ("elephant","drawer"),
    "open the top tap":      ("tap",    "drawer"),
    # push_buttons
    "push the maroon bicycle":("bicycle","button"),
    "push the maroon block":  ("block",  "button"),
    "push the maroon book":   ("book",   "button"),
    "push the maroon car":    ("car",    "button"),
    "push the maroon chicken":("chicken","button"),
    "push the maroon couch":  ("couch",  "button"),
    "push the maroon drawer": ("drawer", "button"),
    "push the maroon elephant":("elephant","button"),
    "push the maroon jar":    ("jar",    "button"),
    "push the maroon safe":   ("safe",   "button"),
    "push the maroon tap":    ("tap",    "button"),
    "push the maroon tree":   ("tree",   "button"),
    # put_money_in_safe
    "put the money away in the bicycle on the middle shelf": ("bicycle","safe"),
    "put the money away in the block on the middle shelf":   ("block",  "safe"),
    "put the money away in the book on the top shelf":      ("book",   "safe"),
    "put the money away in the car on the bottom shelf":    ("car",    "safe"),
    "put the money away in the chicken on the bottom shelf":("chicken","safe"),
    "put the money away in the couch on the top shelf":     ("couch",  "safe"),
    "put the money away in the cube on the top shelf":      ("cube",   "safe"),
    "put the money away in the drawer on the bottom shelf": ("drawer", "safe"),
    "put the money away in the elephant on the bottom shelf":("elephant","safe"),
    "put the money away in the jar on the top shelf":       ("jar",    "safe"),
    "put the money away in the tap on the middle shelf":    ("tap",    "safe"),
    "put the money away in the tree on the middle shelf":   ("tree",   "safe"),
    # slide_block_to_color_target
    "slide the bicycle to yellow target":  ("bicycle","block"),
    "slide the book to green target":      ("book",   "block"),
    "slide the car to green target":       ("car",    "block"),
    "slide the chicken to green target":   ("chicken","block"),
    "slide the couch to blue target":      ("couch",  "block"),
    "slide the cube to green target":      ("cube",   "block"),
    "slide the drawer to blue target":     ("drawer", "block"),
    "slide the elephant to blue target":   ("elephant","block"),
    "slide the jar to blue target":        ("jar",    "block"),
    "slide the safe to yellow target":     ("safe",   "block"),
    "slide the tap to pink target":        ("tap",    "block"),
    "slide the tree to pink target":       ("tree",   "block"),
    # sweep_to_dustpan_of_size
    "sweep bicycle to the tall dustpan":   ("bicycle","dirt"),
    "sweep block to the short dustpan":    ("block",  "dirt"),
    "sweep book to the tall dustpan":      ("book",   "dirt"),
    "sweep car to the short dustpan":      ("car",    "dirt"),
    "sweep chicken to the short dustpan":  ("chicken","dirt"),
    "sweep couch to the short dustpan":    ("couch",  "dirt"),
    "sweep drawer to the short dustpan":   ("drawer", "dirt"),
    "sweep elephant to the short dustpan": ("elephant","dirt"),
    "sweep jar to the tall dustpan":       ("jar",    "dirt"),
    "sweep safe to the tall dustpan":      ("safe",   "dirt"),
    "sweep tap to the tall dustpan":       ("tap",    "dirt"),
    "sweep tree to the tall dustpan":      ("tree",   "dirt"),
    # meat_off_grill
    "take the bicycle off the grill":      ("bicycle","steak"),
    "take the block off the grill":        ("block",  "steak"),
    "take the book off the grill":         ("book",   "steak"),
    "take the car off the grill":          ("car",    "steak"),
    "take the couch off the grill":        ("couch",  "chicken"),
    "take the cube off the grill":         ("cube",   "steak"),
    "take the drawer off the grill":       ("drawer", "steak"),
    "take the elephant off the grill":     ("elephant","chicken"),
    "take the jar off the grill":          ("jar",    "chicken"),
    "take the safe off the grill":         ("safe",   "chicken"),
    "take the tap off the grill":          ("tap",    "chicken"),
    "take the tree off the grill":         ("tree",   "chicken"),
    # turn_tap
    "turn left car":                       ("car",    "tap"),
    "turn left chicken":                   ("chicken","tap"),
    "turn left drawer":                    ("drawer", "tap"),
    "turn left elephant":                  ("elephant","tap"),
    "turn left safe":                      ("safe",   "tap"),
    "turn left tree":                      ("tree",   "tap"),
    "turn right block":                    ("block",  "tap"),
    "turn right car":                      ("car",    "tap"),
    "turn right chicken":                  ("chicken","tap"),
    "turn right drawer":                   ("drawer", "tap"),
    "turn right elephant":                 ("elephant","tap"),
    "turn right tree":                     ("tree",   "tap"),
    # reach_and_drag
    "use the stick to drag the bicycle onto the maroon target": ("bicycle","cube"),
    "use the stick to drag the block onto the maroon target":   ("block",  "cube"),
    "use the stick to drag the book onto the teal target":      ("book",   "cube"),
    "use the stick to drag the car onto the magenta target":    ("car",    "cube"),
    "use the stick to drag the chicken onto the maroon target":("chicken","cube"),
    "use the stick to drag the couch onto the teal target":     ("couch",  "cube"),
    "use the stick to drag the drawer onto the magenta target":("drawer", "cube"),
    "use the stick to drag the elephant onto the magenta target":("elephant","cube"),
    "use the stick to drag the jar onto the teal target":       ("jar",    "cube"),
    "use the stick to drag the safe onto the magenta target":   ("safe",   "cube"),
    "use the stick to drag the tap onto the teal target":       ("tap",    "cube"),
    "use the stick to drag the tree onto the maroon target":    ("tree",   "cube"),
}

def select_matching_false_premise(task_type, original_task, is_easy=True):
    """
    Select a false premise that matches key characteristics of the original task
    
    Args:
        task_type (str): The type of task (e.g., "open_drawer")
        original_task (str): The original task description
        is_easy (bool): Whether to select an easy or hard false premise
        
    Returns:
        str: A matching false premise
    """
    premise_dict = dict_false_premise_easy if is_easy else dict_false_premise_hard
    
    if task_type not in premise_dict or task_type not in remapping_tasks:
        # Fall back to random selection if task not in special list
        return random.choice(premise_dict.get(task_type, ["unknown task"]))
    
    # Extract key information based on task type
    if task_type == "open_drawer":
        # Match top/middle/bottom
        position_match = re.search(r"open the (top|middle|bottom)", original_task)
        if position_match:
            position = position_match.group(1)
            matching_premises = [p for p in premise_dict[task_type] if position in p]
            if matching_premises:
                return random.choice(matching_premises)
    
    elif task_type == "turn_tap":
        # Match left/right
        direction_match = re.search(r"turn (left|right)", original_task)
        if direction_match:
            direction = direction_match.group(1)
            matching_premises = [p for p in premise_dict[task_type] if direction in p]
            if matching_premises:
                return random.choice(matching_premises)
    
    elif task_type == "put_money_in_safe":
        # Match shelf position
        shelf_match = re.search(r"(bottom|middle|top) shelf", original_task)
        if shelf_match:
            shelf = shelf_match.group(1)
            matching_premises = [p for p in premise_dict[task_type] if shelf in p]
            if matching_premises:
                return random.choice(matching_premises)
    
    elif task_type == "slide_block_to_color_target":
        # Match target color
        color_match = re.search(r"to (blue|green|pink|yellow|red) target", original_task)
        if color_match:
            color = color_match.group(1)
            matching_premises = [p for p in premise_dict[task_type] if color in p]
            if matching_premises:
                return random.choice(matching_premises)
    
    elif task_type == "close_jar":
        # Match jar color
        color_match = re.search(r"close the (blue|teal)", original_task)
        if color_match:
            color = color_match.group(1)
            matching_premises = [p for p in premise_dict[task_type] if color in p]
            if matching_premises:
                return random.choice(matching_premises)
    
    elif task_type == "sweep_to_dustpan_of_size":
        # Match dustpan size
        size_match = re.search(r"to the (short|tall) dustpan", original_task)
        if size_match:
            size = size_match.group(1)
            matching_premises = [p for p in premise_dict[task_type] if size in p]
            if matching_premises:
                return random.choice(matching_premises)
    
    elif task_type == "reach_and_drag":
        # Match target color
        color_match = re.search(r"onto the (magenta|maroon|teal) target", original_task)
        if color_match:
            color = color_match.group(1)
            matching_premises = [p for p in premise_dict[task_type] if color in p]
            if matching_premises:
                return random.choice(matching_premises)
    
    # If no match found or extraction failed, fall back to random selection
    print(f"No matching false premise found for task: {original_task}, task_type: {task_type}")
    raise ValueError(f"No matching false premise found for task: {original_task}")

def insert_false_premises(input_file, output_file, hard_premise_ratio=0.2, easy_premise_ratio=0.5, easy_steps_per_episode=10):
    """
    Insert false premises into the JSON dataset
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path to save the modified JSON file
        hard_premise_ratio (float): Ratio of episodes to modify with hard false premises (default: 0.2)
        easy_premise_ratio (float): Ratio of episodes to modify with easy false premises (default: 0.5)
        easy_steps_per_episode (int): Number of steps to inject easy false premises in each episode (default: 10)
    """
    # Load the JSON data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Create a new list for modified data
    modified_data = []
    
    # Organize data by task and episode
    organized_data = defaultdict(lambda: defaultdict(list))
    for item in data:
        task = item.get("task", "unknown")
        episode = item.get("episode", 0)
        organized_data[task][episode].append(item)
    
    # Track which episodes will have hard false premises
    hard_premise_episodes = {}  # {task: {episode: step_index}}
    
    # Track which episodes will have easy false premises
    easy_premise_episodes = {}  # {task: {episode: [step_indices]}}
    
    # Select episodes to modify
    for task, episodes in organized_data.items():
        if task not in dict_false_premise_easy or task not in dict_false_premise_hard:
            continue
            
        # Get all episode numbers for this task
        all_episodes = list(episodes.keys())
        total_episodes = len(all_episodes)
        
        # Calculate number of episodes to modify for hard and easy premises
        num_hard_episodes = int(total_episodes * hard_premise_ratio)
        num_easy_episodes = int(total_episodes * easy_premise_ratio)
        
        
        # Select episodes for hard false premises
        hard_episodes = random.sample(all_episodes, num_hard_episodes)
        
        # Select episodes for easy false premises from the remaining episodes
        remaining_episodes = [ep for ep in all_episodes if ep not in hard_episodes]
        num_easy_episodes = min(num_easy_episodes, len(remaining_episodes))
        easy_episodes = random.sample(remaining_episodes, num_easy_episodes)
        
        # For each episode with hard false premises, select a random step
        for episode in hard_episodes:
            episode_data = episodes[episode]
            step_index = random.randint(0, len(episode_data) - 1)
            
            if task not in hard_premise_episodes:
                hard_premise_episodes[task] = {}
            hard_premise_episodes[task][episode] = step_index
            
        # For each episode with easy false premises, select random steps
        for episode in easy_episodes:
            episode_data = episodes[episode]
            num_steps = len(episode_data)
            
            # Determine how many easy false premises to insert (up to 10 steps or as many as available)
            num_false_premises = min(easy_steps_per_episode, num_steps)
            
            # Select random steps to modify
            steps_to_modify = random.sample(range(num_steps), num_false_premises)
            
            if task not in easy_premise_episodes:
                easy_premise_episodes[task] = {}
            easy_premise_episodes[task][episode] = steps_to_modify
            
        print(f"Task: {task}, Total episodes: {total_episodes}")
        print(f"  Hard episodes: {len(hard_episodes)} ({hard_premise_ratio*100:.1f}%)")
        print(f"  Easy episodes: {len(easy_episodes)} ({easy_premise_ratio*100:.1f}%)")
    
    # Now process each item from the original data
    for index, item in enumerate(data):
        task = item.get("task", "unknown")
        episode = item.get("episode", 0)
        step = item.get("step", 0)
        
        # Create a deep copy of the item
        modified_item = copy.deepcopy(item)
        
        # Check if this episode has a hard false premise and we've reached that step
        if (task in hard_premise_episodes and 
            episode in hard_premise_episodes[task] and 
            step == hard_premise_episodes[task][episode]):
            
            # Apply the HARD false premise modification
            if "conversations" in modified_item:
                # Get original task description
                human_message = modified_item["conversations"][0]["value"]
                
                # Extract the original task using the improved regex
                task_match = re.search(r'The task is \"(.*?)\"(?:,|\s|$)', human_message)
                if task_match:
                    original_task = task_match.group(1)
                    
                    # Select a matching false premise
                    false_task = select_matching_false_premise(task, original_task, is_easy=False)
                    
                    # Replace the task with the false premise
                    modified_message = human_message.replace(f'The task is "{original_task}"', 
                                                            f'The task is "{false_task}"')
                    
                    # Create a new conversation flow with just two messages
                    new_conversations = [
                        {"from": "human", "value": modified_message}
                    ]
                    
                    # Generate appropriate response text
                    response_text = f"I couldn't find a {remapping_dict[false_task][0]} in the current scene."
                    
                    new_conversations.append({
                        "from": "gpt", 
                        "value": response_text
                    })
                    
                    # Update the conversations
                    modified_item["conversations"] = new_conversations
                    modified_item["is_false_premise"] = True
                    modified_item["is_easy_false_premise"] = False
            
            # Add this item to the modified data
            modified_data.append(modified_item)
            
            # Skip all subsequent steps in this episode
            continue
        
        # Check if we should skip this item because it's after a hard false premise
        if (task in hard_premise_episodes and 
            episode in hard_premise_episodes[task] and 
            step > hard_premise_episodes[task][episode]):
            # Skip this item
            continue
        
        # Check if this episode has been selected for easy false premises
        if (task in easy_premise_episodes and
            episode in easy_premise_episodes[task] and
            step in easy_premise_episodes[task][episode]):

            # Apply the EASY false premise modification
            if "conversations" in modified_item:
                # Get original task description
                human_message = modified_item["conversations"][0]["value"]
                
                # Extract the original task using the improved regex
                task_match = re.search(r'The task is \"(.*?)\"(?:,|\s|$)', human_message)
                if task_match:
                    original_task = task_match.group(1)
                    
                    # Select a matching false premise
                    false_task = select_matching_false_premise(task, original_task, is_easy=True)
                    
                    # Replace the task with the false premise
                    modified_message = human_message.replace(f'The task is "{original_task}"', 
                                                            f'The task is "{false_task}"')
                    
                    # Create a new conversation flow
                    original_conversations = modified_item["conversations"]
                    
                    # First message with false premise
                    new_conversations = [
                        {"from": "human", "value": modified_message}
                    ]
                    
                    # Generate appropriate error message
                    error_message = f"I don't see {remapping_dict[false_task][0]} in the current scene. Do you mean {remapping_dict[false_task][1]}?"
                    
                    new_conversations.append({
                        "from": "gpt", 
                        "value": error_message
                    })
                    
                    # Human corrects with "Yes, " prefix
                    # Remove <image>\n if present
                    if human_message.startswith("<image>\n"):
                        cleaned_message = human_message[8:]
                    else:
                        cleaned_message = human_message
                        
                    new_conversations.append({
                        "from": "human",
                        "value": f"Yes, {cleaned_message}"
                    })
                    
                    # Bot gives the original answer
                    new_conversations.append(original_conversations[1])  # Original bot response
                    
                    # Update the conversations
                    modified_item["conversations"] = new_conversations
                    modified_item["is_false_premise"] = True
                    modified_item["is_easy_false_premise"] = True
                else:
                    raise ValueError(f"No task found in {human_message}")

        modified_data.append(modified_item)
    
    # Save the modified JSON
    with open(output_file, 'w') as f:
        json.dump(modified_data, f, indent=2)
    
    # Count the number of modified entries by category
    easy_false_premises = 0
    hard_false_premises = 0
    
    for item in modified_data:
        if item.get("is_false_premise", False):
            if item.get("is_easy_false_premise", False):
                easy_false_premises += 1
            else:
                hard_false_premises += 1
    
    # Print summary statistics
    print(f"Successfully modified {easy_false_premises + hard_false_premises} entries")
    print(f"- Easy false premises: {easy_false_premises}")
    print(f"- Hard false premises: {hard_false_premises}")
    print(f"Total entries processed: {len(data)}")
    print(f"Total entries in output: {len(modified_data)}")
    
    # Count modified episodes
    modified_episode_count = len(set([(item.get("task"), item.get("episode")) 
                                    for item in modified_data 
                                    if item.get("is_false_premise", False)]))
    
    total_episode_count = sum(len(episodes) for episodes in organized_data.values())
    
    print(f"Modified {modified_episode_count} episodes out of {total_episode_count} total episodes ({modified_episode_count/total_episode_count*100:.1f}%)")
    print(f"Modified JSON saved to {output_file}")

# Example usage
if __name__ == "__main__":
    input_file = "parsed_data.json"
    output_file = "robotic_tasks_with_false_premises.json"
    insert_false_premises(
        input_file, 
        output_file, 
        hard_premise_ratio=0.2,     # 20% of episodes get hard false premises
        easy_premise_ratio=0.65,     # 65% of episodes get easy false premises
        easy_steps_per_episode=20   # 20 steps per episode get easy false premises
    )