import json


# TASKS: close_jar       push_buttons       slide_block_to_color_target
# meat_off_grill  put_money_in_safe  sweep_to_dustpan_of_size
# open_drawer     reach_and_drag     turn_tap

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