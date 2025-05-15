import json

input_path = 'eval.json'
output_path = 'eval.json'

# Load the original JSON
with open(input_path, 'r') as fin:
    data = json.load(fin)

# Write it back with 2‑space indentation
with open(output_path, 'w') as fout:
    json.dump(data, fout, indent=2)

print(f"Re‑indented JSON written to {output_path}")