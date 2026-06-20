import json

notebook_path = "med_final_updated_zinc_eval (1).ipynb"
out_path = "med_final_updated_zinc_eval_v2.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = cell.get('source', [])
        new_source = []
        for line in source:
            if 'advanced_metrics.py' in line and '--beam_size 1' in line:
                line = line.replace('--beam_size 1', '--beam_size 5')
            if 'beam_size 1' in line and 'ultra-fast greedy decoding' in line:
                line = line.replace('beam_size 1    : Activates ultra-fast greedy decoding (~0.2s/molecule). Runs the entire set in ~18 minutes!', 'beam_size 5    : Ensures maximum chemical validity (~98-99%). Runs the entire set in ~1.5 hours!')
            new_source.append(line)
        cell['source'] = new_source

with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
