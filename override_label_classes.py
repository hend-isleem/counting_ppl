import os
from glob import glob

def relabel_to_single_class(label_dir):
    label_files = glob(os.path.join(label_dir, '**/*.txt'), recursive=True)
    for path in label_files:
        with open(path, 'r') as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                print(f"Skipping line with insufficient parts: {line.strip()}")
                assert len(parts) == 0, f"Unexpected line format: {line.strip()}"
                # continue

            parts[0] = '0'  # the overriding part: set all classes to '0'
            new_lines.append(' '.join(parts))
        with open(path, 'w') as f:
            f.write('\n'.join(new_lines))

# Run on both train and valid sets
relabel_to_single_class("data/aerial-pool/train")
relabel_to_single_class("data/aerial-pool/valid")
relabel_to_single_class("data/aerial-pool/test")

print("!!!!!!!!!! Relabeling done. All classes mapped to 'person' !!!!!!!!!!")
