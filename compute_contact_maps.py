import os
import glob
import numpy as np
import pickle
from multiprocessing import Pool, cpu_count
from preprocessing.contact_map import compute_contact_map
from data.pdb_utils import extract_sequence_from_pdb

def process_pdb(args):
    pdb_file, out_path = args
    if os.path.exists(out_path):
        print(f"[SKIP] Contact map already exists: {out_path}")
        return
    try:
        contact_map = compute_contact_map(pdb_file)
        sequence = extract_sequence_from_pdb(pdb_file)
        if contact_map.shape[0] != len(sequence):
            print(f"[WARNING] {pdb_file}: contact map shape {contact_map.shape} does not match sequence length {len(sequence)}. Skipping.")
            return
        with open(out_path, 'wb') as f:
            pickle.dump({'contact_map': contact_map, 'sequence': sequence}, f)
        print(f"Saved contact map and sequence: {out_path}")
    except Exception as e:
        print(f"[ERROR] {pdb_file}: {e}")

def main():
    # Directories
    splits = [ 'train']
    input_base = 'data/pdb_files'
    output_base = 'precomputed'

    from itertools import islice

    BATCH_SIZE = 8  # You can adjust this batch size as needed

    def batched(iterable, n):
        """Batch data into lists of length n. The last batch may be shorter."""
        it = iter(iterable)
        while True:
            batch = list(islice(it, n))
            if not batch:
                break
            yield batch

    for split in splits:
        input_dir = os.path.join(input_base, split)
        output_dir = os.path.join(output_base, split)
        os.makedirs(output_dir, exist_ok=True)
        pdb_files = glob.glob(os.path.join(input_dir, '*.pdb'))
        print(f"Processing {len(pdb_files)} PDB files in {split}...")
        args_list = [
            (pdb_file, os.path.join(output_dir, os.path.basename(pdb_file).replace('.pdb', '_contact_map.pkl')))
            for pdb_file in pdb_files
        ]
        with Pool(processes=cpu_count()) as pool:
            for batch in batched(args_list, BATCH_SIZE):
                pool.map(process_pdb, batch)

if __name__ == "__main__":
    main() 