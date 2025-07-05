import os
import requests
from Bio.PDB import PDBParser
import gzip
import shutil
from data.utils.residue_map import residue_3_to_1

def download_pdb_structure(pdb_id: str, output_dir: str = "data/pdb_files") -> str:
    os.makedirs(output_dir, exist_ok=True)
    pdb_id = pdb_id.upper()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb.gz"
    gz_file = os.path.join(output_dir, f"{pdb_id}.pdb.gz")
    pdb_file = os.path.join(output_dir, f"{pdb_id}.pdb")
    if not os.path.exists(pdb_file):
        response = requests.get(url)
        response.raise_for_status()
        with open(gz_file, 'wb') as f:
            f.write(response.content)
        with gzip.open(gz_file, 'rb') as f_in:
            with open(pdb_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(gz_file)
    return pdb_file

def extract_sequence_from_pdb(pdb_file: str) -> str:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    sequence = ""
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == " ":
                    res_name = residue.resname
                    if res_name in residue_3_to_1:
                        sequence += residue_3_to_1[res_name]
    return sequence

# ... existing code ...
# (keep download_pdb_structure and extract_sequence_from_pdb)
# Remove find_similar_proteins function entirely
# ... existing code ... 