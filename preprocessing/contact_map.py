import numpy as np
from Bio.PDB import PDBParser

def compute_contact_map(pdb_file, threshold=8.0):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_file)
    coords = [res["CA"].get_coord() for res in structure.get_residues() if "CA" in res]
    coords = np.stack(coords)  # shape: (N, 3)
    # Compute pairwise distances
    diff = coords[:, None, :] - coords[None, :, :]  # shape: (N, N, 3)
    dist = np.linalg.norm(diff, axis=-1)            # shape: (N, N)
    contact_map = (dist <= threshold).astype(np.float32)
    return contact_map