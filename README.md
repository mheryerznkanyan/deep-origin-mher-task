# Protein Contact Prediction

This project predicts protein contact maps from amino acid sequences using deep learning and ESM2 embeddings.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **(Optional) Download or prepare your PDB files** in `data/pdb_files/train/` and `data/pdb_files/test/`.

## Data Preparation

1. **Generate .pkl files:**
   ```bash
   python preprocessing/compute_contact_maps.py
   ```
   This creates `.pkl` files with contact maps and sequences in `precomputed/train/` and `precomputed/test/`.

2. **No manual changes needed:**
   The data module will automatically use these files for training and testing.

## Model Architecture

- **ESM2 Embedding Projection:** Projects sequence embeddings to a hidden dimension.
- **Transformer Encoder:** Captures sequence context.
- **Pairwise Feature Construction:** Concatenates residue embeddings for each pair.
- **Structural Features:** (Optional) Concatenated to pairwise features.
- **Contact Prediction Head:** MLP predicts contact probability for each residue pair.
- **Loss:** Binary cross-entropy on the contact map.

## Training & Testing

1. **Configure your experiment** in the `configs/` directory (see `configs/data/default.yaml` and `configs/model/default.yaml`).
2. **Train the model:**
   ```bash
   python train_hydra.py
   ```
3. **Test or predict:**
   ```bash
   python predict_hydra.py
   ```

## Configuration

Key configuration files are in the `configs/` directory:

- `configs/data/default.yaml`: Controls data loading, batch size, sequence length, and data splits.
- `configs/model/default.yaml`: Sets model architecture parameters (hidden size, number of layers, dropout, etc.).
- `configs/trainer/default.yaml`: Trainer settings (epochs, device, precision, callbacks).

You can adjust these files or override parameters from the command line, e.g.:
```bash
python train_hydra.py data.batch_size=8 model.hidden_dim=1024 trainer.max_epochs=100
```
