This repository implements a complete pipeline for molecular motif-based representation learning and QT prediction model training.

# Get Started

## Environment Setup
`pip install -r requirements.txt`

For the use of MolFormer, please refer to:[https://github.com/IBM/molformer]

## File Description
- moltokenizer.py Split: molecules into corresponding motifs. The dataset must contain the SMILES of molecules.
- get_embeddings.ipynb: Encode the extracted molecular motifs into embedding vectors using MolFormer.
- dataprogress.py: Perform scaffold split on the processed dataset.
- trainqt.py: Train the prediction model using the preprocessed data and embeddings.

## Usage Pipeline
- Run moltokenizer.py to decompose molecules into motifs.
- Execute get_embeddings.ipynb to generate MolFormer embeddings for motifs.
- Use dataprogress.py to complete scaffold-based dataset splitting.
- Run trainqt.py to train the final model.


## Data Requirement
- The input dataset must include a valid SMILES column for molecular processing and motif extraction.