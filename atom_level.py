from argparse import Namespace
from typing import List, Union
import dgl
import torch
from rdkit import Chem
import numpy as np
from torch.utils.data import Dataset, DataLoader

from MolTokenizer import MolTokenizer
from pytorch_lightning import LightningDataModule
from may22_utils import get_fps
import pandas as pd

token_path = "data/motifs_token_id.json"
tokenizer = MolTokenizer(token_path)
# Atom feature sizes
MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# Distance feature sizes
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14

# Memoization
SMILES_TO_GRAPH = {}


def clear_cache():
    """Clears featurization cache."""
    global SMILES_TO_GRAPH
    SMILES_TO_GRAPH = {}

##
def get_atom_fdim(args: Namespace) -> int:
    """
    Gets the dimensionality of atom features.

    :param: Arguments.
    """
    return ATOM_FDIM

##
def get_bond_fdim(args: Namespace) -> int:
    """
    Gets the dimensionality of bond features.

    :param: Arguments.
    """
    return BOND_FDIM

##
def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding

##
def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
           onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
           onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
           onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
           onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
           onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
           [1 if atom.GetIsAromatic() else 0] + \
           [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    if functional_groups is not None:
        features += functional_groups
    return features

##
def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.

    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond

def mol_to_dgl_graph(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    num_atoms = mol.GetNumAtoms()

    # 节点特征
    atom_feats = []
    for atom in mol.GetAtoms():
        atom_feats.append(atom_features(atom))

    atom_feats = torch.tensor(atom_feats, dtype=torch.float)

    # 边
    src, dst = [], []
    bond_feats = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bf = bond_features(bond)

        # 双向边
        src += [i, j]
        dst += [j, i]
        bond_feats += [bf, bf]

    if len(src) == 0:
        # 处理无键分子（如单原子）
        g = dgl.graph(([], []), num_nodes=num_atoms)
        g.ndata['feat'] = atom_feats
        g.edata['e'] = torch.zeros((0, len(bond_features(None))))
        return g

    g = dgl.graph((src, dst), num_nodes=num_atoms)

    g.ndata['feat'] = atom_feats
    g.edata['e'] = torch.tensor(bond_feats, dtype=torch.float)

    return g

def build_motif_graph(smiles, tokenizer):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # === 1. 原子图 ===
    atom_graph = mol_to_dgl_graph(smiles)
    atom_feats = atom_graph.ndata['feat']   # (N_atom, F)

    # === 2. motif 分解 ===
    motif_list, edges, cliques, subset_bonds = tokenizer.tokenize(smiles)

    num_motifs = len(cliques)

    # === 3. motif 节点特征 ===
    motif_feats = []
    for clique in cliques:
        atom_idx = torch.tensor(clique, dtype=torch.long)

        feat = atom_feats[atom_idx]   # (k, F)

        # --- pooling ---
        motif_feat = feat.mean(dim=0)   # or sum(dim=0)

        motif_feats.append(motif_feat)

    motif_feats = torch.stack(motif_feats, dim=0)  # (N_motif, F)

    # === 4. motif 边 ===
    if len(edges) == 0:
        g = dgl.graph(([], []), num_nodes=num_motifs)
        g.ndata['feat'] = motif_feats
        return g

    src, dst = [], []
    for i, j in edges:
        src += [i, j]
        dst += [j, i]

    g = dgl.graph((src, dst), num_nodes=num_motifs)
    g.ndata['feat'] = motif_feats

    return g



class AtomDataset(Dataset):
    def __init__(self, smiles, labels, tokenizer, motif=False):
        self.smiles = smiles
        self.labels = labels
        self.tokenizer = tokenizer
        self.motif = motif

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smi = self.smiles[idx]
        if self.motif is not True:
            graph = mol_to_dgl_graph(smi)
        else:
            graph = build_motif_graph(smi, self.tokenizer)
        if self.labels is None:
            return graph
        mol = Chem.MolFromSmiles(smi)
        fps = get_fps(mol)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return graph, label, torch.Tensor(0), fps, smi

class AtomDataModule(LightningDataModule):
    def __init__(self, train_path, batch_size, seed, tokenizer, val_path=None, test_path=None, shuffle=True, motif=False):
        super().__init__()

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.tokenizer = tokenizer
        self.motif = motif

        self.train_df = pd.read_csv(train_path)
        if val_path is not None:
            self.val_df = pd.read_csv(val_path)
        else:
            self.val_df = None
        if test_path is not None:
            self.test_df = pd.read_csv(test_path)
        else:
            self.test_df = None

    def setup(self, stage=None):
        self.train_dataset = AtomDataset(self.train_df['SMILES'], self.train_df['label'],  self.tokenizer, motif = self.motif)
        if self.val_df is not None:
            self.val_dataset = AtomDataset(self.val_df['SMILES'], self.val_df['label'], self.tokenizer, motif = self.motif)
        if self.test_df is not None:
            self.test_dataset = AtomDataset(self.test_df['SMILES'], self.test_df['label'], self.tokenizer, motif = self.motif)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=self.shuffle, 
            collate_fn=self.collate_fn, 
            num_workers=0,
            persistent_workers=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=self.collate_fn,
            num_workers=0,
            persistent_workers=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=self.collate_fn,
            num_workers=0,
            persistent_workers=False
        )

    def collate_fn(self, batch):
        graphs, labels, mol_feat, fps, smiles = zip(*batch)
        # 批量处理图（DGL的batch函数会自动拼接图）
        batched_graph = dgl.batch(graphs)
        # 标签拼接成张量
        labels = torch.stack(labels)
        mol_feat = torch.stack(mol_feat)
        fps = torch.stack(fps)
        
        return batched_graph, labels, mol_feat, fps, list(smiles)


if __name__ == "__main__":
    smiles = 'CCCS(=O)(=O)NC1=C(F)C(C(=O)C2=CNC3=NC=C(C=C23)C2=CC=C(Cl)C=C2)=C(F)C=C1'
    g = mol_to_dgl_graph(smiles)
    m_g = build_motif_graph(smiles, tokenizer)
    print(g.ndata['h'])
    print(m_g.ndata['h'].shape)