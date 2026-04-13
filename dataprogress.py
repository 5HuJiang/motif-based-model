
import pandas as pd
import numpy as np
from MolTokenizer import MolTokenizer
import rdkit.Chem as Chem
import dgl
import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
import os
from may22_utils import get_feature_data, get_fps, scaffold_split

class GraphDataset(Dataset):
    def __init__(self, smiles, labels, all_feature_data, tokenizer):
        self.smiles = smiles
        self.labels = labels
        self.all_feature_data = all_feature_data
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        smi = self.smiles[idx]
        # smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        mol = Chem.MolFromSmiles(smi)
        fps = get_fps(mol)
        try:
            motif_ids, edges, ids, subset_bonds = self.tokenizer.tokenize(smi)
            motifs = [self.tokenizer.id_to_token[i] for i in motif_ids]
            feature_data = np.array(
                self.all_feature_data.loc[motifs].values,
                dtype=np.float32
            )
            mol_feat = np.array(
                self.all_feature_data.loc[smi].values,
                dtype=np.float32
            )
        
            src = [e[0] for e in edges] + [e[1] for e in edges]
            dst = [e[1] for e in edges] + [e[0] for e in edges]

            g = dgl.graph((src, dst))
        
            if g.number_of_nodes() <= 1 :
                raise ValueError(f"Invalid graph: only 1 or 0node for SMILES '{smi}'.")
        
        except:
            raise ValueError(f"Invalid SMILES string: '{smi}'.")

        g.ndata['feat'] = torch.tensor(feature_data, dtype=torch.float32)
        g.ndata['motif_id'] = torch.tensor(motif_ids, dtype=torch.long)
        


        return g, label, torch.tensor(mol_feat, dtype=torch.float32), fps, smi
    
class GraphDataModule(LightningDataModule):
    def __init__(self, train_path, feature_path, tokenizer, batch_size, seed, val_path=None, test_path=None, shuffle=True):
        super().__init__()


        self.feature_data = pd.read_parquet(feature_path)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
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
        self.train_dataset = GraphDataset(self.train_df['SMILES'], self.train_df['label'], self.feature_data, self.tokenizer)
        if self.val_df is not None:
            self.val_dataset = GraphDataset(self.val_df['SMILES'], self.val_df['label'], self.feature_data, self.tokenizer)
        if self.test_df is not None:
            self.test_dataset = GraphDataset(self.test_df['SMILES'], self.test_df['label'], self.feature_data, self.tokenizer)

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
    def dataloader(self):
        return DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=self.collate_fn,
            num_workers=0,
            persistent_workers=False
            )
    
    def collate_fn(self, batch):
        """
        将多个图和标签整理成批量数据
        - 输入：batch = [(graph1, label1), (graph2, label2), ...]
        - 输出：batched_graph（DGL批量图）、labels（张量）
        """
        graphs, labels, mol_feat, fps, smiles = zip(*batch)
        # 批量处理图（DGL的batch函数会自动拼接图）
        batched_graph = dgl.batch(graphs)
        # 标签拼接成张量
        labels = torch.stack(labels)
        mol_feat = torch.stack(mol_feat)
        fps = torch.stack(fps)
        
        return batched_graph, labels, mol_feat, fps, list(smiles)
def data_pre(seed):
    data = pd.read_csv('/home/wzn/sdbw6/lwl/data/DIQT/DIQT.csv')
    train_df, val_df, test_df = scaffold_split(data, random_seed=seed)
    train_df.to_csv("/home/wzn/sdbw6/lwl/data/DIQT/train.csv", index=False)
    val_df.to_csv("/home/wzn/sdbw6/lwl/data/DIQT/val.csv", index=False)
    test_df.to_csv("/home/wzn/sdbw6/lwl/data/DIQT/test.csv", index=False)


if __name__ == '__main__':
    data = pd.read_csv('/home/wzn/sdbw6/lwl/data/faers/faers0923.csv')
    train_df, val_df, test_df = scaffold_split(data, random_seed=2025)
    train_df.to_csv("/home/wzn/sdbw6/lwl/data/faers/train.csv", index=False)
    val_df.to_csv("/home/wzn/sdbw6/lwl/data/faers/val.csv", index=False)
    test_df.to_csv("/home/wzn/sdbw6/lwl/data/faers/test.csv", index=False)

    # data = pd.read_csv('/home/wzn/sdbw6/lwl/data/DIQT/DIQT.csv')
    # train_df, val_df, test_df = scaffold_split(data, random_seed=2025)
    # train_df.to_csv("/home/wzn/sdbw6/lwl/data/DIQT/train.csv", index=False)
    # val_df.to_csv("/home/wzn/sdbw6/lwl/data/DIQT/val.csv", index=False)
    # test_df.to_csv("/home/wzn/sdbw6/lwl/data/DIQT/test.csv", index=False)

    

        

