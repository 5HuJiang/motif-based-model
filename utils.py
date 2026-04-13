import pandas as pd
import numpy as np
from MolTokenizer import MolTokenizer
import rdkit.Chem as Chem
from rdkit.Chem import MACCSkeys, AllChem, RDKFingerprint
import random
import dgl
import torch
import torch.nn as nn
from typing import Dict, List, Union, Optional
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import Subset
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch import multiprocessing
from rdkit.Chem import MACCSkeys, AllChem, RDKFingerprint
import os




def get_fps(mol):
    maccs_fp = list(MACCSkeys.GenMACCSKeys(mol))
    morgan_fp = list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024))
    rdkit_fp = list(RDKFingerprint(mol))

    fps = maccs_fp + morgan_fp + rdkit_fp

    return torch.tensor(fps, dtype=torch.float32)

def get_scaffold(smiles):
    """
    从SMILES字符串提取Bemis-Murcko骨架
    :param smiles: 分子的SMILES字符串
    :return: 骨架的SMILES（提取失败返回None）
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # 提取Murcko骨架（含环和连接环的键）
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaffold
    except:
        return None

def scaffold_split(
    df,
    smiles_col="SMILES",
    train_frac=0.7,
    val_frac=0.2,
    test_frac=0.1,
    random_seed=1111
):
    """
    基于分子骨架的数据集划分，确保不同集骨架无重叠
    :param df: 包含SMILES的DataFrame
    :param smiles_col: SMILES列名
    :param train_frac: 训练集比例
    :param val_frac: 验证集比例
    :param test_frac: 测试集比例
    :param random_seed: 随机种子（保证可复现）
    :return: train_df, val_df, test_df
    """
    # 校验比例和为1
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "比例和必须为1"
    
    # 设置随机种子
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # 1. 提取每个分子的骨架
    df = df.copy()
    df["scaffold"] = df[smiles_col].apply(get_scaffold)
    
    # 过滤无效SMILES/骨架
    df = df.dropna(subset=["scaffold"]).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("无有效分子骨架，请检查SMILES格式")
    
    # 2. 按骨架分组，记录每个骨架对应的分子索引
    scaffold_to_indices = defaultdict(list)
    for idx, scaffold in df["scaffold"].items():
        scaffold_to_indices[scaffold].append(idx)
    
    # 3. 随机打乱骨架列表（保证划分随机性）
    unique_scaffolds = list(scaffold_to_indices.keys())
    random.shuffle(unique_scaffolds)
    
    # 4. 按比例分配骨架到不同数据集
    total_scaffolds = len(unique_scaffolds)
    train_size = int(total_scaffolds * train_frac)
    val_size = int(total_scaffolds * val_frac)
    
    train_scaffolds = unique_scaffolds[:train_size]
    val_scaffolds = unique_scaffolds[train_size:train_size+val_size]
    test_scaffolds = unique_scaffolds[train_size+val_size:]
    
    # 5. 根据骨架获取各数据集的分子索引
    train_indices = []
    for scaffold in train_scaffolds:
        train_indices.extend(scaffold_to_indices[scaffold])
    
    val_indices = []
    for scaffold in val_scaffolds:
        val_indices.extend(scaffold_to_indices[scaffold])
    
    test_indices = []
    for scaffold in test_scaffolds:
        test_indices.extend(scaffold_to_indices[scaffold])
    
    # 6. 生成划分后的数据集
    train_df = df.loc[train_indices].drop(columns=["scaffold"]).reset_index(drop=True)
    val_df = df.loc[val_indices].drop(columns=["scaffold"]).reset_index(drop=True)
    test_df = df.loc[test_indices].drop(columns=["scaffold"]).reset_index(drop=True)
    
    # 打印划分信息
    print(f"总有效分子数: {len(df)}")
    print(f"训练集: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"验证集: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"测试集: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    # 验证骨架无重叠
    train_scaffold_set = set(train_scaffolds)
    val_scaffold_set = set(val_scaffolds)
    test_scaffold_set = set(test_scaffolds)
    assert len(train_scaffold_set & val_scaffold_set) == 0, "训练集和验证集骨架重叠"
    assert len(train_scaffold_set & test_scaffold_set) == 0, "训练集和测试集骨架重叠"
    assert len(val_scaffold_set & test_scaffold_set) == 0, "验证集和测试集骨架重叠"
    print("✅ 各数据集骨架无重叠")
    
    return train_df, val_df, test_df


def get_feature_data(smiles, all_feature_data, tokenizer):
    motif_ids, _, _, _ = tokenizer.tokenize(smiles)
    motifs = [tokenizer.id_to_token[i] for i in motif_ids]
    feature_data = pd.DataFrame(
        np.zeros_like(all_feature_data),
        index=all_feature_data.index,
        columns=all_feature_data.columns
    )
    feature_data.loc[motifs] = all_feature_data.loc[motifs]
    return feature_data

