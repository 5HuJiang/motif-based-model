import copy
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import xml.etree.ElementTree as ET
from matplotlib import colors
import rdkit.Chem as Chem
import json
from rdkit import DataStructs
from rdkit import Geometry
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
from rdkit.Chem import rdMolDescriptors as rdMD
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from io import BytesIO
from PIL import Image

class MolTokenizer():
    def __init__(self, vocab_file):
        # 从指定的词汇文件加载词汇表
        self.vocab = json.load(open(vocab_file, 'r'))
        # 计算词汇表的大小
        self.vocab_size = len(self.vocab.keys())
        # 创建从 ID 到标记的反向映射字典
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        # 设置最大权重常量
        self.MST_MAX_WEIGHT = 100
        self.auto_add = True

    def add_to_vocab(self, token):
        """ 在内存中新增 token """
        if token not in self.vocab:
            new_id = len(self.vocab)
            self.vocab[token] = new_id
            self.id_to_token[new_id] = token
            self.vocab_size += 1
            return new_id
        return self.vocab[token]
    
    def tokenize(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string:{smiles}")
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        ids, edge, subset_bonds = self.tree_decomp(mol) 
        motif_list = []

        for id_ in ids:
            _, token_mols = self.get_clique_mol(mol, id_)
            token_id = self.vocab.get(token_mols)
            if token_id is not None:
                motif_list.append(token_id)
            else:
                if self.auto_add:
                    token_id = self.add_to_vocab(token_mols)
                    motif_list.append(token_id)
                else:
                    motif_list.append(self.vocab.get('<unk>'))
        return motif_list, edge, ids, subset_bonds
    
    def sanitize(self, mol):
        """ 规范化分子，确保无效分子被处理 """
        try:
            # 获取分子的 SMILES 表示
            smiles = self.get_smiles(mol)
            # 从 SMILES 生成分子对象
            mol = self.get_mol(smiles)
        except Exception as e:
            # 捕获异常并返回 None
            return None
        return mol  # 返回经过规范化的分子对象

    def get_mol(self, smiles):
        """ 从 SMILES 字符串生成分子对象 """
        # 将 SMILES 转换为分子对象
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # 如果转换失败，则返回 None
            return None
        # 将分子中的芳香环进行凯库化处理
        Chem.Kekulize(mol)
        return mol  # 返回生成的分子对象

    def get_smiles(self, mol):
        """ 将分子对象转换为 SMILES 字符串 """
        # 使用 Chem 库将分子对象转换为 SMILES 字符串
        return Chem.MolToSmiles(mol, kekuleSmiles=True)

    def get_clique_mol(self, mol, atoms_ids):
        """ 获取给定原子 ID 的分子片段 """
        # 提取指定原子的 SMILES 片段表示
        smiles = Chem.MolFragmentToSmiles(mol, atoms_ids, kekuleSmiles=False) 
        # 从 SMILES 字符串生成新分子对象，不进行规范化处理
        new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
        # 复制并编辑生成的新分子对象
        new_mol = self.copy_edit_mol(new_mol).GetMol()
        # 对新分子对象进行规范化处理
        new_mol = self.sanitize(new_mol)  # 假设这不是 None
        return new_mol, smiles  # 返回新分子对象和其 SMILES 表示

    def copy_atom(self, atom):
        """ 复制原子并保持其属性 """
        # 创建一个新原子对象，并复制符号
        new_atom = Chem.Atom(atom.GetSymbol())
        new_atom.SetFormalCharge(atom.GetFormalCharge())  # 复制正式电荷
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())      # 复制原子映射编号
        return new_atom  # 返回复制的原子对象

    def copy_edit_mol(self, mol):
        """ 复制分子对象并保持其结构 """
        # 创建一个空的可编辑分子对象
        new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
        for atom in mol.GetAtoms():
            # 遍历原始分子的原子，并逐个复制到新分子中
            new_atom = self.copy_atom(atom)
            new_mol.AddAtom(new_atom)
        for bond in mol.GetBonds():
            # 遍历原始分子的键，并将其添加到新分子中
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            bt = bond.GetBondType()
            new_mol.AddBond(a1, a2, bt)
        return new_mol  # 返回复制后的分子对象

    def tree_decomp(self, mol):
        """ 从分子中提取树分解 """
        # 获取分子中的原子数量
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1:
            # 如果只有一个原子，返回单一子集和空边列表
            return [[0]], [], []
 
        cliques = []  # 初始化子集列表
        for bond in mol.GetBonds():
            # 遍历分子的每个键
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            if not bond.IsInRing():
                # 如果当前键不在环中，则将其原子索引作为子集添加
                cliques.append([a1, a2])
 
        # 获取分子中的所有基本环
        ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
        cliques.extend(ssr)  # 将基本环添加到子集列表中
 
        # 记录原子出现在哪些bond中
        nei_list = [[] for i in range(n_atoms)]
        for i in range(len(cliques)):
            for atom in cliques[i]:
                # 为每个子集中的原子更新其邻接列表
                nei_list[atom].append(i)
 
        # 合并交集大于 2 原子的环子集
        for i in range(len(cliques)):
            if len(cliques[i]) <= 2: continue  # 忽略小于等于2的子集
            for atom in cliques[i]:
                for j in nei_list[atom]:
                    if i >= j or len(cliques[j]) <= 2: continue  # 确保正确的顺序与条件
                    inter = set(cliques[i]) & set(cliques[j])  # 获取交集
                    if len(inter) > 2:
                        # 如果交集的原子数量大于 2，则合并子集
                        cliques[i].extend(cliques[j])
                        cliques[i] = list(set(cliques[i]))  # 去重
                        cliques[j] = []  # 清空 j 子集
 
        # 删除空子集
        cliques = [c for c in cliques if len(c) > 0]
        nei_list = [[] for i in range(n_atoms)]  # 重新初始化邻接列表
        for i in range(len(cliques)):
            for atom in cliques[i]:
                # 更新邻接列表
                nei_list[atom].append(i)
 
        # 构建边并添加孤立子集
        edges = defaultdict(int)  # 存储边及其权重
        for atom in range(n_atoms):
            if len(nei_list[atom]) <= 1:  # 忽略邻接子集数量小于等于1的原子
                continue
            cnei = nei_list[atom]  # 获取当前原子的邻接子集列表
            bonds = [c for c in cnei if len(cliques[c]) == 2]  # 边
            rings = [c for c in cnei if len(cliques[c]) > 4]  # 环
            if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2):
                # 添加孤立子集的条件
                cliques.append([atom])
                c2 = len(cliques) - 1  # 新子集的索引
                for c1 in cnei:
                    edges[(c1, c2)] = 1  # 记录边
            elif len(rings) > 2:
                # 如果有多个环，则添加孤立子集
                cliques.append([atom])
                c2 = len(cliques) - 1
                for c1 in cnei:
                    edges[(c1, c2)] = self.MST_MAX_WEIGHT - 1  # 设置权重
            else:
                # 处理其他情况
                for i in range(len(cnei)):
                    for j in range(i + 1, len(cnei)):
                        c1, c2 = cnei[i], cnei[j]
                        inter = set(cliques[c1]) & set(cliques[c2])  # 获取交集
                        if edges[(c1, c2)] < len(inter):
                            edges[(c1, c2)] = len(inter)  # 更新边权重
 
        # 格式化边的信息
        edges = [u + (self.MST_MAX_WEIGHT - v,) for u, v in edges.items()]
        if len(edges) == 0:
            # 如果没有边，则返回子集和空边列表
            return cliques, edges, []
 
        # 计算最大生成树
        row, col, data = zip(*edges)  # 提取边的信息
        n_clique = len(cliques)  # 子集数量
        # 使用边的信息构建一个稀疏矩阵表示的子集图
        clique_graph = csr_matrix((data, (row, col)), shape=(n_clique, n_clique))
        # 计算最大生成树
        junc_tree = minimum_spanning_tree(clique_graph)
        row, col = junc_tree.nonzero()  # 提取生成树的边
        edges = [(row[i], col[i]) for i in range(len(row))]  # 形成边的列表


        subset_bonds = {tuple(c): [] for c in cliques}
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            for clique in cliques:
                if a1 in clique and a2 in clique:
                    subset_bonds[tuple(clique)].append(bond.GetIdx())
        return (cliques, edges, subset_bonds)  # 返回分解后的子集和边列表






    