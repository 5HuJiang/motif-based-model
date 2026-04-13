import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from dgl.nn import GATv2Conv
from .basicmodel import BasicModel
from dgl import readout_nodes


class NodeEncoder(nn.Module):
    """节点/边编码模块：将原始特征映射为固定维度嵌入"""
    def __init__(self, node_size, input_size):
        super().__init__()
        self.node_encoder = nn.Linear(node_size, input_size)
        
    def forward(self, g):
        # 编码节点特征
        g.ndata['h'] = F.leaky_relu(self.node_encoder(g.ndata['feat']))
        return g



class MoleculeEmbedding(nn.Module):
    def __init__(self, hidden_size=768, num_heads=8, num_timesteps=3, dropout=0.3):
        super().__init__()
        self.num_timesteps = num_timesteps

        self.gat = nn.ModuleList(
            [
                GATv2Conv(
                    in_feats=hidden_size,
                    out_feats=hidden_size,
                    num_heads=num_heads,
                    allow_zero_in_degree=True,
                    feat_drop=dropout,
                    attn_drop=dropout,
                    
                ) for _ in range(num_timesteps)
            ]
        )

        self.gru = nn.ModuleList(
            [
                nn.GRUCell(hidden_size, hidden_size) for _ in range(num_timesteps)                
            ])

    def add_virtual_nodes_and_edges(self, batch_g, mol_feat):
        device = batch_g.device
        g_list   = dgl.unbatch(batch_g)
        new_graphs = []
        

        for i, g in enumerate(g_list):
            num_nodes = g.num_nodes()
             # 1) 创建新节点特征（长度 = 原节点数 + 1）
            h_old = g.ndata['h']                           # [num_nodes, h_dim]

            h_new = mol_feat[i].unsqueeze(0)                     # [1, h_dim]
            h_all = torch.cat([h_old, h_new], dim=0)       # [num_nodes+1, h_dim]
        

            # 2) 创建 is_virtual 标志
            is_virtual = torch.zeros(num_nodes + 1, dtype=torch.bool, device=device)
            is_virtual[-1] = True                          # 最后一个节点是虚拟节点

            # 3) 一起赋值
            g.add_nodes(1)                                 # 先加 1 个空节点
            g.ndata['h']         = h_all
            g.ndata['is_virtual'] = is_virtual

            atom_ids = torch.arange(num_nodes, device=device)
            virtual_id = torch.tensor([num_nodes], device=device)
            src = atom_ids
            dst = virtual_id.repeat(num_nodes)

            g.add_edges(src, dst)
            attn_mask = torch.zeros(g.num_edges(), dtype=torch.bool, device=device)
            virtual_edge_ids = torch.arange(g.num_edges() - num_nodes, g.num_edges(), device=device)
            attn_mask[virtual_edge_ids] = True
            g.edata['attn_mask'] = attn_mask
        
            new_graphs.append(g)

        return dgl.batch(new_graphs)


    def forward(self, g, mol_feat, Global=True, gru=True):
        if Global is True:
            g = self.add_virtual_nodes_and_edges(g, mol_feat)

            h = g.ndata['h']  # 全部节点（含虚拟节点）特征
            mol_emb = mol_feat
            all_edge_attns = []
            for i in range(self.num_timesteps):
            # GAT 前向传播，包含 attention 权重
                h, attn_weights = self.gat[i](g, h, get_attention=True)
                h = h.mean(dim=1)  # 取平均（多头）
                
                batch_num_nodes = g.batch_num_nodes()
                virtual_nids = torch.cumsum(batch_num_nodes, dim=0) - 1

                out_virtual = h[virtual_nids]

                if gru is True:
                    mol_emb = F.relu(self.gru[i](out_virtual, mol_emb))
                else:
                    mol_emb = out_virtual

                # 提取用于解释的原子→虚拟 attention 权重（平均多头）
                attn_per_edge = attn_weights.mean(dim=1)
                edge_attn = attn_per_edge[g.edata['attn_mask']]
                all_edge_attns.append(edge_attn)

            # 返回嵌入和注意力（用于解释）
            return mol_emb, all_edge_attns
        else:
            emb = g.ndata['h']
            all_edge_attns = []
            for i in range(self.num_timesteps):
            # GAT 前向传播，包含 attention 权重
                h, attn_weights = self.gat[i](g, emb, get_attention=True)
                h = h.mean(dim=1)  # 取平均（多头）
            
                emb = F.relu(self.gru[i](h, emb))

            g.ndata['h'] = emb
            mol_emb = readout_nodes(g, 'h', op='mean')
            return mol_emb, all_edge_attns


class CrossAttnLayer(nn.Module):
    def __init__(self, d_model=768, n_heads=8, drop=0.3):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm  = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(d_model*4, d_model)
        )

    def forward(self, q, kv):
        # q: (B, L_q, d)
        # kv: (B, L_kv, d)
        out, _ = self.cross_attn(q, kv, kv)
        out = self.norm(out + q)          # 残差
        out = self.norm(self.ffn(out) + out)
        return out
    
class PredictionHead(nn.Module):
    """预测模块：全连接层执行分类/回归"""
    def __init__(self, input_size, hidden_size, out_size, num_layers, dropout=0.3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_size if i == 0 else hidden_size // (2**i), 
                          hidden_size // (2**(i+1))),
                nn.BatchNorm1d(hidden_size // (2**(i+1))),
                nn.LeakyReLU(),
                nn.Dropout(dropout)
            ])
        layers.append(nn.Linear(hidden_size // (2**num_layers), out_size))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)

class TransformerPre(nn.Module):
    def __init__(self, d_model=768, nhead=4, num_encoder_layers=2, dropout=0.3):
        super().__init__()
        self.liner1 = nn.Linear(d_model, nhead*64)
        self.liner2 = nn.Linear(nhead*64, 768)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=nhead*64,
            nhead=nhead,
            dim_feedforward=nhead*64,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )
    
    def forward(self, x):
        x = self.liner1(x)
        x = x.unsqueeze(1)
        out = self.transformer_encoder(x)
        out = out[:, 0, :]
        out = self.liner2(out)
        return out

class AttentiveFP_DGL(BasicModel):
    """配合V5datamodule"""
    def __init__(
            self, 
            method="all", # 'all', 'concat', 'only-fp', 'no-fp'
            input_size=768,    # 原子特征维度
            hidden_size=768, 
            num_timesteps=2,   # 分子嵌入迭代次数
            num_gat_heads=6,
            out_size=2,         # 输出维度（分类/回归）
            gat_drop=0.40,
            pre_drop=0.45,
            pre_layers=3,
            trans_drop=0.25,
            trans_layers=3,
            trans_heads=8,
            cross_heads=12,
            cross_drop=0.20,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.method = method
        self.fusion_layer = nn.Linear(768*2, 768)
        self.liner = nn.Linear(133, 768)
        # 1. 节点/边编码模块
        self.encoder = NodeEncoder(
            node_size=input_size,
            input_size=hidden_size,
            
        )
        
        # 3. 分子嵌入模块（虚拟节点 + GAT + GRU）
        self.mol_embedding = MoleculeEmbedding(
            hidden_size=hidden_size,
            num_timesteps=num_timesteps,
            num_heads=num_gat_heads,
            dropout=gat_drop,
        )

        self.transencoder = TransformerPre(
            d_model=3239,
            nhead=trans_heads,
            num_encoder_layers=trans_layers,
            dropout=trans_drop,
        )

        self.cross_attn = CrossAttnLayer(
            d_model=hidden_size,
            n_heads=cross_heads,
            drop=cross_drop,
        )

        # 4. 预测模块
        self.predictor = PredictionHead(
            input_size=hidden_size,
            hidden_size=hidden_size,
            out_size=out_size,
            num_layers=pre_layers,
            dropout=pre_drop
        )
        
    def forward(self, g, mol_feat, fps):
        if self.method == 'all':
            # 步骤1：编码节点和边特征
            g = self.encoder(g)
            
            # 步骤3：生成分子嵌入（带注意力权重）
            mol_emb, attn_weights = self.mol_embedding(g, mol_feat)

            fps = self.transencoder(fps)
            
            mol_emb = self.cross_attn(mol_emb, fps)

            # 步骤4：预测输出
            out = self.predictor(mol_emb)

        elif self.method == 'concat':
            g = self.encoder(g)
            mol_emb, attn_weights = self.mol_embedding(g, mol_feat)
            fps = self.transencoder(fps)
            mol_emb = self.fusion_layer(torch.cat([mol_emb, fps], dim=-1))
            out = self.predictor(mol_emb)

        elif self.method == 'no-global':
            g = self.encoder(g)
            
            mol_emb, attn_weights = self.mol_embedding(g, mol_feat, Global=False)

            fps = self.transencoder(fps)
            
            mol_emb = self.cross_attn(mol_emb, fps)

            out = self.predictor(mol_emb)

        elif self.method == 'atom':
            h = self.liner(g.ndata['feat'])
            g.ndata['feat'] = h
            mol_feat = dgl.readout_nodes(g, 'feat', op='mean')
            g = self.encoder(g)
            
            mol_emb, attn_weights = self.mol_embedding(g, mol_feat)

            fps = self.transencoder(fps)
            
            mol_emb = self.cross_attn(mol_emb, fps)

            out = self.predictor(mol_emb)   

        elif self.method == 'motif':
            h = self.liner(g.ndata['feat'])
            g.ndata['feat'] = h
            mol_feat = dgl.readout_nodes(g, 'feat', op='mean')
            g = self.encoder(g)
            
            mol_emb, attn_weights = self.mol_embedding(g, mol_feat)

            fps = self.transencoder(fps)
            
            mol_emb = self.cross_attn(mol_emb, fps)

            out = self.predictor(mol_emb) 
              
        elif self.method == 'no-gru':
            g = self.encoder(g)
            
            mol_emb, attn_weights = self.mol_embedding(g, mol_feat, gru=False)

            fps = self.transencoder(fps)
            
            mol_emb = self.cross_attn(mol_emb, fps)

            out = self.predictor(mol_emb)
        
        elif self.method == 'molformer':
            fps = self.transencoder(fps)
            mol_emb = mol_feat
            out = self.predictor(mol_emb)
            attn_weights = torch.zeros_like(mol_feat)
            
        elif self.method == 'no-fp':
            g = self.encoder(g)
            mol_emb, attn_weights = self.mol_embedding(g, mol_feat)
            out = self.predictor(mol_emb)

        elif self.method == 'only-fp':
            fps = self.transencoder(fps)
            out = self.predictor(fps)
            attn_weights = torch.zeros_like(fps)
        return out, attn_weights


def map_attn_to_motifs(batch_g, attn_weights, smiles_list, tokenizer):
    """
    返回：
        list[dict]  长度 = batch_size
        每个 dict 固定字段：
            'smiles' : str
            'layers_0': {'motifs': List[str], 'weights': np.ndarray}
            'layers_1': {'motifs': List[str], 'weights': np.ndarray}
            ...
    """
    device = batch_g.device
    g_list   = dgl.unbatch(batch_g)
    bsz      = len(g_list)
    batch_num_nodes = [g.num_nodes()-1 for g in g_list]
    cum_nodes = [0] + torch.cumsum(torch.tensor(batch_num_nodes), 0).tolist()

    # ---- 构造 virtual_edges （同旧逻辑） ----
    virtual_edges = []
    for i, g in enumerate(g_list):
        offset     = cum_nodes[i]
        virtual_id = g.num_nodes() - 1
        src_local, _ = g.in_edges(virtual_id, form='uv')
        src_global = (src_local + offset).cpu().numpy()
        virtual_edges.append(src_global)

    # ---- 预先取出每个图里各节点的 motif_token_id 和对应 token 名 ----
    all_motif_ids, all_motif_tokens = [], []   # list[np.array(str)]
    for g in g_list:
        ids = g.ndata['motif_id'][:-1].cpu().numpy()   # 去掉虚拟节点
        all_motif_ids.append(ids)
        all_motif_tokens.append(np.array([tokenizer.id_to_token[mid] for mid in ids]))

    # ---- 按层、按图收集注意力 ----
    results = []
    for i in range(bsz):
        results.append({'smiles': smiles_list[i]})

    for layer_idx, attn in enumerate(attn_weights):
        attn = attn.squeeze(-1).cpu().numpy()   # [E]
        start = 0
        for i, g in enumerate(g_list):
            n_edge = len(virtual_edges[i])
            scores = attn[start:start+n_edge]   # 该图该层所有原子→虚拟的注意力
            start += n_edge

            # global → local 节点索引
            local_idx = virtual_edges[i] - cum_nodes[i]
            # 取出对应 motif 名字
            tok_names = all_motif_tokens[i][local_idx]   # List[str]
            # 构造该层字段名
            layer_key = f'layers_{layer_idx}'
            results[i][layer_key] = {
                'motifs': tok_names,
                'weights': scores
            }
    return results

def extract_attention_weights(model, dataloader, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    运行模型并提取 Attention 权重用于解释
    
    Args:
        model: 训练好的 AttentiveFP_DGL 模型
        dataloader: 数据加载器 (例如 datamodule.test_dataloader())
        tokenizer: 包含 id_to_token 的 tokenizer 对象
        device: 运行设备
        
    Returns:
        dict: {smiles: {'layer_0': {'motifs': [...], 'weights': [...]}, ...}}
    """
    model.eval()
    model.to(device)
    
    results = {}
    
    print(f"开始提取 Attention 权重，共 {len(dataloader)} 个 batch...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # 1. 解包数据 (根据你的 collate_fn 定义)
            g, labels, mol_feat, fps, smiles_list = batch
            
            g = g.to(device)
            mol_feat = mol_feat.to(device)
            fps = fps.to(device)
            
            # 2. 获取 batch 中每个图的原子数量
            # 这一步很关键：用于将模型输出的平铺(flattened)权重切分回单个分子
            # 注意：这里的 g 是原始图，尚未添加虚拟节点，节点数即为原子数
            batch_num_nodes = g.batch_num_nodes().tolist()
            
            # 3. 获取所有节点的 motif_id (在 Dataset 中已存入 g.ndata)
            all_motif_ids = g.ndata['motif_id']
            
            # 4. 模型前向传播
            # model 返回: out, attn_weights
            # attn_weights 是一个列表，包含 num_timesteps 个张量
            # 每个张量对应 batch 中所有 "原子->虚拟节点" 的边权重
            _, all_layer_attns = model(g, mol_feat, fps)
            
            # 5. 数据切分：将 Batch 数据还原为单个分子数据
            
            # 5.1 切分 Motif IDs
            split_motif_ids = torch.split(all_motif_ids, batch_num_nodes)
            
            # 5.2 切分每一层的 Attention 权重
            # all_layer_attns[layer_idx] 的形状是 [total_atoms_in_batch, 1]
            split_layer_attns_per_layer = []
            for layer_attn in all_layer_attns:
                split_attn = torch.split(layer_attn, batch_num_nodes)
                split_layer_attns_per_layer.append(split_attn)
            
            # 6. 组装结果字典
            for i, smi in enumerate(smiles_list):
                # 如果数据集中有重复 SMILES，这里会覆盖，通常这是预期的
                if smi not in results:
                    results[smi] = {}
                
                # 获取当前分子的 motif tokens
                # 假设 tokenizer.id_to_token 支持索引访问
                current_motif_ids = split_motif_ids[i].cpu().numpy()
                motifs = [tokenizer.id_to_token[mid] for mid in current_motif_ids]
                
                # 遍历每一层 (timestep)
                for layer_idx, layer_split in enumerate(split_layer_attns_per_layer):
                    layer_key = f'layer_{layer_idx}'
                    
                    # 获取当前分子在当前层的权重
                    # layer_split[i] 是对应第 i 个分子的权重张量
                    weights = layer_split[i].flatten().cpu().numpy().tolist()
                    
                    # 写入字典
                    if layer_key not in results[smi]:
                        results[smi][layer_key] = {
                            'motifs': motifs,
                            'weights': weights
                        }
    
    print("提取完成！")
    return results