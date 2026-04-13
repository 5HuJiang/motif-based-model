import torch
import pytorch_lightning as pl
from may22_dataprogress import GraphDataModule, data_pre
from model.may22_modeldiqt import AttentiveFP_DGL
from pytorch_lightning import Trainer
from MolTokenizer import MolTokenizer
import utils
import torch.nn.functional as F
import torch.nn as nn
import tensorboard
from sklearn.model_selection import KFold
import pandas as pd
from rdkit import RDLogger
from qqmessage import send2qq
import argparse
# 只关闭 kekulize 的 warning
RDLogger.DisableLog('rdApp.*')



def graph_train():   

    datamodule = GraphDataModule(train_path=train_path, val_path=val_path, test_path=test_path, feature_path=feature_path, tokenizer=tokenizer, batch_size=256, seed=seed, )
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    logger = pl.loggers.TensorBoardLogger(save_dir="tb_logs", name=f"3.31-{method}-{seed}-qt")
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=1,
        gradient_clip_val=1.0,
        max_epochs=800,
        logger=logger,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=20, mode="min"),
            pl.callbacks.ModelCheckpoint(
                dirpath=f'./checkpoints/3.31-{random}/{method}-{seed}-checkpoints-qt',
                filename= f'model-' + '{epoch}-{val_loss:.4f}',
                monitor="val_loss", mode="min", 
                save_top_k=1,
                every_n_epochs=1,
                )
        ]
    )
    model = AttentiveFP_DGL(
        method=method,
        lr=1e-4
        )
    trainer.fit(model, train_loader, val_loader)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练模型')
    parser.add_argument('--method', type=str, default='all', help='方法')
    parser.add_argument('--random', type=str, default='all', help='随机数')
    args = parser.parse_args()
    method = args.method
    random = args.random
 

    # 定义数据集路径
    train_path = 'data/DIQT/train.csv'
    val_path = 'data/DIQT/val.csv'
    test_path = 'data/DIQT/test.csv'

    feature_path = "data/motifs2features0825.parquet"
    token_path = "data/motifs_token_id.json"
    tokenizer = MolTokenizer(token_path)

    seeds = [1111, 2222, 3333, 4444, 5555]
    # seeds = [1, 2, 3, 4, 5]
    # seeds = [1, 11, 111, 1111, 11111]
    # seeds = [111, 222, 333, 444, 555]

    # 数据加载器
    
    for seed in seeds:
        # data_pre(seed)
        pl.seed_everything(seed)
        graph_train()

