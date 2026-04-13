import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.classification import (
    BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score,
    BinaryMatthewsCorrCoef, BinaryAUROC,  # 新增
    BinarySpecificity, BinaryAveragePrecision  # 新增
)
from torchmetrics import Metric
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy

class BasicModel(pl.LightningModule):
    '''
    有mol_feat
    '''
    def __init__(self, num_classes=2, optimizer=None, loss_fn=None, lr_scheduler=None, lr=1e-3):
        """
        基础分类模型类，用于创建具体的分类模型。
        
        参数:
            num_classes (int): 分类任务的类别数量。
            optimizer (torch.optim.Optimizer, optional): 优化器，默认为 Adam。
            loss_fn (callable, optional): 损失函数，默认为交叉熵损失。
            lr_scheduler (torch.optim.lr_scheduler, optional): 学习率调度器，默认为 ReduceLROnPlateau。
            lr (float, optional): 学习率，默认为 1e-3。
        """
        super().__init__()
        self.num_classes = num_classes
        self.optimizer = optimizer if optimizer else Adam
        self.loss_fn = loss_fn if loss_fn else nn.CrossEntropyLoss(label_smoothing=0.1)
        self.lr_scheduler = lr_scheduler
        self.lr = lr

        # 定义评估指标
        base_metrics = nn.ModuleDict({
            "accuracy": BinaryAccuracy(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
            "f1": BinaryF1Score(),
            "mcc": BinaryMatthewsCorrCoef(),
            "auroc": BinaryAUROC(),
            "specificity": BinarySpecificity(),
            "auprc": BinaryAveragePrecision()

        })

        self.train_metrics = copy.deepcopy(base_metrics)
        self.val_metrics = copy.deepcopy(base_metrics)
        self.test_metrics = copy.deepcopy(base_metrics)

    def forward(self, x):
        """
        前向传播方法，需要在子类中实现。
        
        参数:
            x (torch.Tensor): 输入数据。
        
        返回:
            torch.Tensor: 模型的输出。
        """
        raise NotImplementedError("子类需要实现 forward 方法")

    def _shared_step(self, batch, stage):
        """
        共享的训练/验证/测试步骤。
        
        参数:
            batch (tuple): 包含输入数据和标签的元组。
            stage (str): 阶段名称（"train", "val", "test"）。
        
        返回:
            dict: 包含损失和日志的字典。
        """
        x, y, mol_feat, fps, smiles = batch
        y_hat, _ = self(x, mol_feat, fps)
        y_hat_prob = torch.softmax(y_hat, dim=1)[:, 1]
        y_hat_label = torch.argmax(y_hat, dim=1)
        
        loss = self.loss_fn(y_hat, y)

        # 更新评估指标
        metrics = getattr(self, f"{stage}_metrics")
        for name, metric in metrics.items():
            if name in ["auroc", "auprc"]:
                metric.update(y_hat_prob, y)
            else:
                metric.update(y_hat_label, y)

        # 记录损失
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def _shared_epoch_end(self, stage):
        """
        共享的训练/验证/测试阶段结束时的回调。
        
        参数:
            stage (str): 阶段名称（"train", "val", "test"）。
        """
        metrics = getattr(self, f"{stage}_metrics")
        for name, metric in metrics.items():
            value = metric.compute()
            self.log(f"{stage}_{name}", value, on_epoch=True, prog_bar=True, logger=True)
            metric.reset()

    def on_training_epoch_end(self):
        self._shared_epoch_end("train")

    def on_validation_epoch_end(self):
        self._shared_epoch_end("val")

    def on_test_epoch_end(self):
        self._shared_epoch_end("test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        预测步骤。
        
        参数:
            batch (torch.Tensor): 输入数据。
            batch_idx (int): 批次索引。
            dataloader_idx (int, optional): 数据加载器索引，默认为 0。
        
        返回:
            torch.Tensor: 模型的预测结果。
        """
        x, y, mol_feat, fps, smiles = batch
        y_hat, attn = self(x, mol_feat, fps)
        return {'y_hat': y_hat, 'y': y, 'attn': attn}

    def configure_optimizers(self):
        """
        配置优化器和学习率调度器。
        
        返回:
            dict: 包含优化器和学习率调度器的字典。
        """
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        if self.lr_scheduler:
            scheduler = self.lr_scheduler(optimizer, mode="min", factor=0.5, patience=5, verbose=True)
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_loss",
            }
        return optimizer

    def save_model(self, path):
        """
        保存模型。
        
        参数:
            path (str): 模型保存路径。
        """
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """
        加载模型。
        
        参数:
            path (str): 模型加载路径。
        """
        self.load_state_dict(torch.load(path))
