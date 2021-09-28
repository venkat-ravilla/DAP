# utils 
from models.GuidedAttentionDAC import GuidedAttentionDAC
import torch
import os
import pandas as pd
import gc

# data
from torch.utils.data import Dataset, DataLoader

# models 
from transformers import AutoTokenizer

# training and evaluation
import wandb
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataset.dataset import DADataset
wandb.init()


class LightningModel(pl.LightningModule):
    
    def __init__(self, config):
        super(LightningModel, self).__init__()
        
        self.config = config        
        self.model = GuidedAttentionDAC(config, device=self.config['device'])
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        
    def forward(self, batch):
        # logits shape = [batch, 10, 43]
        logits  = self.model(batch) 
        indexes = torch.argmax(logits, dim=2)
        tags = indexes.apply_(lambda indx: self.index_label_map[indx])
        return tags
    
    def configure_optimizers(self):
        return optim.Adam(params=self.parameters(), lr=self.config['lr'])
    
    def train_dataloader(self):
        train_data = pd.read_csv(os.path.join(self.config['data_dir'], self.config['dataset'], self.config['dataset']+"_train.csv"), converters={'Text': eval,'DamslActTag': eval})
        train_dataset = DADataset(tokenizer=self.tokenizer, data=train_data, max_len=self.config['max_len'], text_field=self.config['text_field'], label_field=self.config['label_field'])
        self.index_label_map = train_dataset.index_label_map
        drop_last = True if len(train_dataset.text) % self.config['batch_size'] == 1 else False  # Drop last batch if it cointains a single sample (causes error)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config['num_workers'], drop_last=drop_last, pin_memory=True)
        return train_loader
    
    def training_step(self, batch, batch_idx):
        batch_size = batch['label'].size()[0]
        targets = batch['label']
        logits = self.model(batch, batch_size)
        logits_acc = logits.view(-1, 43)
        targets_acc = targets.view(-1)
        logits = logits[:,-1,:]
        targets = targets[:,-1]
        loss = F.cross_entropy(logits, targets)
        
        acc = accuracy_score(targets_acc.cpu(), logits_acc.argmax(dim=1).cpu())
        f1 = f1_score(targets.cpu(), logits_acc.argmax(dim=1).cpu(), average=self.config['average'])
        wandb.log({"loss":loss, "accuracy":acc, "f1_score":f1})
        return {"loss":loss, "accuracy":acc, "f1_score":f1}
    
    def val_dataloader(self):
        valid_data = pd.read_csv(os.path.join(self.config['data_dir'], self.config['dataset'], self.config['dataset']+"_validation.csv"), converters={'Text': eval,'DamslActTag': eval}) # valid has ~40k samples this is valid is same as test to run it quickely, test has ~16k samples
        valid_dataset = DADataset(tokenizer=self.tokenizer, data=valid_data, max_len=self.config['max_len'], text_field=self.config['text_field'], label_field=self.config['label_field'])
        drop_last = True if len(valid_dataset.text) % self.config['batch_size'] == 1 else False  # Drop last batch if it cointains a single sample (causes error)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config['num_workers'], drop_last=drop_last, pin_memory=True)
        return valid_loader
    
    def validation_step(self, batch, batch_idx):
        batch_size = batch['label'].size()[0]
        targets = batch['label']
        logits = self.model(batch, batch_size)
        logits_acc = logits.view(-1, 43)
        targets_acc = targets.view(-1)
        logits = logits[:,-1,:]
        targets = targets[:,-1]
        loss = F.cross_entropy(logits, targets)
        acc = accuracy_score(targets_acc.cpu(), logits_acc.argmax(dim=1).cpu())
        f1 = f1_score(targets_acc.cpu(), logits_acc.argmax(dim=1).cpu(), average=self.config['average'])
        precision = precision_score(targets_acc.cpu(), logits_acc.argmax(dim=1).cpu(), average=self.config['average'])
        recall = recall_score(targets_acc.cpu(), logits_acc.argmax(dim=1).cpu(), average=self.config['average'])
        return {"val_loss":loss, "val_accuracy":torch.tensor([acc]), "val_f1":torch.tensor([f1]), "val_precision":torch.tensor([precision]), "val_recall":torch.tensor([recall])}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_accuracy'] for x in outputs]).mean()
        avg_f1 = torch.stack([x['val_f1'] for x in outputs]).mean()
        avg_precision = torch.stack([x['val_precision'] for x in outputs]).mean()
        avg_recall = torch.stack([x['val_recall'] for x in outputs]).mean()
        wandb.log({"val_loss":avg_loss, "val_accuracy":avg_acc, "val_f1":avg_f1, "val_precision":avg_precision, "val_recall":avg_recall})
        return {"val_loss":avg_loss, "val_accuracy":avg_acc, "val_f1":avg_f1, "val_precision":avg_precision, "val_recall":avg_recall}
    
    def test_dataloader(self):
        test_data = pd.read_csv(os.path.join(self.config['data_dir'], self.config['dataset'], self.config['dataset']+"_test.csv"), converters={'Text': eval,'DamslActTag': eval})
        test_dataset = DADataset(tokenizer=self.tokenizer, data=test_data, max_len=self.config['max_len'], text_field=self.config['text_field'], label_field=self.config['label_field'])
        drop_last = True if len(test_dataset.text) % self.config['batch_size'] == 1 else False  # Drop last batch if it cointains a single sample (causes error)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config['num_workers'], drop_last=drop_last, pin_memory=True)
        return test_loader
    
    def test_step(self, batch, batch_idx):
        batch_size = batch['label'].size()[0]
        targets = batch['label']
        logits = self.model(batch, batch_size)
        logits_acc = logits.view(-1, 43)
        targets_acc = targets.view(-1)
        logits = logits[:,-1,:]
        targets = targets[:,-1]
        loss = F.cross_entropy(logits, targets)
        acc = accuracy_score(targets_acc.cpu(), logits_acc.argmax(dim=1).cpu())
        f1 = f1_score(targets_acc.cpu(), logits_acc.argmax(dim=1).cpu(), average=self.config['average'])
        precision = precision_score(targets_acc.cpu(), logits_acc.argmax(dim=1).cpu(), average=self.config['average'])
        recall = recall_score(targets_acc.cpu(), logits_acc.argmax(dim=1).cpu(), average=self.config['average'])
        return {"test_loss":loss, "test_precision":torch.tensor([precision]), "test_recall":torch.tensor([recall]), "test_accuracy":torch.tensor([acc]), "test_f1":torch.tensor([f1])}
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_accuracy'] for x in outputs]).mean()
        avg_f1 = torch.stack([x['test_f1'] for x in outputs]).mean()
        avg_precision = torch.stack([x['test_precision'] for x in outputs]).mean()
        avg_recall = torch.stack([x['test_recall'] for x in outputs]).mean()
        return {"test_loss":avg_loss, "test_precision":avg_precision, "test_recall":avg_recall, "test_acc":avg_acc, "test_f1":avg_f1}
