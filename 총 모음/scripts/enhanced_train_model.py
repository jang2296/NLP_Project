"""
 KoELECTRA   

 , , ,    
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    ElectraForTokenClassification,
    ElectraTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
import json
import os
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import numpy as np
import logging
from datetime import datetime
import wandb  # :  

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EuphemismDataset(Dataset):
    """  NER  ( )"""

    def __init__(
        self,
        texts: List[str],
        labels: List[List[int]],
        tokenizer: ElectraTokenizer,
        max_length: int = 128
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True  #   
        )

        #   (  )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        #  
        label_padded = label + [-100] * (self.max_length - len(label))
        label_padded = label_padded[:self.max_length]

        # CLS, SEP   -100
        label_padded[0] = -100  # CLS
        for i, token_id in enumerate(input_ids):
            if token_id == self.tokenizer.sep_token_id:
                label_padded[i] = -100  # SEP

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label_padded, dtype=torch.long)
        }


class EnhancedEuphemismTrainer:
    """     """

    def __init__(
        self,
        model_name: str = "monologg/koelectra-base-v3-discriminator",
        num_labels: int = 3,
        device: str = None,
        use_wandb: bool = False
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        #    
        self.tokenizer = ElectraTokenizer.from_pretrained(model_name)
        self.model = ElectraForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.model.to(self.device)

        self.num_labels = num_labels
        self.use_wandb = use_wandb

        #  
        self.label_names = ['O', 'B-EUPH', 'I-EUPH']

        if use_wandb:
            wandb.init(project="k-euphemism-detector", name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    def prepare_data(
        self,
        data_path: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        batch_size: int = 16,
        max_length: int = 128
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
           DataLoader  (Train/Val/Test )

        Args:
            data_path:    (JSON)
            test_size:   
            val_size:    (train )
            batch_size:  
            max_length:   

        Returns:
            train_loader, val_loader, test_loader
        """
        logger.info(f"\n : {data_path}")

        #  
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f" : {len(data)}")

        #   
        texts = []
        labels = []

        for item in data:
            text = item['text']
            patterns = item.get('patterns', [])

            # BIO  
            label = self._create_bio_labels(text, patterns, max_length)

            texts.append(text)
            labels.append(label)

        # Train/Test 
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=[sum(l) > 0 for l in labels]
        )

        # Train/Val 
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_val_texts, train_val_labels, test_size=val_size/(1-test_size), random_state=42
        )

        logger.info(f"Train samples: {len(train_texts)}")
        logger.info(f"Val samples: {len(val_texts)}")
        logger.info(f"Test samples: {len(test_texts)}")

        #  
        train_dataset = EuphemismDataset(train_texts, train_labels, self.tokenizer, max_length)
        val_dataset = EuphemismDataset(val_texts, val_labels, self.tokenizer, max_length)
        test_dataset = EuphemismDataset(test_texts, test_labels, self.tokenizer, max_length)

        # DataLoader 
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        return train_loader, val_loader, test_loader

    def _create_bio_labels(
        self,
        text: str,
        patterns: List[Dict],
        max_length: int
    ) -> List[int]:
        """BIO    ( )"""
        tokens = self.tokenizer.tokenize(text)
        labels = [0] * len(tokens)  # : O

        for pattern in patterns:
            pattern_text = pattern['text']
            pattern_start = pattern.get('start', text.find(pattern_text))

            if pattern_start == -1:
                continue

            #   
            token_start_idx = None
            token_end_idx = None

            current_pos = 0
            for idx, token in enumerate(tokens):
                token_text = token.replace('##', '')
                token_len = len(token_text)

                if current_pos <= pattern_start < current_pos + token_len:
                    token_start_idx = idx

                if token_start_idx is not None:
                    if current_pos + token_len > pattern_start + len(pattern_text):
                        token_end_idx = idx
                        break

                current_pos += token_len

            # BIO  
            if token_start_idx is not None:
                labels[token_start_idx] = 1  # B-EUPH

                if token_end_idx is not None:
                    for i in range(token_start_idx + 1, token_end_idx + 1):
                        labels[i] = 2  # I-EUPH

        #   
        return labels[:max_length]

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 1,
        output_dir: str = "./models",
        save_steps: int = 500,
        early_stopping_patience: int = 3
    ):
        """
          (  )

        Args:
            train_loader:  DataLoader
            val_loader:  DataLoader
            epochs:  
            learning_rate: 
            warmup_ratio: Warmup 
            weight_decay: Weight decay
            gradient_accumulation_steps: Gradient accumulation 
            output_dir:   
            save_steps:  
            early_stopping_patience: Early stopping patience
        """
        logger.info("\n===   ===\n")

        # Optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

        # Scheduler
        total_steps = len(train_loader) * epochs // gradient_accumulation_steps
        warmup_steps = int(total_steps * warmup_ratio)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        #  
        best_f1 = 0.0
        patience_counter = 0
        global_step = 0

        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            logger.info("-" * 60)

            # 
            train_loss = self._train_epoch(
                train_loader,
                optimizer,
                scheduler,
                gradient_accumulation_steps
            )
            logger.info(f"Train Loss: {train_loss:.4f}")

            # 
            val_loss, val_metrics = self._validate(val_loader)
            val_f1 = val_metrics['f1']
            val_precision = val_metrics['precision']
            val_recall = val_metrics['recall']

            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"Val F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")

            # WandB 
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_f1': val_f1,
                    'val_precision': val_precision,
                    'val_recall': val_recall
                })

            #   
            if val_f1 > best_f1:
                best_f1 = val_f1
                self.save_model(output_dir, f"best_model_f1_{best_f1:.4f}")
                logger.info(f"    (F1: {best_f1:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"\nEarly stopping triggered (patience: {early_stopping_patience})")
                break

        logger.info(f"\n ! Best F1: {best_f1:.4f}")

    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        gradient_accumulation_steps: int
    ) -> float:
        """   (Gradient Accumulation )"""
        self.model.train()
        total_loss = 0
        optimizer.zero_grad()

        progress_bar = tqdm(train_loader, desc="Training")

        for step, batch in enumerate(progress_bar):
            #  
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss / gradient_accumulation_steps
            total_loss += loss.item() * gradient_accumulation_steps

            # Backward
            loss.backward()

            # Gradient accumulation
            if (step + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            progress_bar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})

        return total_loss / len(train_loader)

    def _validate(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """ (, , F1 )"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()

                # 
                preds = torch.argmax(outputs.logits, dim=-1)

                #  ( )
                for pred, label, mask in zip(preds, labels, attention_mask):
                    valid_length = mask.sum().item()
                    pred_valid = pred[:valid_length].cpu().numpy()
                    label_valid = label[:valid_length].cpu().numpy()

                    # -100 () 
                    mask_valid = label_valid != -100
                    all_preds.extend(pred_valid[mask_valid])
                    all_labels.extend(label_valid[mask_valid])

        #  
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )

        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        return total_loss / len(val_loader), metrics

    def evaluate(self, test_loader: DataLoader) -> Dict:
        """  """
        logger.info("\n===   ===")

        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                preds = torch.argmax(outputs.logits, dim=-1)

                for pred, label, mask in zip(preds, labels, attention_mask):
                    valid_length = mask.sum().item()
                    pred_valid = pred[:valid_length].cpu().numpy()
                    label_valid = label[:valid_length].cpu().numpy()

                    mask_valid = label_valid != -100
                    all_preds.extend(pred_valid[mask_valid])
                    all_labels.extend(label_valid[mask_valid])

        #  
        report = classification_report(
            all_labels,
            all_preds,
            target_names=self.label_names,
            digits=4
        )

        logger.info("\n" + report)

        #  
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro'
        )

        results = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'report': report
        }

        return results

    def save_model(self, output_dir: str, model_name: str = "euphemism_model"):
        """ """
        os.makedirs(output_dir, exist_ok=True)

        #  state dict 
        model_path = os.path.join(output_dir, f"{model_name}.pt")
        torch.save(self.model.state_dict(), model_path)

        #    (HuggingFace )
        model_hf_path = os.path.join(output_dir, model_name)
        self.model.save_pretrained(model_hf_path)
        self.tokenizer.save_pretrained(model_hf_path)

        logger.info(f"  : {output_dir}/{model_name}")


def main():
    """  """
    # 
    config = {
        'data_path': './data/processed/training_dataset.json',
        'output_dir': './models',
        'epochs': 15,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'max_length': 128,
        'gradient_accumulation_steps': 2,
        'early_stopping_patience': 3,
        'use_wandb': False  # WandB  
    }

    print("="*70)
    print("KoELECTRA     ")
    print("="*70)
    print("\n:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    #  
    trainer = EnhancedEuphemismTrainer(use_wandb=config['use_wandb'])

    #  
    logger.info("\n  ...")
    train_loader, val_loader, test_loader = trainer.prepare_data(
        config['data_path'],
        batch_size=config['batch_size'],
        max_length=config['max_length']
    )

    # 
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        warmup_ratio=config['warmup_ratio'],
        weight_decay=config['weight_decay'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        output_dir=config['output_dir'],
        early_stopping_patience=config['early_stopping_patience']
    )

    #  
    results = trainer.evaluate(test_loader)

    print("\n" + "="*70)
    print("  ")
    print("="*70)
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1']:.4f}")

    print("\n !")


if __name__ == "__main__":
    main()
