"""
  

KoELECTRA    NER  .
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    ElectraForTokenClassification,
    ElectraTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import json
import os
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np


class EuphemismDataset(Dataset):
    """  NER """

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
            return_tensors='pt'
        )

        #   (  )
        label_padded = label + [-100] * (self.max_length - len(label))
        label_padded = label_padded[:self.max_length]

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label_padded, dtype=torch.long)
        }


class EuphemismTrainer:
    """    """

    def __init__(
        self,
        model_name: str = "monologg/koelectra-base-v3-discriminator",
        num_labels: int = 3,  # O, B-EUPHEMISM, I-EUPHEMISM
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        #    
        self.tokenizer = ElectraTokenizer.from_pretrained(model_name)
        self.model = ElectraForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.model.to(self.device)

        self.num_labels = num_labels

    def prepare_data(
        self,
        data_path: str,
        test_size: float = 0.2,
        batch_size: int = 16
    ) -> Tuple[DataLoader, DataLoader]:
        """
           DataLoader 

        Args:
            data_path:    (JSON)
            test_size:   
            batch_size:  

        Returns:
            train_loader, val_loader
        """
        print(f"\n : {data_path}")

        #  
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        #   
        texts = []
        labels = []

        for item in data:
            text = item['text']
            patterns = item.get('patterns', [])

            # BIO  
            label = self._create_bio_labels(text, patterns)

            texts.append(text)
            labels.append(label)

        # Train/Val 
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=42
        )

        print(f"Train samples: {len(train_texts)}")
        print(f"Val samples: {len(val_texts)}")

        #  
        train_dataset = EuphemismDataset(
            train_texts, train_labels, self.tokenizer
        )
        val_dataset = EuphemismDataset(
            val_texts, val_labels, self.tokenizer
        )

        # DataLoader 
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size
        )

        return train_loader, val_loader

    def _create_bio_labels(self, text: str, patterns: List[Dict]) -> List[int]:
        """
        BIO   

        Args:
            text:  
            patterns:    

        Returns:
            BIO   (0: O, 1: B-EUPHEMISM, 2: I-EUPHEMISM)
        """
        # 
        tokens = self.tokenizer.tokenize(text)
        labels = [0] * len(tokens)  # : O

        for pattern in patterns:
            #   
            pattern_text = pattern['text']
            pattern_tokens = self.tokenizer.tokenize(pattern_text)

            #   
            for i in range(len(tokens) - len(pattern_tokens) + 1):
                if tokens[i:i+len(pattern_tokens)] == pattern_tokens:
                    # B-EUPHEMISM
                    labels[i] = 1
                    # I-EUPHEMISM
                    for j in range(1, len(pattern_tokens)):
                        labels[i + j] = 2

        return labels

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        learning_rate: float = 2e-5,
        warmup_steps: int = 500,
        output_dir: str = "./models"
    ):
        """
         

        Args:
            train_loader:  DataLoader
            val_loader:  DataLoader
            epochs:  
            learning_rate: 
            warmup_steps: Warmup  
            output_dir:   
        """
        print("\n===   ===\n")

        # Optimizer  Scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        #  
        best_f1 = 0.0

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            # 
            train_loss = self._train_epoch(train_loader, optimizer, scheduler)
            print(f"Train Loss: {train_loss:.4f}")

            # 
            val_loss, val_f1 = self._validate(val_loader)
            print(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")

            #   
            if val_f1 > best_f1:
                best_f1 = val_f1
                self.save_model(output_dir)
                print(f"    (F1: {best_f1:.4f})")

            print()

        print(f" ! Best F1: {best_f1:.4f}")

    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler
    ) -> float:
        """  """
        self.model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc="Training"):
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

            loss = outputs.loss
            total_loss += loss.item()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        return total_loss / len(train_loader)

    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
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

                loss = outputs.loss
                total_loss += loss.item()

                # 
                preds = torch.argmax(outputs.logits, dim=-1)

                #  ( )
                for pred, label, mask in zip(
                    preds, labels, attention_mask
                ):
                    valid_length = mask.sum().item()
                    pred_valid = pred[:valid_length].cpu().numpy()
                    label_valid = label[:valid_length].cpu().numpy()

                    # -100 () 
                    mask_valid = label_valid != -100
                    all_preds.extend(pred_valid[mask_valid])
                    all_labels.extend(label_valid[mask_valid])

        # F1 Score 
        f1 = f1_score(all_labels, all_preds, average='macro')

        return total_loss / len(val_loader), f1

    def save_model(self, output_dir: str):
        """ """
        os.makedirs(output_dir, exist_ok=True)

        #  
        model_path = os.path.join(output_dir, "koelectra_euphemism.pt")
        torch.save(self.model.state_dict(), model_path)

        #  
        tokenizer_path = os.path.join(output_dir, "tokenizer")
        self.tokenizer.save_pretrained(tokenizer_path)

        print(f"  : {output_dir}")


def main():
    """  """

    # 
    config = {
        'data_path': 'data/annotations/labeled_data.json',
        'output_dir': './models',
        'epochs': 10,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'warmup_steps': 500,
        'test_size': 0.2
    }

    print("=== KoELECTRA      ===\n")
    print(":")
    for key, value in config.items():
        print(f"  {key}: {value}")

    #  
    trainer = EuphemismTrainer()

    #  
    train_loader, val_loader = trainer.prepare_data(
        config['data_path'],
        test_size=config['test_size'],
        batch_size=config['batch_size']
    )

    # 
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        warmup_steps=config['warmup_steps'],
        output_dir=config['output_dir']
    )


if __name__ == "__main__":
    main()
