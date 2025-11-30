"""
Phase 3.3: KoELECTRA     NER  

  Vertex AI    KoELECTRA  
  (euphemism)  NER  .

 :
- : gs://k-euphemism-data/training/{train|validation|test}.jsonl
- BIO : B-EUPHEMISM, I-EUPHEMISM, O
-  : gs://k-euphemism-models/koelectra_euphemism_ner/
"""
import os
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Tuple
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    ElectraTokenizer,
    ElectraForTokenClassification,
    AdamW,
    get_linear_schedule_with_warmup
)

from sklearn.metrics import precision_recall_fscore_support, classification_report
from google.cloud import storage
import numpy as np

#  
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========================================
# 1.    
# ========================================

class EuphemismNERDataset(Dataset):
    """KoELECTRA NER  """

    def __init__(
        self,
        data_path: str,
        tokenizer: ElectraTokenizer,
        max_length: int = 128,
        label2id: Dict[str, int] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # BIO  
        if label2id is None:
            self.label2id = {
                'O': 0,
                'B-EUPHEMISM': 1,
                'I-EUPHEMISM': 2
            }
        else:
            self.label2id = label2id

        self.id2label = {v: k for k, v in self.label2id.items()}
        self.num_labels = len(self.label2id)

        #   (GCS  )
        self.samples = self._load_data(data_path)
        logger.info(f"Loaded {len(self.samples)} samples from {data_path}")

    def _load_data(self, path: str) -> List[Dict]:
        """JSONL   """
        samples = []

        # GCS  
        if path.startswith('gs://'):
            bucket_name = path.split('/')[2]
            blob_path = '/'.join(path.split('/')[3:])

            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)

            content = blob.download_as_text()
            lines = content.strip().split('\n')
        else:
            #  
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

        for line in lines:
            if line.strip():
                samples.append(json.loads(line))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        text = sample['text']
        tokens = sample.get('tokens', text.split())
        ner_tags = sample['ner_tags']

        #    
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        #  ID 
        label_ids = [self.label2id.get(tag, 0) for tag in ner_tags]

        #     
        if len(label_ids) < self.max_length:
            label_ids += [-100] * (self.max_length - len(label_ids))
        else:
            label_ids = label_ids[:self.max_length]

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }


# ========================================
# 2.    
# ========================================

class KoELECTRANERTrainer:
    """KoELECTRA NER  """

    def __init__(
        self,
        model_name: str = "monologg/koelectra-base-v3-discriminator",
        num_labels: int = 3,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        num_epochs: int = 10,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        output_dir: str = "./output",
        checkpoint_dir: str = "./checkpoints",
        tensorboard_dir: str = "./runs",
        device: str = None
    ):
        # 
        self.model_name = model_name
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm

        #  
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.tensorboard_dir = Path(tensorboard_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        #  
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        #    
        self.tokenizer = ElectraTokenizer.from_pretrained(self.model_name)
        self.model = ElectraForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        self.model.to(self.device)

        # TensorBoard 
        self.writer = SummaryWriter(self.tensorboard_dir)

        #    (   )
        self.optimizer = None
        self.scheduler = None

        #  
        self.best_f1 = 0.0
        self.global_step = 0

    def prepare_data_loaders(
        self,
        train_path: str,
        val_path: str,
        test_path: str = None,
        max_length: int = 128
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """ """

        #  
        train_dataset = EuphemismNERDataset(
            train_path,
            self.tokenizer,
            max_length=max_length
        )

        val_dataset = EuphemismNERDataset(
            val_path,
            self.tokenizer,
            max_length=max_length,
            label2id=train_dataset.label2id
        )

        test_dataset = None
        if test_path:
            test_dataset = EuphemismNERDataset(
                test_path,
                self.tokenizer,
                max_length=max_length,
                label2id=train_dataset.label2id
            )

        #  
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )

        test_loader = None
        if test_dataset:
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4
            )

        self.label2id = train_dataset.label2id
        self.id2label = train_dataset.id2label

        return train_loader, val_loader, test_loader

    def _init_optimizer_scheduler(self, train_loader):
        """   """

        # 
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate
        )

        # 
        total_steps = len(train_loader) * self.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """1  """
        self.model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            #    
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss

            # Backward pass
            loss.backward()

            #  
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )

            #  
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            self.global_step += 1

            # TensorBoard 
            if batch_idx % 50 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/learning_rate',
                                     self.scheduler.get_last_lr()[0],
                                     self.global_step)

                logger.info(
                    f"Epoch {epoch+1}/{self.num_epochs} | "
                    f"Step {batch_idx}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"LR: {self.scheduler.get_last_lr()[0]:.2e}"
                )

        avg_loss = total_loss / len(train_loader)
        return avg_loss

    def evaluate(self, val_loader: DataLoader, epoch: int = None) -> Dict:
        """  """
        self.model.eval()

        all_preds = []
        all_labels = []
        total_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                logits = outputs.logits

                total_loss += loss.item()

                #    
                predictions = torch.argmax(logits, dim=-1)

                # -100   ()
                mask = labels != -100

                preds_flat = predictions[mask].cpu().numpy()
                labels_flat = labels[mask].cpu().numpy()

                all_preds.extend(preds_flat)
                all_labels.extend(labels_flat)

        #  
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels,
            all_preds,
            average='weighted',
            zero_division=0
        )

        #  
        report = classification_report(
            all_labels,
            all_preds,
            target_names=list(self.id2label.values()),
            output_dict=True,
            zero_division=0
        )

        avg_loss = total_loss / len(val_loader)

        metrics = {
            'loss': avg_loss,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'classification_report': report
        }

        # TensorBoard 
        if epoch is not None:
            self.writer.add_scalar('val/loss', avg_loss, epoch)
            self.writer.add_scalar('val/precision', precision, epoch)
            self.writer.add_scalar('val/recall', recall, epoch)
            self.writer.add_scalar('val/f1', f1, epoch)

        logger.info(
            f"Validation | "
            f"Loss: {avg_loss:.4f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1:.4f}"
        )

        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """ """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'global_step': self.global_step,
            'label2id': self.label2id,
            'id2label': self.id2label
        }

        #   
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        #    
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        early_stopping_patience: int = 3
    ):
        """  """

        logger.info("=" * 60)
        logger.info("Starting KoELECTRA NER Training")
        logger.info("=" * 60)
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Number of labels: {self.num_labels}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Learning rate: {self.learning_rate}")
        logger.info(f"Number of epochs: {self.num_epochs}")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        logger.info("=" * 60)

        #    
        self._init_optimizer_scheduler(train_loader)

        # Early stopping
        patience_counter = 0

        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{self.num_epochs}")
            logger.info("-" * 60)

            # 
            train_loss = self.train_epoch(train_loader, epoch)
            logger.info(f"Average training loss: {train_loss:.4f}")

            # 
            val_metrics = self.evaluate(val_loader, epoch)

            #   
            is_best = False
            if val_metrics['f1'] > self.best_f1:
                self.best_f1 = val_metrics['f1']
                is_best = True
                patience_counter = 0
                logger.info(f"New best F1-Score: {self.best_f1:.4f}")
            else:
                patience_counter += 1

            #  
            self.save_checkpoint(epoch, val_metrics, is_best)

            # Early stopping 
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        logger.info("=" * 60)
        logger.info("Training completed")
        logger.info(f"Best F1-Score: {self.best_f1:.4f}")
        logger.info("=" * 60)

    def save_model_to_gcs(self, gcs_path: str):
        """  GCS """
        logger.info(f"Saving model to GCS: {gcs_path}")

        #   
        local_model_dir = self.output_dir / 'final_model'
        local_model_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(local_model_dir)
        self.tokenizer.save_pretrained(local_model_dir)

        #   
        label_config = {
            'label2id': self.label2id,
            'id2label': self.id2label
        }
        with open(local_model_dir / 'label_config.json', 'w', encoding='utf-8') as f:
            json.dump(label_config, f, ensure_ascii=False, indent=2)

        # GCS 
        if gcs_path.startswith('gs://'):
            bucket_name = gcs_path.split('/')[2]
            blob_prefix = '/'.join(gcs_path.split('/')[3:])

            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)

            #   
            for local_file in local_model_dir.glob('**/*'):
                if local_file.is_file():
                    relative_path = local_file.relative_to(local_model_dir)
                    blob_path = f"{blob_prefix}/{relative_path}"

                    blob = bucket.blob(blob_path)
                    blob.upload_from_filename(str(local_file))
                    logger.info(f"Uploaded: {blob_path}")

            logger.info(f"Model saved to {gcs_path}")
        else:
            logger.info(f"Model saved locally to {local_model_dir}")


# ========================================
# 3.   
# ========================================

def main():
    parser = argparse.ArgumentParser(
        description='Train KoELECTRA NER model for Korean euphemism detection'
    )

    #  
    parser.add_argument(
        '--train-data',
        type=str,
        default='gs://k-euphemism-data/training/train.jsonl',
        help='Training data path (GCS or local)'
    )
    parser.add_argument(
        '--val-data',
        type=str,
        default='gs://k-euphemism-data/training/validation.jsonl',
        help='Validation data path (GCS or local)'
    )
    parser.add_argument(
        '--test-data',
        type=str,
        default='gs://k-euphemism-data/training/test.jsonl',
        help='Test data path (GCS or local)'
    )

    #  
    parser.add_argument(
        '--model-name',
        type=str,
        default='monologg/koelectra-base-v3-discriminator',
        help='Pretrained model name'
    )
    parser.add_argument(
        '--num-labels',
        type=int,
        default=3,
        help='Number of NER labels (default: 3 for BIO scheme)'
    )

    #  
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--early-stopping-patience', type=int, default=3)

    #  
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Output directory for model and artifacts'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./checkpoints',
        help='Checkpoint directory'
    )
    parser.add_argument(
        '--tensorboard-dir',
        type=str,
        default='./runs',
        help='TensorBoard log directory'
    )
    parser.add_argument(
        '--gcs-model-path',
        type=str,
        default='gs://k-euphemism-models/koelectra_euphemism_ner',
        help='GCS path to save final model'
    )

    # 
    parser.add_argument('--device', type=str, default=None, help='cuda or cpu')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    #  
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    #  
    trainer = KoELECTRANERTrainer(
        model_name=args.model_name,
        num_labels=args.num_labels,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        tensorboard_dir=args.tensorboard_dir,
        device=args.device
    )

    #   
    train_loader, val_loader, test_loader = trainer.prepare_data_loaders(
        train_path=args.train_data,
        val_path=args.val_data,
        test_path=args.test_data,
        max_length=args.max_length
    )

    #  
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        early_stopping_patience=args.early_stopping_patience
    )

    #  
    if test_loader:
        logger.info("\n" + "=" * 60)
        logger.info("Final Test Evaluation")
        logger.info("=" * 60)
        test_metrics = trainer.evaluate(test_loader)

        logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
        logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
        logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
        logger.info(f"Test F1-Score: {test_metrics['f1']:.4f}")

        #   
        logger.info("\nClassification Report:")
        for label, metrics in test_metrics['classification_report'].items():
            if isinstance(metrics, dict):
                logger.info(
                    f"{label:20s} | "
                    f"Precision: {metrics['precision']:.4f} | "
                    f"Recall: {metrics['recall']:.4f} | "
                    f"F1: {metrics['f1-score']:.4f}"
                )

    #   GCS 
    trainer.save_model_to_gcs(args.gcs_model_path)

    logger.info("\n" + "=" * 60)
    logger.info("Training pipeline completed successfully")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
