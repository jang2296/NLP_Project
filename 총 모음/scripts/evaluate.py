"""
  

 KoELECTRA    .
"""

import torch
from transformers import ElectraForTokenClassification, ElectraTokenizer
import json
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import os


class EuphemismEvaluator:
    """    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str = None,
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        #  
        if tokenizer_path is None:
            tokenizer_path = os.path.dirname(model_path) + "/tokenizer"

        self.tokenizer = ElectraTokenizer.from_pretrained(
            "monologg/koelectra-base-v3-discriminator"
            if not os.path.exists(tokenizer_path)
            else tokenizer_path
        )

        self.model = ElectraForTokenClassification.from_pretrained(
            "monologg/koelectra-base-v3-discriminator",
            num_labels=3
        )

        #   
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        print(f"  : {model_path}")

        #  
        self.id2label = {
            0: 'O',
            1: 'B-EUPHEMISM',
            2: 'I-EUPHEMISM'
        }
        self.label2id = {v: k for k, v in self.id2label.items()}

    def evaluate(
        self,
        test_data_path: str,
        output_dir: str = "./results"
    ) -> Dict:
        """
           

        Args:
            test_data_path:    (JSON)
            output_dir:   

        Returns:
              
        """
        print(f"\n===   ===")
        print(f" : {test_data_path}\n")

        #   
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        print(f"  : {len(test_data)}")

        #     
        all_predictions = []
        all_labels = []

        for item in test_data:
            text = item['text']
            patterns = item.get('patterns', [])

            #   
            true_labels = self._create_bio_labels(text, patterns)

            # 
            pred_labels = self.predict(text)

            #   (  )
            min_len = min(len(true_labels), len(pred_labels))
            all_predictions.extend(pred_labels[:min_len])
            all_labels.extend(true_labels[:min_len])

        #  
        metrics = self._compute_metrics(all_predictions, all_labels)

        #  
        self._print_metrics(metrics)

        #  
        os.makedirs(output_dir, exist_ok=True)
        self._save_results(metrics, output_dir)

        #   
        self._plot_confusion_matrix(
            all_labels,
            all_predictions,
            output_dir
        )

        return metrics

    def predict(self, text: str) -> List[int]:
        """
          

        Args:
            text:  

        Returns:
              
        """
        # 
        encoding = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # 
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        #   
        valid_length = attention_mask.sum().item()
        pred_labels = predictions[0][:valid_length].cpu().numpy().tolist()

        return pred_labels

    def _create_bio_labels(self, text: str, patterns: List[Dict]) -> List[int]:
        """BIO   """
        tokens = self.tokenizer.tokenize(text)
        labels = [0] * len(tokens)  # : O

        for pattern in patterns:
            pattern_text = pattern['text']
            pattern_tokens = self.tokenizer.tokenize(pattern_text)

            #   
            for i in range(len(tokens) - len(pattern_tokens) + 1):
                if tokens[i:i+len(pattern_tokens)] == pattern_tokens:
                    labels[i] = 1  # B-EUPHEMISM
                    for j in range(1, len(pattern_tokens)):
                        labels[i + j] = 2  # I-EUPHEMISM

        return labels

    def _compute_metrics(
        self,
        predictions: List[int],
        labels: List[int]
    ) -> Dict:
        """ """

        #  
        precision, recall, f1, support = precision_recall_fscore_support(
            labels,
            predictions,
            average='macro',
            zero_division=0
        )

        #  
        class_report = classification_report(
            labels,
            predictions,
            target_names=list(self.id2label.values()),
            output_dict=True,
            zero_division=0
        )

        # Confusion Matrix
        cm = confusion_matrix(labels, predictions)

        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support,
            'class_report': class_report,
            'confusion_matrix': cm.tolist(),
            'total_samples': len(labels)
        }

        return metrics

    def _print_metrics(self, metrics: Dict):
        """ """
        print("\n===   ===\n")

        print(f" :")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  Total Samples: {metrics['total_samples']}")

        print("\n :")
        for label_name, label_metrics in metrics['class_report'].items():
            if isinstance(label_metrics, dict):
                print(f"\n  {label_name}:")
                print(f"    Precision: {label_metrics['precision']:.4f}")
                print(f"    Recall: {label_metrics['recall']:.4f}")
                print(f"    F1-Score: {label_metrics['f1-score']:.4f}")
                print(f"    Support: {label_metrics['support']}")

    def _save_results(self, metrics: Dict, output_dir: str):
        """ """

        # JSON 
        results_path = os.path.join(output_dir, 'evaluation_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            # numpy array list 
            metrics_serializable = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in metrics.items()
            }
            json.dump(metrics_serializable, f, ensure_ascii=False, indent=2)

        print(f"\n  : {results_path}")

    def _plot_confusion_matrix(
        self,
        true_labels: List[int],
        pred_labels: List[int],
        output_dir: str
    ):
        """  """

        cm = confusion_matrix(true_labels, pred_labels)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=list(self.id2label.values()),
            yticklabels=list(self.id2label.values())
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        # 
        plot_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  : {plot_path}")

        plt.close()

    def test_sample_predictions(self, test_texts: List[str]):
        """   """
        print("\n===    ===\n")

        for i, text in enumerate(test_texts, 1):
            print(f"{i}. : {text}")

            # 
            pred_labels = self.predict(text)
            tokens = self.tokenizer.tokenize(text)

            #   
            euphemisms = []
            current_euphemism = []
            current_label = None

            for token, label in zip(tokens, pred_labels):
                if label == 1:  # B-EUPHEMISM
                    if current_euphemism:
                        euphemisms.append(''.join(current_euphemism))
                    current_euphemism = [token.replace('##', '')]
                    current_label = 'EUPHEMISM'
                elif label == 2 and current_label == 'EUPHEMISM':  # I-EUPHEMISM
                    current_euphemism.append(token.replace('##', ''))
                else:
                    if current_euphemism:
                        euphemisms.append(''.join(current_euphemism))
                        current_euphemism = []
                        current_label = None

            if current_euphemism:
                euphemisms.append(''.join(current_euphemism))

            if euphemisms:
                print(f"     : {euphemisms}")
            else:
                print(f"      ")

            print()


def main():
    """  """

    # 
    config = {
        'model_path': './models/koelectra_euphemism.pt',
        'tokenizer_path': './models/tokenizer',
        'test_data_path': 'data/annotations/test_data.json',
        'output_dir': './results'
    }

    print("=== KoELECTRA      ===\n")

    #  
    evaluator = EuphemismEvaluator(
        model_path=config['model_path'],
        tokenizer_path=config['tokenizer_path']
    )

    #   
    if os.path.exists(config['test_data_path']):
        metrics = evaluator.evaluate(
            test_data_path=config['test_data_path'],
            output_dir=config['output_dir']
        )
    else:
        print(f"  : {config['test_data_path']}")
        print("  .\n")

    #   
    test_samples = [
        'S   .',
        '    .',
        'K   .',
        'L S   .',
        '  .',
    ]

    evaluator.test_sample_predictions(test_samples)


if __name__ == "__main__":
    main()
