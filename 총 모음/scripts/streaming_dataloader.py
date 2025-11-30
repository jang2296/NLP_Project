"""
    
- ijson   JSON    
-   BIO   
-    
"""

import ijson
import torch
from torch.utils.data import IterableDataset
from transformers import ElectraTokenizerFast
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamingEuphemismDataset(IterableDataset):
    """    """

    def __init__(
        self,
        json_path: str,
        tokenizer: ElectraTokenizerFast,
        max_length: int = 128,
        skip_no_patterns: bool = True
    ):
        self.json_path = json_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.skip_no_patterns = skip_no_patterns

        # BIO  
        self.label_map = {
            'O': 0,      # Outside ( )
            'B-EUPH': 1, # Begin (  )
            'I-EUPH': 2  # Inside (  )
        }

    def __iter__(self):
        """   """
        with open(self.json_path, 'rb') as f:
            items = ijson.items(f, 'item')

            for idx, item in enumerate(items):
                #     
                if self.skip_no_patterns and not item.get('patterns'):
                    continue

                # BIO  
                try:
                    encoded = self._encode_item(item)
                    if encoded:
                        yield encoded
                except Exception as e:
                    logger.warning(f"Error encoding item {idx}: {e}")
                    continue

    def _encode_item(self, item: Dict) -> Dict:
        """ BIO   """
        text = item['text']
        patterns = item.get('patterns', [])

        # 
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )

        # BIO   ( O )
        labels = [self.label_map['O']] * self.max_length

        # offset_mapping     
        offset_mapping = encoding['offset_mapping'][0].tolist()

        #    BIO  
        for pattern in patterns:
            start = pattern['start']
            end = pattern['end']

            #    
            is_first_token = True

            for idx, (token_start, token_end) in enumerate(offset_mapping):
                # [CLS], [SEP], [PAD]  
                if token_start == token_end:
                    continue

                #      
                if token_start >= start and token_end <= end:
                    if is_first_token:
                        labels[idx] = self.label_map['B-EUPH']
                        is_first_token = False
                    else:
                        labels[idx] = self.label_map['I-EUPH']
                elif token_start < end and token_end > start:
                    #   
                    if is_first_token:
                        labels[idx] = self.label_map['B-EUPH']
                        is_first_token = False
                    else:
                        labels[idx] = self.label_map['I-EUPH']

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


def create_streaming_dataloader(
    json_path: str,
    tokenizer: ElectraTokenizerFast,
    batch_size: int = 2,
    max_length: int = 128,
    num_workers: int = 0
) -> torch.utils.data.DataLoader:
    """  """

    dataset = StreamingEuphemismDataset(
        json_path=json_path,
        tokenizer=tokenizer,
        max_length=max_length
    )

    def collate_fn(batch):
        """  """
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return dataloader


def get_dataset_stats(json_path: str) -> Dict:
    """   ( )"""

    total_items = 0
    items_with_patterns = 0
    total_patterns = 0
    pattern_types = {}

    with open(json_path, 'rb') as f:
        items = ijson.items(f, 'item')

        for item in items:
            total_items += 1

            patterns = item.get('patterns', [])
            if patterns:
                items_with_patterns += 1
                total_patterns += len(patterns)

                for pattern in patterns:
                    ptype = pattern.get('type', 'unknown')
                    pattern_types[ptype] = pattern_types.get(ptype, 0) + 1

            #       
            if total_items % 10000 == 0:
                logger.info(f" : {total_items:,}...")

    return {
        'total_items': total_items,
        'items_with_patterns': items_with_patterns,
        'total_patterns': total_patterns,
        'pattern_types': pattern_types,
        'pattern_ratio': items_with_patterns / total_items if total_items > 0 else 0
    }


if __name__ == "__main__":
    #  
    from transformers import ElectraTokenizerFast

    print("=" * 80)
    print("   ")
    print("=" * 80)

    #  
    print("\n[STATS]    ...")
    stats = get_dataset_stats('data/processed/final_training_dataset.json')

    print(f"\n  :")
    print(f"  -   : {stats['total_items']:,}")
    print(f"  -   : {stats['items_with_patterns']:,} ({stats['pattern_ratio']*100:.1f}%)")
    print(f"  -   : {stats['total_patterns']:,}")
    print(f"\n    :")
    for ptype, count in sorted(stats['pattern_types'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    - {ptype}: {count:,}")

    #  
    print("\n[BATCH]   ...")
    tokenizer = ElectraTokenizerFast.from_pretrained("monologg/koelectra-base-v3-discriminator")

    dataloader = create_streaming_dataloader(
        json_path='data/processed/final_training_dataset.json',
        tokenizer=tokenizer,
        batch_size=2,
        max_length=128
    )

    print("\n    :")
    batch = next(iter(dataloader))

    print(f"  - input_ids shape: {batch['input_ids'].shape}")
    print(f"  - attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  - labels shape: {batch['labels'].shape}")
    print(f"  -    : {batch['labels'][0][:20]}...")
    print(f"  - B-EUPH  : {(batch['labels'] == 1).sum().item()}")
    print(f"  - I-EUPH  : {(batch['labels'] == 2).sum().item()}")

    print("\n[OK]     !")
