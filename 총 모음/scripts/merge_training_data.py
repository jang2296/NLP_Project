"""
         
"""

import json
import random
from pathlib import Path
from typing import List, Dict
from collections import Counter


def load_json(path: str) -> List[Dict]:
    """JSON  """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: List[Dict], path: str):
    """JSON  """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def validate_sample(sample: Dict) -> bool:
    """  """
    required_fields = ['text', 'patterns', 'has_euphemism', 'char_labels']

    for field in required_fields:
        if field not in sample:
            return False

    if not sample['text'] or len(sample['text']) < 5:
        return False

    if not sample['patterns']:
        return False

    # char_labels  
    if len(sample['char_labels']) != len(sample['text']):
        return False

    return True


def get_sample_type(sample: Dict) -> str:
    """  """
    if sample['patterns']:
        return sample['patterns'][0].get('type', 'unknown')
    return 'unknown'


def merge_datasets(
    news_path: str,
    namuwiki_path: str,
    output_path: str,
    train_ratio: float = 0.8
) -> Dict:
    """ """

    #  
    print("  ...")
    news_data = load_json(news_path)
    namuwiki_data = load_json(namuwiki_path)

    print(f"   : {len(news_data)}")
    print(f"   : {len(namuwiki_data)}")

    #  
    print("\n  ...")
    valid_news = [s for s in news_data if validate_sample(s)]
    valid_namuwiki = [s for s in namuwiki_data if validate_sample(s)]

    print(f"    : {len(valid_news)}")
    print(f"    : {len(valid_namuwiki)}")

    #   
    for sample in valid_news:
        sample['source'] = 'synthetic_news'
        sample['domain'] = 'formal'  #  

    for sample in valid_namuwiki:
        sample['source'] = 'namuwiki'
        sample['domain'] = 'informal'  #  

    # 
    all_data = valid_news + valid_namuwiki
    random.shuffle(all_data)

    print(f"\n  : {len(all_data)}")

    #  
    type_counts = Counter()
    domain_counts = Counter()
    source_counts = Counter()

    for sample in all_data:
        type_counts[get_sample_type(sample)] += 1
        domain_counts[sample.get('domain', 'unknown')] += 1
        source_counts[sample.get('source', 'unknown')] += 1

    # Train/Test 
    split_idx = int(len(all_data) * train_ratio)
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]

    # 
    output_dir = Path(output_path).parent

    #  
    save_json(all_data, output_path)

    # Train/Test  
    train_path = output_dir / 'combined_train.json'
    test_path = output_dir / 'combined_test.json'

    save_json(train_data, str(train_path))
    save_json(test_data, str(test_path))

    # JSONL   (  )
    train_jsonl_path = output_dir / 'combined_train.jsonl'
    test_jsonl_path = output_dir / 'combined_test.jsonl'

    with open(train_jsonl_path, 'w', encoding='utf-8') as f:
        for sample in train_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    with open(test_jsonl_path, 'w', encoding='utf-8') as f:
        for sample in test_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    #  
    stats = {
        'total': len(all_data),
        'train': len(train_data),
        'test': len(test_data),
        'by_source': dict(source_counts),
        'by_domain': dict(domain_counts),
        'by_type': dict(type_counts)
    }

    print(f"\n===   ===")
    print(f" : {stats['total']}")
    print(f" : {stats['train']} ({train_ratio*100:.0f}%)")
    print(f" : {stats['test']} ({(1-train_ratio)*100:.0f}%)")

    print(f"\n===   ===")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {source}: {count} ({count/len(all_data)*100:.1f}%)")

    print(f"\n===   ===")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"  {domain}: {count} ({count/len(all_data)*100:.1f}%)")

    print(f"\n===   ( 15) ===")
    for type_name, count in type_counts.most_common(15):
        print(f"  {type_name}: {count} ({count/len(all_data)*100:.1f}%)")

    #  
    stats_path = output_dir / 'combined_statistics.json'
    save_json(stats, str(stats_path))

    print(f"\n===   ===")
    print(f"   : {output_path}")
    print(f"   : {train_path}")
    print(f"   : {test_path}")
    print(f"   JSONL: {train_jsonl_path}")
    print(f"   JSONL: {test_jsonl_path}")
    print(f"  : {stats_path}")

    return stats


if __name__ == '__main__':
    news_path = '/mnt/c/claude_project/NLP/data/processed/train.json'
    namuwiki_path = '/mnt/c/claude_project/NLP/data/processed/namuwiki_training_data.json'
    output_path = '/mnt/c/claude_project/NLP/data/processed/combined_all_data.json'

    stats = merge_datasets(news_path, namuwiki_path, output_path)

    #  
    print("\n===   ===")
    all_data = load_json(output_path)

    #   
    news_samples = [s for s in all_data if s.get('domain') == 'formal'][:2]
    print("\n[  ]")
    for sample in news_samples:
        print(f"  : {sample['text'][:60]}...")
        print(f"  : {sample['patterns'][0]['text']} → {sample['patterns'][0]['entity']}")

    #   
    internet_samples = [s for s in all_data if s.get('domain') == 'informal'][:2]
    print("\n[  ]")
    for sample in internet_samples:
        print(f"  : {sample['text'][:60]}...")
        print(f"  : {sample['patterns'][0]['text']} → {sample['patterns'][0]['entity']}")
