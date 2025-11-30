"""
    

       .
"""

import json
import re
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
import pandas as pd
from konlpy.tag import Mecab  #  Okt
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """ """

    def __init__(self, use_mecab: bool = True):
        """
        Args:
            use_mecab: Mecab     (False Okt )
        """
        try:
            if use_mecab:
                from konlpy.tag import Mecab
                self.tokenizer = Mecab()
            else:
                from konlpy.tag import Okt
                self.tokenizer = Okt()
        except:
            logger.warning("KoNLPy  : pip install konlpy")
            self.tokenizer = None

        #   
        self.blacklist_terms = [
            '', '', '', '', '',
            'NSFW', '', '', ''
        ]

    def clean_text(self, text: str) -> str:
        """
         

        Args:
            text:  

        Returns:
             
        """
        # HTML  
        text = re.sub(r'<[^>]+>', '', text)

        #   
        text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)  # [[]] → 
        text = re.sub(r'\[([^\]]+)\]', '', text)  # [] 

        #  
        text = re.sub(r'\s+', ' ', text)  #   
        text = text.strip()

        #   ()
        # text = re.sub(r'[^\w\s-]', '', text)

        return text

    def remove_duplicates(self, data: List[Dict]) -> List[Dict]:
        """
          

        Args:
            data:   

        Returns:
              
        """
        seen_texts = set()
        unique_data = []

        for item in data:
            text = item.get('text', '')
            euphemism = item.get('euphemism', '')

            #      
            key = f"{text}_{euphemism}"

            if key not in seen_texts:
                seen_texts.add(key)
                unique_data.append(item)

        logger.info(f" : {len(data)} → {len(unique_data)} ({len(data) - len(unique_data)} )")

        return unique_data

    def filter_quality(self, data: List[Dict]) -> List[Dict]:
        """
           

        Args:
            data:  

        Returns:
             
        """
        filtered = []

        for item in data:
            text = item.get('text', '')
            euphemism = item.get('euphemism', '')
            entity = item.get('entity', 'UNKNOWN')

            #  
            if len(text) < 10 or len(text) > 500:
                continue  #    

            if entity == 'UNKNOWN':
                continue  #   

            if not euphemism or len(euphemism) < 2:
                continue  #    

            #  
            if any(term in text.lower() for term in self.blacklist_terms):
                continue

            #    
            if not re.search(r'[-]', text):
                continue

            filtered.append(item)

        logger.info(f" : {len(data)} → {len(filtered)} ({len(data) - len(filtered)} )")

        return filtered

    def augment_data(self, data: List[Dict]) -> List[Dict]:
        """
         

        Args:
            data:  

        Returns:
             
        """
        augmented = list(data)  #  

        for item in data:
            text = item['text']
            euphemism = item['euphemism']
            entity = item['entity']

            # 1.  
            variations = self._generate_variations(euphemism)

            for var in variations:
                if var != euphemism and var in text:
                    #   
                    new_item = item.copy()
                    new_item['text'] = text.replace(euphemism, var, 1)
                    new_item['euphemism'] = var
                    new_item['augmented'] = True
                    augmented.append(new_item)

            # 2.   ( )
            templates = [
                f"{euphemism}() .",
                f"{euphemism}() {entity}() .",
                f" {euphemism}   .",
                f"{euphemism}   ."
            ]

            for template in templates:
                new_item = {
                    'text': template,
                    'euphemism': euphemism,
                    'entity': entity,
                    'category': item.get('category', 'unknown'),
                    'augmented': True,
                    'confidence': 0.7  #    
                }
                augmented.append(new_item)

        logger.info(f" : {len(data)} → {len(augmented)} (+{len(augmented) - len(data)})")

        return augmented

    def _generate_variations(self, euphemism: str) -> List[str]:
        """   """
        variations = [euphemism]

        #  
        variations.append(euphemism.replace(' ', ''))  #  
        variations.append(re.sub(r'\s+', '  ', euphemism))  #  2

        #   (: "S " → "S", "S  ")
        if '' in euphemism:
            variations.append(euphemism.replace(' ', ''))
            variations.append(euphemism.replace(' ', ''))

        return list(set(variations))

    def balance_categories(
        self,
        data: List[Dict],
        target_per_category: int = 1000
    ) -> List[Dict]:
        """
           

        Args:
            data:  
            target_per_category:    

        Returns:
              
        """
        #  
        by_category = defaultdict(list)

        for item in data:
            category = item.get('category', 'unknown')
            by_category[category].append(item)

        balanced = []

        for category, items in by_category.items():
            if len(items) > target_per_category:
                # 
                import random
                sampled = random.sample(items, target_per_category)
                balanced.extend(sampled)
                logger.info(f"{category}: {len(items)} → {target_per_category} ()")
            elif len(items) < target_per_category:
                #  ()
                factor = target_per_category // len(items)
                remainder = target_per_category % len(items)

                balanced.extend(items * factor)
                balanced.extend(items[:remainder])
                logger.info(f"{category}: {len(items)} → {target_per_category} ()")
            else:
                balanced.extend(items)
                logger.info(f"{category}: {len(items)} ()")

        return balanced

    def create_bio_labels(
        self,
        text: str,
        patterns: List[Dict]
    ) -> List[int]:
        """
        BIO   

        Args:
            text:  
            patterns:    

        Returns:
            BIO   (0: O, 1: B-EUPH, 2: I-EUPH)
        """
        if not self.tokenizer:
            #    
            labels = [0] * len(text)
        else:
            #  
            tokens = self.tokenizer.morphs(text)
            labels = [0] * len(tokens)

            for pattern in patterns:
                pattern_text = pattern['text']
                pattern_tokens = self.tokenizer.morphs(pattern_text)

                #   
                for i in range(len(tokens) - len(pattern_tokens) + 1):
                    if tokens[i:i+len(pattern_tokens)] == pattern_tokens:
                        labels[i] = 1  # B-EUPH
                        for j in range(1, len(pattern_tokens)):
                            labels[i + j] = 2  # I-EUPH

        return labels

    def analyze_dataset(self, data: List[Dict]) -> Dict:
        """
          

        Args:
            data: 

        Returns:
             
        """
        df = pd.DataFrame(data)

        stats = {
            'total_samples': len(df),
            'unique_texts': df['text'].nunique(),
            'unique_euphemisms': df['euphemism'].nunique(),
            'unique_entities': df['entity'].nunique(),
            'category_distribution': df['category'].value_counts().to_dict(),
            'avg_text_length': df['text'].str.len().mean(),
            'min_text_length': df['text'].str.len().min(),
            'max_text_length': df['text'].str.len().max(),
        }

        #   
        stats['top_euphemisms'] = df['euphemism'].value_counts().head(10).to_dict()

        #  
        stats['top_entities'] = df['entity'].value_counts().head(10).to_dict()

        return stats

    def save_processed_data(
        self,
        data: List[Dict],
        output_path: str
    ):
        """  """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f" : {output_path} ({len(data)} )")


def main():
    """  """
    print("="*70)
    print("  ")
    print("="*70)

    # 1.  
    input_path = "./data/raw/namuwiki_euphemisms_latest.json"
    logger.info(f"\n : {input_path}")

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"   : {input_path}")
        logger.info(" namuwiki_crawler.py   .")
        return

    logger.info(f" : {len(raw_data)} ")

    # 2.  
    preprocessor = DataPreprocessor(use_mecab=False)  # Okt 

    #  
    logger.info("\n[1/6]   ...")
    for item in raw_data:
        item['text'] = preprocessor.clean_text(item['text'])
        if 'euphemism' in item:
            item['euphemism'] = preprocessor.clean_text(item['euphemism'])

    #  
    logger.info("\n[2/6]   ...")
    data = preprocessor.remove_duplicates(raw_data)

    #  
    logger.info("\n[3/6]   ...")
    data = preprocessor.filter_quality(data)

    #  
    logger.info("\n[4/6]   ...")
    data = preprocessor.augment_data(data)

    #   
    logger.info("\n[5/6]    ...")
    data = preprocessor.balance_categories(data, target_per_category=500)

    #  
    logger.info("\n[6/6]   ...")
    stats = preprocessor.analyze_dataset(data)

    #  
    print("\n" + "="*70)
    print(" ")
    print("="*70)
    print(f"\n  : {stats['total_samples']}")
    print(f" : {stats['unique_texts']}")
    print(f"  : {stats['unique_euphemisms']}")
    print(f" : {stats['unique_entities']}")
    print(f"\n  : {stats['avg_text_length']:.1f}")
    print(f"/ : {stats['min_text_length']}/{stats['max_text_length']}")

    print("\n :")
    for cat, count in stats['category_distribution'].items():
        print(f"  - {cat}: {count}")

    print("\n  :")
    for euph, count in list(stats['top_euphemisms'].items())[:5]:
        print(f"  - {euph}: {count}")

    # 
    output_path = "./data/processed/preprocessed_dataset.json"
    preprocessor.save_processed_data(data, output_path)

    #   
    logger.info("\n   ...")
    training_data = []

    for item in data:
        text = item['text']
        euphemism = item['euphemism']

        start = text.find(euphemism)
        if start == -1:
            continue

        end = start + len(euphemism)

        training_item = {
            'text': text,
            'patterns': [{
                'text': euphemism,
                'start': start,
                'end': end,
                'type': item.get('category', 'unknown'),
                'entity': item.get('entity', 'UNKNOWN'),
                'confidence': item.get('confidence', 0.8)
            }]
        }

        training_data.append(training_item)

    training_path = "./data/processed/training_dataset.json"
    with open(training_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)

    logger.info(f" : {training_path} ({len(training_data)})")

    print("\n" + "="*70)
    print("!")
    print("="*70)
    print(f"\n : {output_path}")
    print(f" : {training_path}")


if __name__ == "__main__":
    main()
