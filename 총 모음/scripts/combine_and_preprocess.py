"""
    

   + AI Hub   
ML    .
"""

import json
import re
from typing import List, Dict
from pathlib import Path
import logging
from collections import Counter
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataCombiner:
    """   """

    def __init__(self):
        self.namuwiki_path = Path("./data/raw")
        self.aihub_path = Path("./data/raw/aihub")
        self.output_path = Path("./data/processed")
        self.output_path.mkdir(parents=True, exist_ok=True)

    def load_namuwiki_data(self) -> List[Dict]:
        """   """
        #    
        json_files = list(self.namuwiki_path.glob("namuwiki_euphemisms_*.json"))
        if not json_files:
            raise FileNotFoundError("     .")

        latest_file = sorted(json_files)[-1]
        logger.info(f"  : {latest_file}")

        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"  → {len(data)}  ")
        return data

    def load_aihub_data(self) -> List[Dict]:
        """AI Hub   """
        # process_aihub_data.py    
        aihub_file = self.output_path / "aihub_training_data.json"

        if not aihub_file.exists():
            logger.warning("AI Hub   . .")
            return []

        logger.info(f"AI Hub  : {aihub_file}")

        with open(aihub_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"  → {len(data)}  ")
        return data

    def clean_text(self, text: str) -> str:
        """ """
        # URL 
        text = re.sub(r'https?://\S+', '', text)

        # HTML  
        text = re.sub(r'<[^>]+>', '', text)

        #   
        text = re.sub(r'\s+', ' ', text)

        #   
        text = text.strip()

        return text

    def is_valid_item(self, text: str) -> bool:
        """   """
        #   
        if len(text) < 3:
            return False

        # / 
        if any(word in text for word in ['[]', '↑', '', '']):
            return False

        #   
        if text.isdigit():
            return False

        return True

    def merge_duplicates(self, items: List[Dict]) -> List[Dict]:
        """   """
        unique_items = {}

        for item in items:
            text = item['text']

            if text not in unique_items:
                unique_items[text] = item
            else:
                #   
                existing = unique_items[text]

                #  
                if 'categories' not in existing:
                    existing['categories'] = [existing.get('category', 'unknown')]

                if 'category' in item:
                    cat = item['category']
                    if cat not in existing['categories']:
                        existing['categories'].append(cat)

        return list(unique_items.values())

    def convert_to_training_format(self, items: List[Dict]) -> List[Dict]:
        """  """
        training_data = []

        for item in items:
            # AI Hub      
            if 'patterns' in item and isinstance(item.get('patterns'), list):
                training_data.append(item)
                continue

            #    
            text = item['text']
            euphemism = item.get('euphemism', '')

            if not euphemism:
                continue

            # BIO    
            start_idx = text.find(euphemism)

            if start_idx == -1:
                #       
                continue

            training_item = {
                'text': text,
                'patterns': [{
                    'text': euphemism,
                    'start': start_idx,
                    'end': start_idx + len(euphemism),
                    'type': item.get('category', 'unknown'),
                    'entity': item.get('entity', 'UNKNOWN'),
                    'confidence': 0.85 if item.get('has_euphemism', False) else 0.6
                }],
                'source': item.get('source', 'Namuwiki'),
                'metadata': {
                    'collected_at': item.get('collected_at', datetime.now().isoformat()),
                    'categories': item.get('categories', [item.get('category', 'unknown')])
                }
            }

            training_data.append(training_item)

        return training_data

    def get_statistics(self, data: List[Dict]) -> Dict:
        """  """
        categories = []
        euphemisms = []
        entities = []

        for item in data:
            if 'category' in item:
                categories.append(item['category'])
            if 'categories' in item:
                categories.extend(item['categories'])

            if 'euphemism' in item:
                euphemisms.append(item['euphemism'])

            if 'entity' in item and item['entity'] != 'UNKNOWN':
                entities.append(item['entity'])

        return {
            'total_items': len(data),
            'category_dist': dict(Counter(categories).most_common(20)),
            'unique_euphemisms': len(set(euphemisms)),
            'unique_entities': len(set(entities)),
            'avg_text_length': sum(len(item['text']) for item in data) / len(data) if data else 0
        }

    def run(self):
        """    """
        logger.info("="*70)
        logger.info("    ")
        logger.info("="*70)

        # 1.  
        logger.info("\n[1]  ")
        namuwiki_data = self.load_namuwiki_data()
        aihub_data = self.load_aihub_data()

        # 2.  
        logger.info("\n[2]  ")
        cleaned_namuwiki = []
        for item in namuwiki_data:
            item['text'] = self.clean_text(item['text'])
            if self.is_valid_item(item['text']):
                cleaned_namuwiki.append(item)

        logger.info(f"  : {len(namuwiki_data)} → {len(cleaned_namuwiki)} ( )")

        # AI Hub      
        cleaned_aihub = []
        for item in aihub_data:
            # AI Hub      
            if self.is_valid_item(item['text']):
                cleaned_aihub.append(item)

        logger.info(f"  AI Hub: {len(aihub_data)} → {len(cleaned_aihub)} ( )")

        # 3.  
        logger.info("\n[3]  ")
        combined_data = cleaned_namuwiki + cleaned_aihub
        logger.info(f"   : {len(combined_data)}")

        # 4.  
        logger.info("\n[4]  ")
        merged_data = self.merge_duplicates(combined_data)
        logger.info(f"    : {len(merged_data)}")

        # 5.   
        logger.info("\n[5]   ")
        training_data = self.convert_to_training_format(merged_data)
        logger.info(f"   : {len(training_data)}")

        # 6.  
        logger.info("\n[6]  ")
        stats = self.get_statistics(merged_data)

        logger.info(f"\n : {stats['total_items']}")
        logger.info(f"  : {stats['unique_euphemisms']}")
        logger.info(f" : {stats['unique_entities']}")
        logger.info(f"  : {stats['avg_text_length']:.1f}")

        logger.info("\n :")
        for cat, count in list(stats['category_dist'].items())[:10]:
            logger.info(f"  - {cat}: {count}")

        # 7. 
        logger.info("\n[7]  ")

        #    
        combined_file = self.output_path / "combined_raw_data.json"
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        logger.info(f"   : {combined_file}")

        #   
        training_file = self.output_path / "final_training_dataset.json"
        with open(training_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        logger.info(f"   : {training_file}")

        #  
        stats_file = self.output_path / "data_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"   : {stats_file}")

        logger.info("\n" + "="*70)
        logger.info("[OK]   !")
        logger.info("="*70)
        logger.info(f"\n  : {training_file}")
        logger.info(f"ML  !")
        logger.info("="*70)

        return training_file


if __name__ == '__main__':
    combiner = DataCombiner()
    combiner.run()
