"""
AI Hub      

6,243 JSON    
 , ,    .
"""

import json
import os
import re
from typing import List, Dict, Set
from pathlib import Path
from collections import Counter
import logging
from datetime import datetime
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AIHubDataProcessor:
    """AI Hub   """

    def __init__(self, aihub_dir: str = None):
        """
        Args:
            aihub_dir: AI Hub   
        """
        if aihub_dir is None:
            aihub_dir = "/mnt/c/claude_project/NLP/ /031.   /01./1.Training_220728_add/"

        self.aihub_dir = Path(aihub_dir)
        self.output_dir = Path("./data/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        #   
        self.euphemism_patterns = [
            r'[A-Z-]\s*\s*(|||)',  # S 
            r'[A-Z-]\s*',  # S, K
            r'[A-Z-]\s*',  # L
            r'[A-Z-]\s*',  # K
            r'\s*',  #  
            r'\s*(|)',  #  
            r'\s*',  # 
            r'[-]{1,2}\s*',  # 
        ]

        # /  
        self.profanity_indicators = [
            '()',
            '()',
            '()',
            '()',
        ]

        # //  
        self.neologism_patterns = [
            r'{2,}',  # 
            r'{2,}',  # 
            r'',  # 
            r'',  # 
            r'',  # 
            r'',  # 
            r'',  # 
            r'',  # 
            r'',  # 
            r'',
            r'',
            r'\s*\S+',  # XX
        ]

    def find_json_files(self) -> List[Path]:
        """ JSON  """
        logger.info(f"JSON   : {self.aihub_dir}")
        json_files = list(self.aihub_dir.rglob("*.json"))
        logger.info(f"  → {len(json_files)}  ")
        return json_files

    def parse_json_file(self, filepath: Path) -> List[Dict]:
        """JSON  """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            items = []

            # named_entity  
            for entity in data.get('named_entity', []):
                for content_item in entity.get('content', []):
                    sentence = content_item.get('sentence', '').strip()
                    labels = content_item.get('labels', [])

                    if not sentence or len(sentence) < 3:
                        continue

                    #  
                    item = {
                        'text': sentence,
                        'source': 'AI_Hub',
                        'category': self._extract_category(filepath),
                        'source_site': entity.get('source_site', 'unknown'),
                        'write_date': entity.get('write_date', 'unknown'),
                        'labels': labels,
                        'has_profanity': self._has_profanity(sentence),
                        'has_euphemism': self._has_euphemism(sentence),
                        'has_neologism': self._has_neologism(sentence),
                        'collected_at': datetime.now().isoformat()
                    }

                    items.append(item)

            return items

        except Exception as e:
            logger.error(f"   ({filepath}): {e}")
            return []

    def _extract_category(self, filepath: Path) -> str:
        """   """
        parts = filepath.parts
        for i, part in enumerate(parts):
            if part == '' and i + 1 < len(parts):
                return parts[i + 1]
        return 'unknown'

    def _has_profanity(self, text: str) -> bool:
        """/  """
        for indicator in self.profanity_indicators:
            if indicator in text:
                return True
        return False

    def _has_euphemism(self, text: str) -> bool:
        """   """
        for pattern in self.euphemism_patterns:
            if re.search(pattern, text):
                return True
        return False

    def _has_neologism(self, text: str) -> bool:
        """/  """
        for pattern in self.neologism_patterns:
            if re.search(pattern, text):
                return True
        return False

    def detect_patterns(self, text: str) -> List[Dict]:
        """  """
        patterns = []

        #   
        for pattern_str in self.euphemism_patterns:
            matches = re.finditer(pattern_str, text)
            for match in matches:
                patterns.append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'type': 'euphemism',
                    'pattern': pattern_str
                })

        #  
        for pattern_str in self.neologism_patterns:
            matches = re.finditer(pattern_str, text)
            for match in matches:
                patterns.append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'type': 'neologism',
                    'pattern': pattern_str
                })

        return patterns

    def process_all_files(self, max_files: int = None) -> List[Dict]:
        """  """
        logger.info("\n" + "="*70)
        logger.info("AI Hub     ")
        logger.info("="*70)

        #   
        json_files = self.find_json_files()

        if max_files:
            json_files = json_files[:max_files]
            logger.info(f" : {len(json_files)} ()")

        #   
        all_items = []
        logger.info("\n  ...")

        for filepath in tqdm(json_files, desc="Processing"):
            items = self.parse_json_file(filepath)
            all_items.extend(items)

        logger.info(f"\n {len(all_items)}  ")

        return all_items

    def analyze_data(self, items: List[Dict]) -> Dict:
        """ """
        logger.info("\n  ...")

        categories = Counter()
        sources = Counter()
        profanity_count = 0
        euphemism_count = 0
        neologism_count = 0

        for item in items:
            categories[item['category']] += 1
            sources[item['source_site']] += 1

            if item['has_profanity']:
                profanity_count += 1
            if item['has_euphemism']:
                euphemism_count += 1
            if item['has_neologism']:
                neologism_count += 1

        return {
            'total_items': len(items),
            'categories': dict(categories.most_common(20)),
            'sources': dict(sources.most_common(10)),
            'profanity_count': profanity_count,
            'euphemism_count': euphemism_count,
            'neologism_count': neologism_count,
            'avg_text_length': sum(len(item['text']) for item in items) / len(items) if items else 0
        }

    def convert_to_training_format(self, items: List[Dict]) -> List[Dict]:
        """  """
        logger.info("\n   ...")

        training_data = []

        for item in items:
            #  
            patterns = self.detect_patterns(item['text'])

            if not patterns:
                #     
                patterns = [{
                    'text': item['text'][:20],
                    'start': 0,
                    'end': min(20, len(item['text'])),
                    'type': 'colloquial',
                    'entity': 'UNKNOWN',
                    'confidence': 0.5
                }]

            training_item = {
                'text': item['text'],
                'patterns': [{
                    'text': p['text'],
                    'start': p['start'],
                    'end': p['end'],
                    'type': p['type'],
                    'entity': 'UNKNOWN',
                    'confidence': 0.7
                } for p in patterns],
                'source': item['source'],
                'metadata': {
                    'category': item['category'],
                    'source_site': item['source_site'],
                    'has_profanity': item['has_profanity'],
                    'has_euphemism': item['has_euphemism'],
                    'has_neologism': item['has_neologism'],
                    'collected_at': item['collected_at']
                }
            }

            training_data.append(training_item)

        logger.info(f"  → {len(training_data)}   ")
        return training_data

    def save_data(self, items: List[Dict], training_data: List[Dict], stats: Dict):
        """ """
        logger.info("\n  ...")

        #   
        raw_file = self.output_dir / "aihub_raw_data.json"
        with open(raw_file, 'w', encoding='utf-8') as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        logger.info(f"   : {raw_file}")

        #   
        training_file = self.output_dir / "aihub_training_data.json"
        with open(training_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        logger.info(f"   : {training_file}")

        #  
        stats_file = self.output_dir / "aihub_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"   : {stats_file}")

    def run(self, max_files: int = None):
        """  """
        #  
        items = self.process_all_files(max_files=max_files)

        #  
        stats = self.analyze_data(items)

        #  
        logger.info("\n" + "="*70)
        logger.info("AI Hub  ")
        logger.info("="*70)
        logger.info(f"\n : {stats['total_items']:,}")
        logger.info(f"/: {stats['profanity_count']:,}")
        logger.info(f" : {stats['euphemism_count']:,}")
        logger.info(f"/: {stats['neologism_count']:,}")
        logger.info(f"  : {stats['avg_text_length']:.1f}")

        logger.info("\n 10 :")
        for cat, count in list(stats['categories'].items())[:10]:
            logger.info(f"  {cat}: {count:,}")

        #   
        training_data = self.convert_to_training_format(items)

        # 
        self.save_data(items, training_data, stats)

        logger.info("\n" + "="*70)
        logger.info("[OK] AI Hub   !")
        logger.info("="*70)

        return items, training_data, stats


if __name__ == '__main__':
    processor = AIHubDataProcessor()

    #    (    )
    processor.run()  #  9,474  

    #   
    # processor.run(max_files=1000)  # 1,000  
