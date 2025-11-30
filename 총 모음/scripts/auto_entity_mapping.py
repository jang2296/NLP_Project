#!/usr/bin/env python3
"""
 Entity  
      

: ENTITY_MAPPING_STRATEGY.md 
: 30-50% Entity  
"""

import json
import re
from typing import Dict, List, Optional, Tuple
from collections import Counter
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EntityMapper:
    """  Entity  """

    def __init__(self):
        self.company_rules = self._init_company_rules()
        self.country_rules = self._init_country_rules()
        self.person_rules = self._init_person_rules()

        # 
        self.stats = {
            'total_patterns': 0,
            'mapped_patterns': 0,
            'company_mapped': 0,
            'country_mapped': 0,
            'person_mapped': 0,
            'by_confidence': Counter()
        }

    def _init_company_rules(self) -> Dict:
        """   """
        return {
            'S': {
                'samsung': {
                    'keywords': ['', '', '', '', '',
                                '', 'OLED', '', 'SSD', ''],
                    'entity': '',
                    'base_confidence': 0.85
                },
                'sk': {
                    'keywords': ['', '', 'SK', '', 'LTE', '5G',
                                '', '', ''],
                    'entity': 'SK',
                    'base_confidence': 0.85
                },
                'spc': {
                    'keywords': ['', '', '', '', ''],
                    'entity': 'SPC',
                    'base_confidence': 0.80
                }
            },
            'K': {
                'kt': {
                    'keywords': ['', '', '5G', 'LTE', '', ''],
                    'entity': 'KT',
                    'base_confidence': 0.90
                },
                'kakao': {
                    'keywords': ['', '', '', '', '', ''],
                    'entity': '',
                    'base_confidence': 0.90
                },
                'kia': {
                    'keywords': ['', '', 'K5', 'K7', '', ''],
                    'entity': '',
                    'base_confidence': 0.85
                }
            },
            'L': {
                'lg': {
                    'keywords': ['', 'TV', '', '', '', 'OLED',
                                '', '', ''],
                    'entity': 'LG',
                    'base_confidence': 0.90
                },
                'lotte': {
                    'keywords': ['', '', '', '', '', ''],
                    'entity': '',
                    'base_confidence': 0.85
                }
            },
            'H': {
                'hyundai': {
                    'keywords': ['', '', '', '', '', '',
                                '', ''],
                    'entity': '',
                    'base_confidence': 0.85
                },
                'hanwha': {
                    'keywords': ['', '', '', '', '', ''],
                    'entity': '',
                    'base_confidence': 0.80
                }
            },
            'N': {
                'naver': {
                    'keywords': ['', '', '', '', '', '',
                                '', ''],
                    'entity': '',
                    'base_confidence': 0.90
                },
                'nexon': {
                    'keywords': ['', '', '', '', 'PC'],
                    'entity': '',
                    'base_confidence': 0.85
                }
            },
            'C': {
                'coupang': {
                    'keywords': ['', '', '', '', '', ''],
                    'entity': '',
                    'base_confidence': 0.90
                }
            }
        }

    def _init_country_rules(self) -> Dict:
        """   """
        return {
            '': {
                'keywords': ['', '', '', '', '',
                            '', '', '', '', '',
                            '', '', '', ''],
                'negative': ['', '', ''],  #  
                'base_confidence': 0.80
            },
            '': {
                'keywords': ['', '', '', '', '', '',
                            '', '', '', '', '',
                            '', '', ''],
                'negative': ['', ''],
                'base_confidence': 0.80
            },
            '': {
                'keywords': ['', '', '', '', '', 'FBI',
                            'CIA', '', '', '', '',
                            '', '', 'NASA'],
                'negative': ['', ''],
                'base_confidence': 0.80
            },
            '': {
                'keywords': ['', '', '', '', '',
                            '', '', ''],
                'negative': ['', 'EU'],
                'base_confidence': 0.75
            },
            '': {
                'keywords': ['', '', '', '', '', '',
                            '', '38'],
                'negative': ['', ''],
                'base_confidence': 0.85
            }
        }

    def _init_person_rules(self) -> Dict:
        """    ()"""
        return {
            #  
            'common_surnames': ['', '', '', '', '', '', '', '', '', '']
        }

    def map_entity(self, pattern_text: str, context: str, pattern_type: str) -> Optional[Dict]:
        """
           Entity 

        Args:
            pattern_text:   (: "S ")
            context:   ()
            pattern_type:  

        Returns:
            {"entity": "", "confidence": 0.9}  None
        """
        self.stats['total_patterns'] += 1

        # 1.   
        company_result = self._match_company(pattern_text, context)
        if company_result:
            self.stats['mapped_patterns'] += 1
            self.stats['company_mapped'] += 1
            self._update_confidence_stats(company_result['confidence'])
            return company_result

        # 2.   
        country_result = self._match_country(pattern_text, context)
        if country_result:
            self.stats['mapped_patterns'] += 1
            self.stats['country_mapped'] += 1
            self._update_confidence_stats(country_result['confidence'])
            return country_result

        # 3.   
        person_result = self._match_person(pattern_text, context)
        if person_result:
            self.stats['mapped_patterns'] += 1
            self.stats['person_mapped'] += 1
            self._update_confidence_stats(person_result['confidence'])
            return person_result

        return None

    def _match_company(self, pattern: str, context: str) -> Optional[Dict]:
        """  """
        #   
        # "S ", "S", "S", "" 
        initial_patterns = [
            r'([A-Z-])\s*\s*(?:||)',
            r'([A-Z-])\s*',
            r'([A-Z-])\s*',
            r'([A-Z-])\s*',
            r'([A-Z-])\s*'
        ]

        initial = None
        for regex in initial_patterns:
            match = re.search(regex, pattern)
            if match:
                initial = match.group(1)
                break

        if not initial:
            return None

        #     
        if initial not in self.company_rules:
            return None

        candidates = self.company_rules[initial]
        context_lower = context.lower()

        #     
        best_match = None
        best_score = 0

        for candidate_key, candidate_info in candidates.items():
            score = 0
            matched_keywords = []

            #  
            for keyword in candidate_info['keywords']:
                if keyword.lower() in context_lower:
                    score += 1
                    matched_keywords.append(keyword)

            if score > best_score:
                best_score = score
                best_match = {
                    'entity': candidate_info['entity'],
                    'confidence': min(0.95, candidate_info['base_confidence'] + (score * 0.03)),
                    'matched_keywords': matched_keywords
                }

        #  1   
        if best_score >= 1:
            return best_match

        return None

    def _match_country(self, pattern: str, context: str) -> Optional[Dict]:
        """  """
        #   
        country_patterns = [
            ' ', ' ', ' ', '', ' ',
            ' ', '', '', ''
        ]

        #    
        is_country_pattern = any(cp in pattern for cp in country_patterns)
        if not is_country_pattern:
            return None

        context_lower = context.lower()

        #     
        scores = []

        for country, rules in self.country_rules.items():
            score = 0
            matched_keywords = []

            #  
            for keyword in rules['keywords']:
                if keyword.lower() in context_lower:
                    score += 1
                    matched_keywords.append(keyword)

            #   (  )
            for neg_keyword in rules.get('negative', []):
                if neg_keyword in context_lower:
                    score -= 2

            if score > 0:
                scores.append({
                    'country': country,
                    'score': score,
                    'keywords': matched_keywords,
                    'base_confidence': rules['base_confidence']
                })

        #   
        scores.sort(key=lambda x: x['score'], reverse=True)

        #  2     3 
        if scores and (scores[0]['score'] >= 3 or len(scores[0]['keywords']) >= 2):
            best = scores[0]
            return {
                'entity': best['country'],
                'confidence': min(0.95, best['base_confidence'] + (best['score'] * 0.05)),
                'matched_keywords': best['keywords']
            }

        return None

    def _match_person(self, pattern: str, context: str) -> Optional[Dict]:
        """   ( -  )"""
        # ,   
        person_match = re.search(r'([-])\s*', pattern)
        if person_match:
            surname = person_match.group(1)

            #    
            if surname in self.person_rules['common_surnames']:
                return {
                    'entity': f'{surname}',
                    'confidence': 0.50  #   (  )
                }

        # "" 
        person_match2 = re.search(r'([-]{1,2})\s*', pattern)
        if person_match2:
            name_part = person_match2.group(1)
            return {
                'entity': f'{name_part}',
                'confidence': 0.55
            }

        return None

    def _update_confidence_stats(self, confidence: float):
        """  """
        if confidence >= 0.9:
            self.stats['by_confidence']['high (>=0.9)'] += 1
        elif confidence >= 0.8:
            self.stats['by_confidence']['medium (0.8-0.9)'] += 1
        elif confidence >= 0.7:
            self.stats['by_confidence']['low (0.7-0.8)'] += 1
        else:
            self.stats['by_confidence']['very_low (<0.7)'] += 1

    def get_statistics(self) -> Dict:
        """  """
        mapping_rate = 0
        if self.stats['total_patterns'] > 0:
            mapping_rate = (self.stats['mapped_patterns'] / self.stats['total_patterns']) * 100

        return {
            'total_patterns': self.stats['total_patterns'],
            'mapped_patterns': self.stats['mapped_patterns'],
            'mapping_rate': f'{mapping_rate:.1f}%',
            'by_type': {
                'company': self.stats['company_mapped'],
                'country': self.stats['country_mapped'],
                'person': self.stats['person_mapped']
            },
            'by_confidence': dict(self.stats['by_confidence'])
        }


def process_dataset(input_path: str, output_path: str, stats_path: str = None):
    """
      Entity  

    Args:
        input_path:  JSON  
        output_path:  JSON  
        stats_path:  JSON   ()
    """
    logger.info(f"[START] Entity  ")
    logger.info(f"   : {input_path}")
    logger.info(f"   : {output_path}")

    #  
    logger.info("   ...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"    {len(data):,} ")

    #  
    mapper = EntityMapper()

    #  
    total_items = len(data)
    processed = 0

    logger.info("[PROGRESS] Entity   ...")

    for item in data:
        text = item['text']

        for pattern in item.get('patterns', []):
            #    
            if pattern.get('entity') and pattern['entity'] not in ['', 'UNKNOWN']:
                mapper.stats['total_patterns'] += 1
                mapper.stats['mapped_patterns'] += 1
                continue

            #   
            result = mapper.map_entity(
                pattern_text=pattern['text'],
                context=text,
                pattern_type=pattern.get('type', '')
            )

            if result:
                pattern['entity'] = result['entity']
                pattern['confidence'] = result['confidence']

                #  ()
                if result.get('matched_keywords'):
                    pattern['_matched_keywords'] = result['matched_keywords']

        processed += 1

        #   (10,000)
        if processed % 10000 == 0:
            logger.info(f"   : {processed:,}/{total_items:,} ({processed/total_items*100:.1f}%)")

    #  
    logger.info("   ...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    #  
    stats = mapper.get_statistics()

    logger.info("\n" + "="*80)
    logger.info("[OK] Entity  !")
    logger.info("="*80)
    logger.info(f"\n[STATS]  :")
    logger.info(f"     : {stats['total_patterns']:,}")
    logger.info(f"    : {stats['mapped_patterns']:,}")
    logger.info(f"   : {stats['mapping_rate']}")
    logger.info(f"\n    :")
    logger.info(f"      : {stats['by_type']['company']:,}")
    logger.info(f"      : {stats['by_type']['country']:,}")
    logger.info(f"      : {stats['by_type']['person']:,}")
    logger.info(f"\n    :")
    for conf_level, count in stats['by_confidence'].items():
        logger.info(f"      {conf_level}: {count:,}")

    #   
    if stats_path:
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"\n[STATS]   : {stats_path}")

    logger.info(f"\n[OK]  : {output_path}")
    logger.info("="*80)

    return stats


if __name__ == "__main__":
    import sys

    #  
    input_file = 'data/processed/final_training_dataset.json'
    output_file = 'data/processed/final_training_dataset_mapped.json'
    stats_file = 'data/processed/entity_mapping_statistics.json'

    #   
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    try:
        stats = process_dataset(input_file, output_file, stats_file)

        print("\n" + " Entity   !")
        print(f"   : {stats['mapping_rate']}")
        print(f"    : ML      ")

    except FileNotFoundError as e:
        logger.error(f"[ERROR]    : {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"[ERROR]  : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
