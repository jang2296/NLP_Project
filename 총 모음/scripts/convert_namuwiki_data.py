"""
     

  ():
{
    "text": " ...",
    "euphemism": "",
    "entity": "%EC%9D%B8%EC%8A%A4%ED%8A%B8%EB%A3%A8%EB%A9%98%ED%83%88",
    "category": "",
    "source": "https://namu.wiki/...",
    "collected_at": "2025-11-17T...",
    "has_euphemism": true
}

  ( ):
{
    "text": "    ?",
    "patterns": [{
        "text": "",
        "type": "internet_slang",
        "start": 4,
        "end": 7,
        "entity": "",
        "confidence": 1.0
    }],
    "has_euphemism": true,
    "source": "namuwiki",
    "char_labels": ["O", "O", "O", "O", "B-EUPH", "I-EUPH", "I-EUPH", ...]
}
"""

import json
import re
import random
from urllib.parse import unquote
from typing import List, Dict, Tuple
from pathlib import Path

#  →  
CATEGORY_TO_TYPE = {
    '': 'yamin',
    '': 'abbreviation',
    ' ': 'internet_meme',
    '': 'internet_meme',
    '': 'profanity',
    '': 'profanity',
    '': 'euphemism',
    '': 'euphemism',
    '': 'neologism',
    '': 'slang',
    ' ': 'slang',
    ' ': 'slang',
    '': 'nickname',
    ' ': 'dark_humor',
    '': 'satire',
    ' ': 'internet_slang',
    '': 'internet_slang',
    '': 'internet_slang',
    '': 'internet_slang',
    '': 'internet_slang',
    '': 'internet_slang',
}

#   
CONTEXT_TEMPLATES = [
    " {euphemism}   ?",
    "SNS {euphemism}    ?",
    "  {euphemism}   ?",
    "{euphemism}  ?",
    "  {euphemism}   ?",
    " {euphemism}  ",
    " {euphemism}   ",
    " {euphemism}   ",
    "{euphemism}     ",
    " {euphemism}   ?",
    " {euphemism}   ?",
    "  {euphemism} ",
    "{euphemism}   ?",
    " {euphemism} ",
    "{euphemism}  ?",
    " {euphemism} ",
]

# /   
NEGATIVE_TEMPLATES = [
    " {euphemism}  ?",
    "{euphemism}      ?",
    " {euphemism}  ?",
    "{euphemism}  ?",
]


def decode_entity(entity: str) -> str:
    """URL  entity """
    if not entity:
        return "  "
    try:
        decoded = unquote(entity)
        # HTML  
        decoded = re.sub(r'&#\d+;', '', decoded)
        decoded = decoded.strip()
        return decoded if decoded else "  "
    except:
        return entity


def get_type_from_category(category: str) -> str:
    """  """
    return CATEGORY_TO_TYPE.get(category, 'internet_slang')


def generate_char_labels(text: str, start: int, end: int) -> List[str]:
    """BIO   """
    labels = ['O'] * len(text)
    if 0 <= start < len(text) and start < end <= len(text):
        labels[start] = 'B-EUPH'
        for i in range(start + 1, min(end, len(text))):
            labels[i] = 'I-EUPH'
    return labels


def create_training_sample(
    euphemism: str,
    entity: str,
    category: str,
    original_text: str = None
) -> Dict:
    """   """

    #  
    pattern_type = get_type_from_category(category)

    #  
    if category in ['', '']:
        templates = CONTEXT_TEMPLATES + NEGATIVE_TEMPLATES
    else:
        templates = CONTEXT_TEMPLATES

    template = random.choice(templates)

    #  
    text = template.format(euphemism=euphemism)

    # euphemism  
    start = text.find(euphemism)
    end = start + len(euphemism) if start != -1 else -1

    if start == -1:
        #     (   )
        return None

    # BIO  
    char_labels = generate_char_labels(text, start, end)

    return {
        "text": text,
        "patterns": [{
            "text": euphemism,
            "type": pattern_type,
            "start": start,
            "end": end,
            "entity": entity,
            "confidence": 1.0,
            "category": category
        }],
        "has_euphemism": True,
        "source": "namuwiki",
        "char_labels": char_labels,
        "original_context": original_text[:200] if original_text else None
    }


def clean_euphemism(euphemism: str) -> str:
    """euphemism """
    #  
    euphemism = ' '.join(euphemism.split())
    #   (, , ,   )
    euphemism = re.sub(r'[^\w\s---a-zA-Z0-9~!@#$%^&*()_+\-=]', '', euphemism)
    return euphemism.strip()


def is_valid_euphemism(euphemism: str) -> bool:
    """ euphemism """
    if not euphemism:
        return False
    if len(euphemism) < 1 or len(euphemism) > 30:
        return False
    #    
    if euphemism.isdigit():
        return False
    #    
    common_words = {'', '', '', '', '', '', '', '', '', '', '', ''}
    if euphemism in common_words:
        return False
    return True


def convert_namuwiki_data(input_path: str, output_path: str) -> Dict:
    """  """

    #  
    with open(input_path, 'r', encoding='utf-8') as f:
        namuwiki_data = json.load(f)

    print(f"  : {len(namuwiki_data)}")

    # 
    converted_data = []
    seen_euphemisms = set()  #  
    stats = {
        'total': len(namuwiki_data),
        'converted': 0,
        'duplicates': 0,
        'invalid': 0,
        'by_category': {},
        'by_type': {}
    }

    for item in namuwiki_data:
        euphemism = clean_euphemism(item.get('euphemism', ''))
        entity = decode_entity(item.get('entity', ''))
        category = item.get('category', 'unknown')
        original_text = item.get('text', '')

        #  
        if not is_valid_euphemism(euphemism):
            stats['invalid'] += 1
            continue

        #   ( euphemism + entity )
        key = f"{euphemism}|{entity}"
        if key in seen_euphemisms:
            stats['duplicates'] += 1
            continue
        seen_euphemisms.add(key)

        # 
        sample = create_training_sample(euphemism, entity, category, original_text)
        if sample:
            converted_data.append(sample)
            stats['converted'] += 1

            #  
            stats['by_category'][category] = stats['by_category'].get(category, 0) + 1

            #  
            pattern_type = sample['patterns'][0]['type']
            stats['by_type'][pattern_type] = stats['by_type'].get(pattern_type, 0) + 1

    # 
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)

    print(f"\n===   ===")
    print(f": {stats['total']}")
    print(f": {stats['converted']}")
    print(f" : {stats['duplicates']}")
    print(f" : {stats['invalid']}")

    print(f"\n===   ===")
    for type_name, count in sorted(stats['by_type'].items(), key=lambda x: -x[1]):
        print(f"  {type_name}: {count}")

    return stats


if __name__ == '__main__':
    input_path = '/mnt/c/claude_project/NLP/data/raw/namuwiki_euphemisms_20251117_221548.json'
    output_path = '/mnt/c/claude_project/NLP/data/processed/namuwiki_training_data.json'

    stats = convert_namuwiki_data(input_path, output_path)

    #  
    with open(output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"\n===   ===")
    for sample in random.sample(data, min(5, len(data))):
        print(f"\n: {sample['text']}")
        print(f": {sample['patterns'][0]['text']} → {sample['patterns'][0]['entity']}")
        print(f": {sample['patterns'][0]['type']}")
