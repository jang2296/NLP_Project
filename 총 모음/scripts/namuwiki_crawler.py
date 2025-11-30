"""
    

 , ,       .
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import os
from urllib.parse import urljoin, quote
import logging

#  
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NamuwikiCrawler:
    """ """

    BASE_URL = "https://namu.wiki"

    #    (,   )
    TARGET_PAGES = {
        # /
        '': '/w/',
        '': '/w/',
        '': '/w/',
        ' ': '/w/%20',
        '': '/w/',
        '': '/w/',
        ' ': '/w/%20',
        '': '/w/',
        '': '/w/',
        '': '/w/',

        #   
        '': '/w/',
        '': '/w/',
        ' ': '/w/',
        ' ': '/w/',

        #   
        '': '/w/',
        ' ': '/w/%20',
        '': '/w/',
        '': '/w/',
        '': '/w/',
        '': '/w/',

        # /
        '': '/w/',
        '': '/w/',
        ' ': '/w/%20',
        ' ': '/w/%20',
        '': '/w/',

        # /
        '': '/w/',
        ' ': '/w//',
        ' ': '/w//',
        ' ': '/w/',
        ' ': '/w/',
        ' ': '/w/',
        ' ': '/w/',
        '': '/w/',
        '': '/w/',

        # / 
        ' ': '/w/%20',
        ' ': '/w/%20',
        ' ': '/w/%20',
        ' ': '/w/%20',
        'e ': '/w/e',
        ' ': '/w/%20',
        ' ': '/w/',
        ' ': '/w/',

        #  
        '': '/w/',
        '': '/w/',
        '': '/w/',
        '': '/w/',
        '': '/w/',
        '': '/w/',
        '': '/w/',
        '': '/w/',
        '': '/w/',

        # / 
        '10 ': '/w/10',
        '20 ': '/w/20',
        ' ': '/w/%20',
        ' ': '/w/%20',
        ' ': '/w/%20',
    }

    #   
    EUPHEMISM_PATTERNS = [
        r'[A-Z-]\s*\s*(|||)',  # S 
        r'[A-Z-]\s*',  # S, K
        r'[A-Z-]\s*',  # L
        r'[A-Z-]\s*',  # K
        r'\s*',  #  
        r'\s*(|)',  #  
        r'\s*',  # 
        r'[-]{1,2}\s*',  # 
        r'|',  #  
    ]

    def __init__(self, output_dir: str = "./data/raw"):
        """
        Args:
            output_dir:   
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
        })

        self.collected_data = []
        self.visited_urls = set()

    def crawl_all(self, max_pages_per_category: int = 50) -> List[Dict]:
        """
           

        Args:
            max_pages_per_category:    

        Returns:
              
        """
        logger.info(f"   - {len(self.TARGET_PAGES)} ")

        for category, page_path in self.TARGET_PAGES.items():
            logger.info(f"\n{'='*60}")
            logger.info(f": {category}  ")
            logger.info(f"{'='*60}")

            try:
                self._crawl_category(category, page_path, max_pages_per_category)
                time.sleep(2)  # Rate limiting
            except Exception as e:
                logger.error(f"{category}  : {e}")
                continue

        logger.info(f"\n {len(self.collected_data)}   ")
        return self.collected_data

    def _crawl_category(
        self,
        category: str,
        page_path: str,
        max_pages: int
    ):
        """ """
        url = urljoin(self.BASE_URL, page_path)

        if url in self.visited_urls:
            return

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            self.visited_urls.add(url)

            #   
            data = self._parse_page(soup, category, url)

            if data:
                self.collected_data.extend(data)
                logger.info(f"  → {len(data)}  ")

            #    ()
            if len(self.visited_urls) < max_pages:
                links = self._extract_related_links(soup)
                for link in links[:5]:  #  5
                    if len(self.visited_urls) >= max_pages:
                        break
                    time.sleep(1)
                    self._crawl_category(category, link, max_pages)

        except requests.RequestException as e:
            logger.error(f"  ({url}): {e}")
        except Exception as e:
            logger.error(f"  ({url}): {e}")

    def _parse_page(
        self,
        soup: BeautifulSoup,
        category: str,
        url: str
    ) -> List[Dict]:
        """  """
        data = []

        #   class  body   
        # h1-h6, p, li    
        body = soup.find('body')

        if not body:
            logger.warning(f"body    : {url}")
            return data

        #      ( body)
        paragraphs = body.find_all(['p', 'li', 'dd', 'dt'])

        for para in paragraphs:
            text = para.get_text().strip()

            #       
            if len(text) < 5:
                continue

            # ,     
            if '[]' in text or text.startswith('↑') or text.startswith('['):
                continue

            #      
            # ,       
            euphemisms = self._detect_euphemisms(text)

            #  ,   
            if euphemisms or para.name == 'li':
                entities = self._extract_entities(text, para)

                #      
                if euphemisms:
                    for euphemism in euphemisms:
                        item = {
                            'text': text,
                            'euphemism': euphemism,
                            'entity': entities.get(euphemism, 'UNKNOWN'),
                            'category': category,
                            'source': url,
                            'collected_at': datetime.now().isoformat(),
                            'has_euphemism': True
                        }
                        data.append(item)
                #      / 
                else:
                    item = {
                        'text': text,
                        'euphemism': text.split()[0] if text.split() else text[:20],  #   
                        'entity': 'UNKNOWN',
                        'category': category,
                        'source': url,
                        'collected_at': datetime.now().isoformat(),
                        'has_euphemism': False  #  
                    }
                    data.append(item)

        #    
        tables = body.find_all('table')
        for table in tables:
            table_data = self._parse_table(table, category, url)
            data.extend(table_data)

        return data

    def _detect_euphemisms(self, text: str) -> List[str]:
        """   """
        detected = []

        for pattern in self.EUPHEMISM_PATTERNS:
            matches = re.finditer(pattern, text)
            for match in matches:
                euphemism = match.group().strip()
                if euphemism not in detected:
                    detected.append(euphemism)

        #   
        # "", "***", "XXX"   
        if re.search(r'[]{2,}|[*]{2,}|[X]{2,}', text):
            detected.append(re.search(r'[]{2,}|[*]{2,}|[X]{2,}', text).group())

        return detected

    def _extract_entities(
        self,
        text: str,
        element: BeautifulSoup
    ) -> Dict[str, str]:
        """     """
        entities = {}

        #   
        links = element.find_all('a')
        for link in links:
            link_text = link.get_text().strip()
            href = link.get('href', '')

            #    
            if '/w/' in href:
                entity_name = href.split('/w/')[-1]
                entity_name = entity_name.split('?')[0]  #  
                entity_name = entity_name.replace('_', ' ')

                #    
                for euphemism in self._detect_euphemisms(text):
                    if euphemism not in entities:
                        entities[euphemism] = entity_name

        #    
        # : "S ()"
        bracket_matches = re.finditer(r'([^(]+)\(([^)]+)\)', text)
        for match in bracket_matches:
            potential_euph = match.group(1).strip()
            potential_entity = match.group(2).strip()

            if any(re.search(pattern, potential_euph) for pattern in self.EUPHEMISM_PATTERNS):
                entities[potential_euph] = potential_entity

        return entities

    def _parse_table(
        self,
        table: BeautifulSoup,
        category: str,
        url: str
    ) -> List[Dict]:
        """ /  """
        data = []

        rows = table.find_all('tr')

        for row in rows:
            cells = row.find_all(['td', 'th'])

            if len(cells) >= 2:
                #  : /,  :  
                euphemism = cells[0].get_text().strip()
                entity = cells[1].get_text().strip()

                if euphemism and entity and len(euphemism) < 50:
                    #    
                    if any(re.search(pattern, euphemism) for pattern in self.EUPHEMISM_PATTERNS):
                        item = {
                            'text': f"{euphemism}() {entity}() .",
                            'euphemism': euphemism,
                            'entity': entity,
                            'category': category,
                            'source': url,
                            'collected_at': datetime.now().isoformat(),
                            'has_euphemism': True,
                            'confidence': 0.95  #    
                        }
                        data.append(item)

        return data

    def _extract_related_links(self, soup: BeautifulSoup) -> List[str]:
        """   """
        links = []

        #    
        content = soup.find('div', class_='wiki-content') or soup.find('article')

        if content:
            for link in content.find_all('a', href=True):
                href = link.get('href')

                #   
                if href.startswith('/w/'):
                    #   
                    if not any(x in href for x in [':', '', '', '', '']):
                        links.append(href)

        return list(set(links))  #  

    def save_to_json(self, filename: Optional[str] = None):
        """JSON  """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"namuwiki_euphemisms_{timestamp}.json"

        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.collected_data, f, ensure_ascii=False, indent=2)

        logger.info(f"\n  : {filepath}")
        logger.info(f" {len(self.collected_data)} ")

        return filepath

    def save_to_csv(self, filename: Optional[str] = None):
        """CSV  """
        import pandas as pd

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"namuwiki_euphemisms_{timestamp}.csv"

        filepath = os.path.join(self.output_dir, filename)

        df = pd.DataFrame(self.collected_data)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')

        logger.info(f"CSV  : {filepath}")

        return filepath

    def get_statistics(self) -> Dict:
        """ """
        if not self.collected_data:
            return {}

        import pandas as pd
        df = pd.DataFrame(self.collected_data)

        stats = {
            'total_items': len(df),
            'categories': df['category'].value_counts().to_dict(),
            'unique_euphemisms': df['euphemism'].nunique(),
            'unique_entities': df['entity'].nunique(),
            'unknown_entities': len(df[df['entity'] == 'UNKNOWN']),
            'average_text_length': df['text'].str.len().mean()
        }

        return stats


def create_training_dataset(
    raw_data_path: str,
    output_path: str,
    min_confidence: float = 0.7
):
    """
         

    Args:
        raw_data_path:  JSON  
        output_path:  JSON  
        min_confidence:   (   )
    """
    logger.info(f"  : {raw_data_path}")

    with open(raw_data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    training_data = []

    for item in raw_data:
        #  
        confidence = item.get('confidence', 0.8)
        if confidence < min_confidence:
            continue

        if item['entity'] == 'UNKNOWN':
            continue  #    

        #   
        text = item['text']
        euphemism = item['euphemism']
        entity = item['entity']

        #   
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
                'type': item['category'],
                'entity': entity,
                'confidence': confidence
            }]
        }

        training_data.append(training_item)

    # 
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)

    logger.info(f"   : {len(training_data)}")
    logger.info(f" : {output_path}")

    return training_data


def main():
    """  """
    print("="*70)
    print("    ")
    print("="*70)

    #  
    crawler = NamuwikiCrawler(output_dir="./data/raw")

    #  
    print("\n ...")
    print("(:       )\n")

    data = crawler.crawl_all(max_pages_per_category=20)

    # 
    json_path = crawler.save_to_json()
    csv_path = crawler.save_to_csv()

    #  
    print("\n" + "="*70)
    print(" ")
    print("="*70)
    stats = crawler.get_statistics()

    print(f"\n  : {stats.get('total_items', 0)}")
    print(f"   : {stats.get('unique_euphemisms', 0)}")
    print(f"  : {stats.get('unique_entities', 0)}")
    print(f"  : {stats.get('unknown_entities', 0)}")

    print("\n :")
    for category, count in stats.get('categories', {}).items():
        print(f"  - {category}: {count}")

    #   
    print("\n   ...")
    training_path = "./data/processed/training_dataset.json"
    os.makedirs("./data/processed", exist_ok=True)

    create_training_dataset(
        raw_data_path=json_path,
        output_path=training_path,
        min_confidence=0.7
    )

    print("\n" + "="*70)
    print("!")
    print("="*70)
    print(f"\n : {json_path}")
    print(f" : {training_path}")


if __name__ == "__main__":
    main()
