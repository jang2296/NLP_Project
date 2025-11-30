"""
  

        .
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import List, Dict
import time
import json
from datetime import datetime
import os


class DataCollector:
    """   """

    def __init__(self):
        self.sources = {
            'news': [
                'https://news.naver.com',
                'https://news.daum.net'
            ],
            'community': [
                'https://www.clien.net',
                'https://www.dcinside.com'
            ]
        }

        self.target_keywords = [
            ' ', ' ', ' ',
            'S', 'S', 'K', 'L',
            ' ', ' ',
            '', 'K', 'L'
        ]

        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def collect_samples(self, num_samples: int = 1000) -> List[Dict]:
        """
          

        Args:
            num_samples:   

        Returns:
              
        """
        samples = []

        print(f"  ... : {num_samples}")

        for source_type, urls in self.sources.items():
            print(f"\n{source_type}   ...")

            for url in urls:
                try:
                    texts = self.scrape_texts(url)
                    filtered = self.filter_euphemism_candidates(texts)

                    #    
                    samples_per_source = num_samples // (len(self.sources) * len(urls))
                    samples.extend(filtered[:samples_per_source])

                    print(f"  {url} {len(filtered[:samples_per_source])} ")

                    # Rate limiting
                    time.sleep(1)

                except Exception as e:
                    print(f"  {url}  : {str(e)}")
                    continue

        print(f"\n {len(samples)}   ")
        return samples

    def scrape_texts(self, url: str) -> List[str]:
        """
          

        Args:
            url:  URL

        Returns:
              
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            #    ( ,  )
            paragraphs = soup.find_all(
                ['p', 'div'],
                class_=lambda x: x and any(
                    keyword in x.lower()
                    for keyword in ['content', 'article', 'body', 'text']
                )
            )

            texts = [
                p.get_text().strip()
                for p in paragraphs
                if len(p.get_text().strip()) > 50
            ]

            return texts

        except Exception as e:
            print(f" : {str(e)}")
            return []

    def filter_euphemism_candidates(self, texts: List[str]) -> List[Dict]:
        """
           

        Args:
            texts:  

        Returns:
                
        """
        candidates = []

        for text in texts:
            #  
            if any(keyword in text for keyword in self.target_keywords):
                candidates.append({
                    'text': text,
                    'has_euphemism': True,  #  
                    'source': 'web_scraping',
                    'collected_at': datetime.now().isoformat(),
                    'keywords_found': [
                        kw for kw in self.target_keywords if kw in text
                    ]
                })

        return candidates

    def save_to_csv(self, samples: List[Dict], output_path: str):
        """
          CSV 

        Args:
            samples:  
            output_path:   
        """
        df = pd.DataFrame(samples)

        #  
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n  : {output_path}")
        print(f" {len(df)} ")

    def save_to_json(self, samples: List[Dict], output_path: str):
        """
          JSON 

        Args:
            samples:  
            output_path:   
        """
        #  
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)

        print(f"\n  : {output_path}")
        print(f" {len(samples)} ")


def create_synthetic_samples() -> List[Dict]:
    """
       

    Returns:
          
    """
    synthetic_samples = [
        {
            'text': 'S    .',
            'has_euphemism': True,
            'source': 'synthetic',
            'patterns': [{
                'text': 'S ',
                'type': 'company_anonymized',
                'entity': '',
                'confidence': 1.0
            }]
        },
        {
            'text': '      .',
            'has_euphemism': True,
            'source': 'synthetic',
            'patterns': [{
                'text': ' ',
                'type': 'country_reference',
                'entity': '',
                'confidence': 0.9
            }]
        },
        {
            'text': 'K   .',
            'has_euphemism': True,
            'source': 'synthetic',
            'patterns': [{
                'text': 'K',
                'type': 'person_initial',
                'entity': 'OO',
                'confidence': 0.8
            }]
        },
        {
            'text': 'L S   .',
            'has_euphemism': True,
            'source': 'synthetic',
            'patterns': [
                {
                    'text': 'L',
                    'type': 'initial_company',
                    'entity': 'LG',
                    'confidence': 0.95
                },
                {
                    'text': 'S',
                    'type': 'initial_company',
                    'entity': '',
                    'confidence': 0.9
                }
            ]
        },
        {
            'text': '     .',
            'has_euphemism': True,
            'source': 'synthetic',
            'patterns': [{
                'text': ' ',
                'type': 'country_reference',
                'entity': 'UNKNOWN',
                'confidence': 0.7
            }]
        }
    ]

    return synthetic_samples


def main():
    """  """

    print("===      ===\n")

    #   
    collector = DataCollector()

    #  
    print("1.    ")
    print("2.    ()")

    choice = input("\n (1  2): ").strip()

    if choice == '1':
        #  
        num_samples = int(input("  : "))
        samples = collector.collect_samples(num_samples)

    elif choice == '2':
        #  
        samples = create_synthetic_samples()
        print(f"\n{len(samples)}    ")

    else:
        print(" .")
        return

    # 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV 
    csv_path = f"data/raw/samples_{timestamp}.csv"
    collector.save_to_csv(samples, csv_path)

    # JSON 
    json_path = f"data/raw/samples_{timestamp}.json"
    collector.save_to_json(samples, json_path)

    #  
    print("\n===   ===")
    df = pd.DataFrame(samples)
    print(f"  : {len(df)}")

    if 'source' in df.columns:
        print("\n :")
        print(df['source'].value_counts())

    if 'keywords_found' in df.columns:
        all_keywords = [kw for kws in df['keywords_found'] for kw in kws]
        print(f"\n  : {len(all_keywords)}")


if __name__ == "__main__":
    main()
