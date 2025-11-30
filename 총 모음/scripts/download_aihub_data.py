"""
AI Hub     

AI Hub     .
SNS, ,       .
"""

import os
import requests
import json
import time
from typing import List, Dict
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AIHubDataDownloader:
    """AI Hub  """

    def __init__(self, output_dir: str = "./data/raw/aihub"):
        """
        Args:
            output_dir:   
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # AI Hub API  (   API  )
        self.api_key = os.getenv('AIHUB_API_KEY', '')

        #   
        self.datasets = {
            '  ': {
                'url': 'https://aihub.or.kr/aidata/87',  #  URL  
                'description': 'SNS, ,     '
            },
            'SNS  ': {
                'url': 'https://aihub.or.kr/aidata/30718',
                'description': 'SNS    '
            },
        }

    def download_sample_data(self) -> List[Dict]:
        """
        AI Hub API    
          AI Hub API   

        Returns:
               
        """
        logger.warning("AI Hub API     .")
        logger.warning("   AIHUB_API_KEY   .")

        #    ( AI Hub )
        sample_texts = [
            "   ",
            "  ",
            "    ",
            "? ",
            "  ",
            " ",
            " ?",
            " ?",
            "  ",
            "  ",
            "S   ",
            "  ",
            "K ",
            "L  ",
            "  ",
            "  ",
            "  ",
            " ",
            " ",
            " ",
        ]

        data = []
        for text in sample_texts:
            item = {
                'text': text,
                'source': 'sample_data',
                'platform': 'unknown',
                'is_colloquial': True,
                'collected_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            data.append(item)

        return data

    def download_from_aihub(self) -> List[Dict]:
        """
        AI Hub API    

        Note:
               AI Hub   API   
            https://aihub.or.kr    

        Returns:
              
        """
        if not self.api_key:
            logger.warning("AI Hub API   .")
            return self.download_sample_data()

        # TODO:  AI Hub API  
        # headers = {
        #     'Authorization': f'Bearer {self.api_key}',
        #     'Content-Type': 'application/json'
        # }
        # response = requests.get(url, headers=headers)
        # ...

        logger.info("AI Hub API    ")
        return self.download_sample_data()

    def save_data(self, data: List[Dict], filename: str = "aihub_colloquial.json"):
        """ """
        filepath = self.output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"  : {filepath}")
        logger.info(f" {len(data)} ")

        return str(filepath)

    def run(self):
        """ """
        logger.info("="*70)
        logger.info("AI Hub   ")
        logger.info("="*70)

        #  
        data = self.download_from_aihub()

        #  
        filepath = self.save_data(data)

        #  
        logger.info("\n" + "="*70)
        logger.info(" ")
        logger.info("="*70)
        logger.info(f"  : {len(data)}")
        logger.info(f" : {filepath}")
        logger.info("\n  :")
        for i, item in enumerate(data[:5], 1):
            logger.info(f"  {i}. {item['text']}")

        logger.info("\n" + "="*70)
        logger.info("!")
        logger.info("="*70)
        logger.info("\n:    .")
        logger.info(" AI Hub  :")
        logger.info("1. https://aihub.or.kr  ")
        logger.info("2.     ")
        logger.info("3. API  ")
        logger.info("4.  : export AIHUB_API_KEY='your-api-key'")
        logger.info("="*70)

        return filepath


if __name__ == '__main__':
    downloader = AIHubDataDownloader()
    downloader.run()
