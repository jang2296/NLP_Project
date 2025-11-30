"""
      

  :
1.   ()
2.  
3.  
4.  
5.  
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

#   Python  
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineRunner:
    """   """

    def __init__(self, config: dict):
        """
        Args:
            config:  
        """
        self.config = config
        self.project_root = project_root
        self.data_dir = self.project_root / 'data'
        self.models_dir = self.project_root / 'models'

        #  
        self._create_directories()

    def _create_directories(self):
        """  """
        directories = [
            self.data_dir / 'raw',
            self.data_dir / 'processed',
            self.models_dir,
            self.project_root / 'logs',
            self.project_root / 'results'
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f" : {directory}")

    def check_requirements(self) -> bool:
        """  """
        logger.info("=" * 70)
        logger.info("   ...")
        logger.info("=" * 70)

        required_packages = [
            'torch',
            'transformers',
            'sentence_transformers',
            'pandas',
            'numpy',
            'beautifulsoup4',
            'requests',
            'konlpy'
        ]

        missing_packages = []

        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                logger.info(f" {package}")
            except ImportError:
                logger.error(f" {package} -  ")
                missing_packages.append(package)

        if missing_packages:
            logger.error("\n  :")
            logger.error(f"pip install {' '.join(missing_packages)}")
            return False

        logger.info("\n    !")
        return True

    def run_data_collection(self) -> str:
        """  """
        if not self.config.get('collect_data', True):
            logger.info("   ( )")
            return None

        logger.info("\n" + "=" * 70)
        logger.info("1:   ")
        logger.info("=" * 70)

        try:
            from scripts.namuwiki_crawler import NamuwikiCrawler

            crawler = NamuwikiCrawler(
                max_samples=self.config.get('max_samples', 1000)
            )

            #  
            data = crawler.collect_all_data()

            # 
            output_file = self.data_dir / 'raw' / f'namuwiki_euphemisms_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            crawler.save_to_json(data, str(output_file))

            logger.info(f"   : {len(data)} ")
            logger.info(f"  : {output_file}")

            return str(output_file)

        except Exception as e:
            logger.error(f"   : {str(e)}")
            raise

    def run_preprocessing(self, input_file: str = None) -> str:
        """  """
        logger.info("\n" + "=" * 70)
        logger.info("2:  ")
        logger.info("=" * 70)

        try:
            from scripts.data_preprocessing import DataPreprocessor

            #   
            if not input_file:
                #   raw  
                raw_files = list(self.data_dir.glob('raw/namuwiki_euphemisms_*.json'))
                if not raw_files:
                    raise FileNotFoundError("     .")
                input_file = str(sorted(raw_files)[-1])

            logger.info(f" : {input_file}")

            #  
            with open(input_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            logger.info(f" : {len(raw_data)}")

            #  
            preprocessor = DataPreprocessor(use_mecab=False)

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

            #   ()
            if self.config.get('augment_data', True):
                logger.info("\n[4/6]   ...")
                data = preprocessor.augment_data(data)
            else:
                logger.info("\n[4/6]   ")

            #   
            if self.config.get('balance_categories', True):
                logger.info("\n[5/6]    ...")
                data = preprocessor.balance_categories(
                    data,
                    target_per_category=self.config.get('target_per_category', 500)
                )
            else:
                logger.info("\n[5/6]    ")

            #  
            logger.info("\n[6/6]   ...")
            stats = preprocessor.analyze_dataset(data)

            #  
            logger.info("\n" + "=" * 70)
            logger.info("  ")
            logger.info("=" * 70)
            logger.info(f"  : {stats['total_samples']}")
            logger.info(f" : {stats['unique_texts']}")
            logger.info(f"  : {stats['unique_euphemisms']}")
            logger.info(f"  : {stats['avg_text_length']:.1f}")

            # 
            processed_file = self.data_dir / 'processed' / 'preprocessed_dataset.json'
            preprocessor.save_processed_data(data, str(processed_file))

            #   
            training_data = self._create_training_format(data)
            training_file = self.data_dir / 'processed' / 'training_dataset.json'

            with open(training_file, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, ensure_ascii=False, indent=2)

            logger.info(f"\n  : {processed_file}")
            logger.info(f"  : {training_file} ({len(training_data)})")

            return str(training_file)

        except Exception as e:
            logger.error(f"  : {str(e)}")
            raise

    def _create_training_format(self, data: list) -> list:
        """    """
        training_data = []

        for item in data:
            text = item['text']
            euphemism = item.get('euphemism', '')

            if not euphemism:
                continue

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

        return training_data

    def run_training(self, training_file: str) -> str:
        """  """
        logger.info("\n" + "=" * 70)
        logger.info("3: KoELECTRA  ")
        logger.info("=" * 70)

        try:
            from scripts.enhanced_train_model import EnhancedEuphemismTrainer

            #   
            with open(training_file, 'r', encoding='utf-8') as f:
                training_data = json.load(f)

            logger.info(f" : {len(training_data)} ")

            #  
            trainer = EnhancedEuphemismTrainer(
                model_name=self.config.get('model_name', 'monologg/koelectra-base-v3-discriminator'),
                max_length=self.config.get('max_length', 128),
                batch_size=self.config.get('batch_size', 16),
                learning_rate=self.config.get('learning_rate', 2e-5)
            )

            #  
            logger.info("\n  ...")
            train_loader, val_loader, test_loader = trainer.prepare_data(
                training_data,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15
            )

            #  
            logger.info("\n  ...")
            trainer.train(
                train_loader,
                val_loader,
                epochs=self.config.get('epochs', 10),
                gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 1),
                early_stopping_patience=self.config.get('early_stopping_patience', 3)
            )

            #  
            logger.info("\n   ...")
            test_metrics = trainer.evaluate(test_loader)

            logger.info("\n" + "=" * 70)
            logger.info("   ")
            logger.info("=" * 70)
            logger.info(f"Precision: {test_metrics['precision']:.4f}")
            logger.info(f"Recall: {test_metrics['recall']:.4f}")
            logger.info(f"F1-Score: {test_metrics['f1']:.4f}")

            #  
            model_path = self.models_dir / f'koelectra_euphemism_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
            trainer.save_model(str(model_path))

            logger.info(f"\n  : {model_path}")

            #   
            metrics_path = self.project_root / 'results' / f'metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(test_metrics, f, ensure_ascii=False, indent=2)

            return str(model_path)

        except Exception as e:
            logger.error(f"  : {str(e)}")
            raise

    def run_inference_test(self, model_path: str):
        """  """
        logger.info("\n" + "=" * 70)
        logger.info("4:   ")
        logger.info("=" * 70)

        try:
            #   
            sys.path.insert(0, str(self.project_root / 'backend'))

            from app.ml.detector import EuphemismDetector

            #  
            detector = EuphemismDetector(model_path=model_path)

            #  
            test_samples = [
                "S   .",
                "     .",
                "K   .",
                "L     .",
                "    ."
            ]

            logger.info("\n   ...\n")

            for idx, text in enumerate(test_samples, 1):
                logger.info(f"[ {idx}] {text}")

                # 3  
                result = detector.detect(text)

                if result['detections']:
                    for detection in result['detections']:
                        logger.info(f"  → : {detection['text']}")
                        logger.info(f"     : {detection['type']}")
                        logger.info(f"     : {detection.get('entity', 'UNKNOWN')}")
                        logger.info(f"     : {detection['confidence']:.2%}")
                else:
                    logger.info("  →   ")

                logger.info("")

            logger.info("   ")

        except Exception as e:
            logger.error(f"   : {str(e)}")
            raise

    def run_full_pipeline(self):
        """  """
        start_time = datetime.now()

        logger.info("\n" + "=" * 70)
        logger.info("     -   ")
        logger.info("=" * 70)
        logger.info(f" : {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        try:
            # 1.  
            if not self.check_requirements():
                logger.error("   .  .")
                return

            # 2.  
            raw_data_file = self.run_data_collection()

            # 3.  
            training_file = self.run_preprocessing(raw_data_file)

            # 4.  
            model_path = self.run_training(training_file)

            # 5.  
            self.run_inference_test(model_path)

            #  
            end_time = datetime.now()
            duration = end_time - start_time

            logger.info("\n" + "=" * 70)
            logger.info("  !")
            logger.info("=" * 70)
            logger.info(f" : {duration}")
            logger.info(f"  : {model_path}")
            logger.info(f" : {training_file}")

        except Exception as e:
            logger.error(f"\n    : {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    """ """
    parser = argparse.ArgumentParser(description='      ')

    #   
    parser.add_argument('--skip-collection', action='store_true',
                        help='  ')
    parser.add_argument('--skip-augmentation', action='store_true',
                        help='  ')

    #  
    parser.add_argument('--max-samples', type=int, default=1000,
                        help='   ')
    parser.add_argument('--target-per-category', type=int, default=500,
                        help='   ')

    #  
    parser.add_argument('--epochs', type=int, default=10,
                        help='  ')
    parser.add_argument('--batch-size', type=int, default=16,
                        help=' ')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                        help='')
    parser.add_argument('--max-length', type=int, default=128,
                        help='  ')

    args = parser.parse_args()

    #   
    config = {
        'collect_data': not args.skip_collection,
        'augment_data': not args.skip_augmentation,
        'max_samples': args.max_samples,
        'target_per_category': args.target_per_category,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_length': args.max_length,
        'model_name': 'monologg/koelectra-base-v3-discriminator',
        'gradient_accumulation_steps': 1,
        'early_stopping_patience': 3,
        'balance_categories': True
    }

    #  
    runner = PipelineRunner(config)
    runner.run_full_pipeline()


if __name__ == "__main__":
    main()
