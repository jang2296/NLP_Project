"""
     

3      :
1.    
2. ML   
3.    
4.    
5.   (, )
"""

import sys
import time
import json
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

#   
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'backend'))

from app.ml.patterns import PatternMatcher
from app.ml.inference import EntityResolver
from app.ml.detector import EuphemismDetector


class IntegrationTester:
    """  """

    def __init__(self, model_path: str = None):
        """
        Args:
            model_path:    (   )
        """
        self.model_path = model_path
        self.pattern_matcher = PatternMatcher()
        self.entity_resolver = EntityResolver()
        self.detector = None

        #   
        self.results = {
            'pattern_matching': {},
            'ml_detection': {},
            'entity_resolution': {},
            'integration': {},
            'performance': {}
        }

    def load_test_data(self) -> List[Dict]:
        """    """
        #   
        test_samples = [
            #  
            {
                'text': 'S   .',
                'expected': [{'type': 'company_anonymized', 'entity': ''}]
            },
            {
                'text': 'L     .',
                'expected': [{'type': 'initial_company', 'entity': 'LG'}]
            },
            {
                'text': 'SK 5G  .',
                'expected': []  #    
            },

            #  
            {
                'text': '     .',
                'expected': [{'type': 'country_reference', 'entity': ''}]
            },
            {
                'text': '    .',
                'expected': [{'type': 'country_reference', 'entity': ''}]
            },

            #  
            {
                'text': 'K   .',
                'expected': [{'type': 'person_initial', 'entity': 'UNKNOWN'}]
            },

            #  
            {
                'text': '    .',
                'expected': [{'type': 'media_anonymous', 'entity': 'UNKNOWN'}]
            },

            #  
            {
                'text': '    .',
                'expected': [{'type': 'government_reference', 'entity': ''}]
            },

            #  
            {
                'text': 'S  L    .',
                'expected': [
                    {'type': 'company_anonymized', 'entity': ''},
                    {'type': 'initial_company', 'entity': 'LG'},
                    {'type': 'country_reference', 'entity': ''}
                ]
            },

            #   (  )
            {
                'text': '   .',
                'expected': []
            }
        ]

        return test_samples

    def test_pattern_matching(self) -> Dict:
        """   """
        print("\n" + "=" * 70)
        print("1.    ")
        print("=" * 70)

        test_data = self.load_test_data()
        total = len(test_data)
        passed = 0
        failed = []

        for idx, sample in enumerate(test_data, 1):
            text = sample['text']
            expected = sample['expected']

            #  
            detected = self.pattern_matcher.detect_patterns(text)

            #  
            if len(detected) == len(expected):
                #   
                detected_types = set(d['type'] for d in detected)
                expected_types = set(e['type'] for e in expected)

                if detected_types == expected_types:
                    passed += 1
                    print(f"   [{idx}/{total}] : {text[:40]}...")
                else:
                    failed.append({
                        'text': text,
                        'expected': expected_types,
                        'detected': detected_types
                    })
                    print(f"   [{idx}/{total}] : {text[:40]}...")
                    print(f"      : {expected_types}")
                    print(f"      : {detected_types}")
            else:
                failed.append({
                    'text': text,
                    'expected_count': len(expected),
                    'detected_count': len(detected)
                })
                print(f"   [{idx}/{total}] : {text[:40]}...")
                print(f"       : {len(expected)},  : {len(detected)}")

        accuracy = (passed / total) * 100
        print(f"\n: {passed}/{total} ({accuracy:.1f}%)")

        self.results['pattern_matching'] = {
            'total': total,
            'passed': passed,
            'failed': len(failed),
            'accuracy': accuracy,
            'failed_cases': failed
        }

        return self.results['pattern_matching']

    def test_entity_resolution(self) -> Dict:
        """   """
        print("\n" + "=" * 70)
        print("2.    ")
        print("=" * 70)

        test_cases = [
            {
                'euphemism': 'S ',
                'context': '    ',
                'expected': ''
            },
            {
                'euphemism': 'L',
                'context': ' TV ',
                'expected': 'LG'
            },
            {
                'euphemism': ' ',
                'context': '   ',
                'expected': ''
            },
            {
                'euphemism': ' ',
                'context': '    ',
                'expected': ''
            },
            {
                'euphemism': '',
                'context': '   ',
                'expected': ''
            }
        ]

        total = len(test_cases)
        passed = 0
        failed = []

        for idx, case in enumerate(test_cases, 1):
            result = self.entity_resolver.resolve_entity(
                case['euphemism'],
                case['context']
            )

            if result['entity'] == case['expected']:
                passed += 1
                print(f"   [{idx}/{total}] {case['euphemism']} → {result['entity']} (: {result['confidence']:.2%})")
            else:
                failed.append({
                    'euphemism': case['euphemism'],
                    'expected': case['expected'],
                    'got': result['entity'],
                    'confidence': result['confidence']
                })
                print(f"   [{idx}/{total}] {case['euphemism']}")
                print(f"      : {case['expected']}")
                print(f"      : {result['entity']} (: {result['confidence']:.2%})")

        accuracy = (passed / total) * 100
        print(f"\n: {passed}/{total} ({accuracy:.1f}%)")

        self.results['entity_resolution'] = {
            'total': total,
            'passed': passed,
            'failed': len(failed),
            'accuracy': accuracy,
            'failed_cases': failed
        }

        return self.results['entity_resolution']

    def test_integration(self) -> Dict:
        """   """
        print("\n" + "=" * 70)
        print("3.    ")
        print("=" * 70)

        #  
        if self.detector is None:
            print("\n  ...")
            self.detector = EuphemismDetector(model_path=self.model_path)
            print("   ")

        test_samples = [
            "S      .",
            "L H   .",
            "     .",
            "K    ."
        ]

        total_detections = 0
        total_time = 0

        print("\n  :")
        for idx, text in enumerate(test_samples, 1):
            print(f"\n[{idx}] {text}")

            #   ( )
            start_time = time.time()
            result = self.detector.detect(text)
            elapsed_time = (time.time() - start_time) * 1000  # ms

            total_time += elapsed_time

            if result['detections']:
                total_detections += len(result['detections'])
                for detection in result['detections']:
                    print(f"   {detection['text']}")
                    print(f"    → : {detection['type']}")
                    print(f"    → : {detection.get('entity', 'UNKNOWN')}")
                    print(f"    → : {detection['confidence']:.2%}")
                    print(f"    → : {detection['method']}")
            else:
                print("     ")

            print(f"  ⏱  : {elapsed_time:.2f}ms")

        avg_time = total_time / len(test_samples)

        print(f"\n:")
        print(f"   : {len(test_samples)}")
        print(f"    : {total_detections}")
        print(f"    : {avg_time:.2f}ms")
        print(f"    : {total_time:.2f}ms")

        self.results['integration'] = {
            'total_texts': len(test_samples),
            'total_detections': total_detections,
            'avg_time_ms': avg_time,
            'total_time_ms': total_time
        }

        return self.results['integration']

    def test_performance(self) -> Dict:
        """ """
        print("\n" + "=" * 70)
        print("4.  ")
        print("=" * 70)

        #  
        if self.detector is None:
            self.detector = EuphemismDetector(model_path=self.model_path)

        #    (100)
        benchmark_samples = [
            "S   .",
            "    .",
            "L   .",
            "   .",
            "K   ."
        ] * 20  # 100 

        print(f"\n : {len(benchmark_samples)}")
        print(" ...", end='', flush=True)

        #  
        start_time = time.time()
        results = []

        for sample in benchmark_samples:
            result = self.detector.detect(sample)
            results.append(result)

        total_time = time.time() - start_time
        avg_time = (total_time / len(benchmark_samples)) * 1000  # ms
        throughput = len(benchmark_samples) / total_time  # texts/sec

        print(" !")

        print(f"\n :")
        print(f"    : {total_time:.2f}")
        print(f"    : {avg_time:.2f}ms")
        print(f"  : {throughput:.2f} texts/sec")

        #   
        target_latency = 200  # ms
        latency_ok = avg_time < target_latency

        print(f"\n  :")
        print(f"  {'' if latency_ok else ''}   < 200ms: {avg_time:.2f}ms")

        self.results['performance'] = {
            'total_samples': len(benchmark_samples),
            'total_time_sec': total_time,
            'avg_time_ms': avg_time,
            'throughput_per_sec': throughput,
            'target_met': latency_ok
        }

        return self.results['performance']

    def generate_report(self) -> str:
        """  """
        print("\n" + "=" * 70)
        print("   ")
        print("=" * 70)

        #  
        pm = self.results['pattern_matching']
        print(f"\n1.   ")
        print(f"   : {pm['accuracy']:.1f}% ({pm['passed']}/{pm['total']})")

        #  
        er = self.results['entity_resolution']
        print(f"\n2.   ")
        print(f"   : {er['accuracy']:.1f}% ({er['passed']}/{er['total']})")

        #  
        it = self.results['integration']
        print(f"\n3.  ")
        print(f"     : {it['avg_time_ms']:.2f}ms")
        print(f"     : {it['total_detections']}")

        #  
        perf = self.results['performance']
        print(f"\n4.  ")
        print(f"     : {perf['avg_time_ms']:.2f}ms")
        print(f"   : {perf['throughput_per_sec']:.2f} texts/sec")
        print(f"    : {' ' if perf['target_met'] else ' '}")

        #  
        overall_accuracy = (pm['accuracy'] + er['accuracy']) / 2
        print(f"\n" + "=" * 70)
        print(f"  : {overall_accuracy:.1f}%")
        print(f" F1-Score 85% {' ' if overall_accuracy >= 85 else ' '}")
        print(f"  200ms {' ' if perf['target_met'] else ' '}")
        print("=" * 70)

        # JSON 
        report_path = project_root / 'results' / f'integration_test_report_{int(time.time())}.json'
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        print(f"\n  : {report_path}")

        return str(report_path)

    def run_all_tests(self):
        """  """
        print("\n" + "=" * 70)
        print("    -   ")
        print("=" * 70)

        start_time = time.time()

        try:
            # 1.   
            self.test_pattern_matching()

            # 2.   
            self.test_entity_resolution()

            # 3.  
            self.test_integration()

            # 4.  
            self.test_performance()

            # 5.  
            report_path = self.generate_report()

            # 
            elapsed = time.time() - start_time
            print(f"\n  : {elapsed:.2f}")
            print("   ")

        except Exception as e:
            print(f"\n    : {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """ """
    import argparse

    parser = argparse.ArgumentParser(description='    ')
    parser.add_argument('--model', type=str,
                        help='   (   )')

    args = parser.parse_args()

    #   
    if args.model:
        model_path = args.model
    else:
        models_dir = project_root / 'models'
        model_files = list(models_dir.glob('koelectra_euphemism_*.pt'))

        if not model_files:
            print("    .")
            print(" setup_and_train.py   .")
            sys.exit(1)

        model_path = str(sorted(model_files)[-1])
        print(f"  : {model_path}")

    #  
    tester = IntegrationTester(model_path=model_path)
    tester.run_all_tests()


if __name__ == "__main__":
    main()
