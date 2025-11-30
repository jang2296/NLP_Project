"""
   

     .
"""

import sys
import argparse
from pathlib import Path
import json

#   
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'backend'))

from app.ml.detector import EuphemismDetector


def test_samples(detector: EuphemismDetector, samples: list):
    """  """
    print("\n" + "=" * 70)
    print("   ")
    print("=" * 70 + "\n")

    for idx, text in enumerate(samples, 1):
        print(f"[ {idx}] {text}")
        print("-" * 70)

        #  
        result = detector.detect(text)

        if result['detections']:
            for detection in result['detections']:
                print(f"    :")
                print(f"    • : {detection['text']}")
                print(f"    • : {detection['type']}")
                print(f"    •  : {detection.get('entity', 'UNKNOWN')}")
                print(f"    • : {detection['confidence']:.2%}")
                print(f"    •  : {detection['method']}")
                print()
        else:
            print("     \n")

    print("=" * 70)


def interactive_mode(detector: EuphemismDetector):
    """ """
    print("\n" + "=" * 70)
    print("   (: 'quit'  'exit')")
    print("=" * 70 + "\n")

    while True:
        text = input("  : ").strip()

        if text.lower() in ['quit', 'exit', '']:
            print(" .")
            break

        if not text:
            continue

        print("\n ...\n")
        result = detector.detect(text)

        if result['detections']:
            print(f" {len(result['detections'])}  :\n")
            for i, detection in enumerate(result['detections'], 1):
                print(f"  [{i}] {detection['text']}")
                print(f"      → : {detection.get('entity', 'UNKNOWN')}")
                print(f"      → : {detection['type']}")
                print(f"      → : {detection['confidence']:.2%}")
                print()
        else:
            print("   \n")

        print("-" * 70 + "\n")


def batch_test(detector: EuphemismDetector, input_file: str, output_file: str = None):
    """ """
    print(f"\n  : {input_file}")

    #   
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict) and 'texts' in data:
        texts = data['texts']
    elif isinstance(data, list):
        texts = data
    else:
        texts = [data]

    results = []
    for idx, text in enumerate(texts, 1):
        print(f" : {idx}/{len(texts)}", end='\r')

        result = detector.detect(text)
        results.append({
            'text': text,
            'detections': result['detections']
        })

    print(f"\n: {len(results)}  ")

    #  
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f" : {output_file}")

    #  
    total_detections = sum(len(r['detections']) for r in results)
    texts_with_detections = sum(1 for r in results if r['detections'])

    print("\n" + "=" * 70)
    print("  ")
    print("=" * 70)
    print(f" : {len(results)}")
    print(f"  : {texts_with_detections} ({texts_with_detections/len(results)*100:.1f}%)")
    print(f"  : {total_detections}")
    print(f"  : {total_detections/len(results):.2f}")


def main():
    parser = argparse.ArgumentParser(description='    ')

    parser.add_argument('--model', type=str,
                        help='   (   )')
    parser.add_argument('--interactive', action='store_true',
                        help=' ')
    parser.add_argument('--batch', type=str,
                        help='   ')
    parser.add_argument('--output', type=str,
                        help='    ')

    args = parser.parse_args()

    #   
    if args.model:
        model_path = args.model
    else:
        #   
        models_dir = project_root / 'models'
        model_files = list(models_dir.glob('koelectra_euphemism_*.pt'))

        if not model_files:
            print("    .")
            print(" setup_and_train.py   .")
            sys.exit(1)

        model_path = str(sorted(model_files)[-1])
        print(f"  : {model_path}")

    #  
    print("\n  ...")
    detector = EuphemismDetector(model_path=model_path)
    print("   ")

    #  
    if args.interactive:
        #  
        interactive_mode(detector)

    elif args.batch:
        #  
        batch_test(detector, args.batch, args.output)

    else:
        #   
        default_samples = [
            "S   .",
            "     .",
            "K   .",
            "L     .",
            "    .",
            "    .",
            "H     .",
            "  .",
            "   .",
            "   ."
        ]

        test_samples(detector, default_samples)

        print("\n[TIP]  : python test_inference.py --interactive")
        print("[TIP]  : python test_inference.py --batch input.json --output results.json")


if __name__ == "__main__":
    main()
