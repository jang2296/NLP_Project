"""
      

:    "S ", " ", "K" 
     .

BIO :
- O (0): Outside -     
- B-EUPH (1): Beginning -   
- I-EUPH (2): Inside -   
"""

import json
import random
import os
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EuphemismPattern:
    """   """
    pattern: str           #   (: "S ")
    entity_type: str       #  (company, country, person)
    actual_entity: str     #   (: "")
    context_keywords: List[str]  #  


#    
EUPHEMISM_PATTERNS = {
    "company": [
        EuphemismPattern("S ", "company", "", ["", "", "", ""]),
        EuphemismPattern("S ", "company", "", ["", "", ""]),
        EuphemismPattern("S ", "company", "", ["", "OLED", ""]),
        EuphemismPattern("L ", "company", "LG", ["", "TV", "", ""]),
        EuphemismPattern("L ", "company", "LG", ["", "", ""]),
        EuphemismPattern("H ", "company", "", ["", "", ""]),
        EuphemismPattern("H ", "company", "", ["", "", ""]),
        EuphemismPattern("K ", "company", "", ["SUV", "", ""]),
        EuphemismPattern("S ", "company", "", ["", "", "CMO"]),
        EuphemismPattern("N ", "company", "", ["", "", "AI"]),
        EuphemismPattern("K ", "company", "", ["", "", ""]),
        EuphemismPattern("S ", "company", "", ["", "", ""]),
        EuphemismPattern("P ", "company", "", ["", "", ""]),
        EuphemismPattern("L ", "company", "LG", ["", "", ""]),
        EuphemismPattern("H ", "company", "", ["", "", ""]),
        #  
        EuphemismPattern("A", "company", "A", ["", ""]),
        EuphemismPattern("B", "company", "B", ["", ""]),
        EuphemismPattern(" IT A", "company", "/", ["IT", ""]),
        EuphemismPattern("  S", "company", "", ["", ""]),
        EuphemismPattern("  S", "company", "", ["", ""]),
    ],
    "country": [
        EuphemismPattern(" ", "country", "", ["", "", "", ""]),
        EuphemismPattern(" ", "country", "", ["", "", ""]),
        EuphemismPattern(" ", "country", "/", ["", "", ""]),
        EuphemismPattern(" ", "country", "", ["", "", ""]),
        EuphemismPattern(" ", "country", " ", ["", ""]),
        EuphemismPattern("  ", "country", " ", ["", ""]),
        EuphemismPattern(" ", "country", "", ["", "", ""]),
        EuphemismPattern(" ", "country", "", ["", "", ""]),
    ],
    "person": [
        EuphemismPattern("K", "person", "", ["", "", ""]),
        EuphemismPattern("L", "person", "", ["", ""]),
        EuphemismPattern("P", "person", "", ["", ""]),
        EuphemismPattern("", "person", "OO", ["", "", ""]),
        EuphemismPattern("", "person", "OO", ["", ""]),
        EuphemismPattern("", "person", "OO", ["", ""]),
        EuphemismPattern("", "person", "OO", ["", ""]),
        EuphemismPattern("A", "person", "", ["", ""]),
        EuphemismPattern(" ", "person", " ", ["", ""]),
        EuphemismPattern(" ", "person", "", ["", "", ""]),
        EuphemismPattern("  ", "person", " ", ["", "", ""]),
    ],
}

#   
NEWS_TEMPLATES = {
    "company": [
        "{euphemism} {context}   .",
        "{euphemism} {context}   .",
        "{euphemism} {context}    .",
        "{context}  {euphemism}   .",
        "{euphemism} {context}   .",
        "   {euphemism} {context}     .",
        "{euphemism} {context}    .",
        "{context}  {euphemism}  .",
        "{euphemism} {context}    .",
        "{euphemism} {context}    .",
        " {euphemism} {context}   .",
        "{euphemism} {context}   1 .",
        "{context}  {euphemism}   .",
        "{euphemism} {context}     .",
        " {euphemism} {context}     .",
    ],
    "country": [
        "{euphemism} {context}   .",
        "{euphemism} {context}    .",
        "{context}  {euphemism}   .",
        " {euphemism} {context}  .",
        "{euphemism} {context}    .",
        "{context}  {euphemism}  .",
        " {euphemism} {context}   .",
        "{euphemism} {context}   .",
        " {euphemism} {context}    .",
        "{context}  {euphemism}   .",
    ],
    "person": [
        "{euphemism} {context}    .",
        "{euphemism} {context}  .",
        " {euphemism} {context}  .",
        "{context}    {euphemism} .",
        "{euphemism} {context}   .",
        " {euphemism} {context}  .",
        "{euphemism} {context}   .",
        " {euphemism} {context}    .",
        "{context}  {euphemism}  .",
        "{euphemism} {context}    .",
    ],
}

#     
GENERAL_TEMPLATES = [
    " {euphemism}  .",
    "{euphemism}    .",
    " {euphemism}     .",
    " {euphemism}   .",
    "{euphemism}   .",
]

#      (  )
NEGATIVE_SENTENCES = [
    "     .",
    "   .",
    "     .",
    "    .",
    "      .",
    "  .",
    "     .",
    "    .",
    "    .",
    "     .",
    "   .",
    "    .",
    "     .",
    "     .",
    "    .",
    "   .",
    "    .",
    " R&D  .",
    "   .",
    "    .",
]


class TrainingDataGenerator:
    """  """

    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_positive_sample(self, pattern: EuphemismPattern,
                                  templates: List[str]) -> Dict:
        """   (  )"""
        template = random.choice(templates)
        context = random.choice(pattern.context_keywords)

        #  
        sentence = template.format(euphemism=pattern.pattern, context=context)

        #   
        start_idx = sentence.find(pattern.pattern)
        end_idx = start_idx + len(pattern.pattern)

        return {
            "text": sentence,
            "patterns": [{
                "text": pattern.pattern,
                "type": pattern.entity_type,
                "start": start_idx,
                "end": end_idx,
                "entity": pattern.actual_entity,
                "confidence": 1.0
            }],
            "has_euphemism": True,
            "source": "synthetic"
        }

    def generate_negative_sample(self) -> Dict:
        """   (  )"""
        sentence = random.choice(NEGATIVE_SENTENCES)

        return {
            "text": sentence,
            "patterns": [],
            "has_euphemism": False,
            "source": "synthetic"
        }

    def generate_multi_pattern_sample(self) -> Dict:
        """   """
        #      
        entity_types = list(EUPHEMISM_PATTERNS.keys())
        selected_types = random.sample(entity_types, min(2, len(entity_types)))

        patterns_to_use = []
        for et in selected_types:
            patterns_to_use.append(random.choice(EUPHEMISM_PATTERNS[et]))

        #   
        multi_templates = [
            "{p1} {p2}  .",
            "{p1} {context1}  {p2}  .",
            "{p2} {p1} {context1}   .",
            " {p1} {p2}     .",
        ]

        template = random.choice(multi_templates)
        p1 = patterns_to_use[0].pattern
        p2 = patterns_to_use[1].pattern if len(patterns_to_use) > 1 else patterns_to_use[0].pattern
        context1 = random.choice(patterns_to_use[0].context_keywords)

        sentence = template.format(p1=p1, p2=p2, context1=context1)

        #   
        detected_patterns = []
        for pattern in patterns_to_use:
            start_idx = sentence.find(pattern.pattern)
            if start_idx != -1:
                detected_patterns.append({
                    "text": pattern.pattern,
                    "type": pattern.entity_type,
                    "start": start_idx,
                    "end": start_idx + len(pattern.pattern),
                    "entity": pattern.actual_entity,
                    "confidence": 1.0
                })

        return {
            "text": sentence,
            "patterns": detected_patterns,
            "has_euphemism": True,
            "source": "synthetic_multi"
        }

    def generate_dataset(self,
                         num_positive: int = 3000,
                         num_negative: int = 1000,
                         num_multi: int = 500) -> List[Dict]:
        """  """
        print(f"[STATS]    ...")
        print(f"  -  : {num_positive}")
        print(f"  -  : {num_negative}")
        print(f"  -   : {num_multi}")

        dataset = []

        #   
        print("\n[OK]    ...")
        samples_per_type = num_positive // len(EUPHEMISM_PATTERNS)

        for entity_type, patterns in EUPHEMISM_PATTERNS.items():
            templates = NEWS_TEMPLATES.get(entity_type, GENERAL_TEMPLATES)
            samples_per_pattern = samples_per_type // len(patterns)

            for pattern in patterns:
                for _ in range(samples_per_pattern):
                    sample = self.generate_positive_sample(pattern, templates)
                    dataset.append(sample)

        #   
        print("[ERROR]    ...")
        for _ in range(num_negative):
            sample = self.generate_negative_sample()
            dataset.append(sample)

        #    
        print("     ...")
        for _ in range(num_multi):
            sample = self.generate_multi_pattern_sample()
            dataset.append(sample)

        # 
        random.shuffle(dataset)

        print(f"\n  {len(dataset)}   !")
        return dataset

    def create_bio_labels(self, sample: Dict) -> Dict:
        """BIO   ( )"""
        text = sample["text"]
        patterns = sample["patterns"]

        #    
        char_labels = ["O"] * len(text)

        #   BIO  
        for pattern in patterns:
            start = pattern["start"]
            end = pattern["end"]

            #  : B-EUPH
            char_labels[start] = "B-EUPH"

            #  : I-EUPH
            for i in range(start + 1, end):
                char_labels[i] = "I-EUPH"

        sample["char_labels"] = char_labels
        return sample

    def save_dataset(self, dataset: List[Dict],
                     filename: str = "training_data.json"):
        """ """
        # BIO  
        labeled_dataset = [self.create_bio_labels(sample) for sample in dataset]

        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(labeled_dataset, f, ensure_ascii=False, indent=2)

        print(f"   : {output_path}")

        #  
        self._print_statistics(labeled_dataset)

        return output_path

    def save_for_training(self, dataset: List[Dict]):
        """   """
        # BIO  
        labeled_dataset = [self.create_bio_labels(sample) for sample in dataset]

        # 
        random.shuffle(labeled_dataset)

        #  (80/10/10)
        n = len(labeled_dataset)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)

        train_data = labeled_dataset[:train_end]
        val_data = labeled_dataset[train_end:val_end]
        test_data = labeled_dataset[val_end:]

        # 
        for data, name in [(train_data, "train"),
                           (val_data, "val"),
                           (test_data, "test")]:
            path = self.output_dir / f"{name}.json"
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f" {name} : {len(data)} → {path}")

        # JSONL   ( )
        for data, name in [(train_data, "train"),
                           (val_data, "val"),
                           (test_data, "test")]:
            path = self.output_dir / f"{name}.jsonl"
            with open(path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        return train_data, val_data, test_data

    def _print_statistics(self, dataset: List[Dict]):
        """  """
        print("\n[STATS]  :")

        total = len(dataset)
        positive = sum(1 for d in dataset if d["has_euphemism"])
        negative = total - positive

        print(f"  - : {total}")
        print(f"  -  (  ): {positive} ({positive/total*100:.1f}%)")
        print(f"  -  (  ): {negative} ({negative/total*100:.1f}%)")

        #   
        type_counts = {}
        for d in dataset:
            for p in d["patterns"]:
                t = p["type"]
                type_counts[t] = type_counts.get(t, 0) + 1

        print("\n   :")
        for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"    - {t}: {count}")


def main():
    """ """
    print("=" * 60)
    print("[TARGET]       ")
    print("=" * 60)

    generator = TrainingDataGenerator(output_dir="data/processed")

    #  
    dataset = generator.generate_dataset(
        num_positive=4000,  #   (  )
        num_negative=1500,  #   ( )
        num_multi=500       #   
    )

    #   
    generator.save_dataset(dataset, "training_data.json")

    # //  
    print("\n    ...")
    generator.save_for_training(dataset)

    print("\n[OK]    !")
    print("=" * 60)


if __name__ == "__main__":
    main()
