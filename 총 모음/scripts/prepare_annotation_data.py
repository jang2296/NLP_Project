#!/usr/bin/env python3
"""
Prepare training data for annotation workflow

This script prepares data from GCS for the annotation API, creating sample
datasets that can be used for manual labeling and entity mapping validation.
"""
import json
import random
from pathlib import Path
from typing import List, Dict
import argparse


def load_gcs_training_data(file_path: str) -> List[Dict]:
    """Load training data from processed JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'data' in data:
        return data['data']
    else:
        raise ValueError(f"Unexpected data format in {file_path}")


def extract_annotation_candidates(data: List[Dict], max_samples: int = 1000) -> List[Dict]:
    """Extract candidate texts for annotation"""
    candidates = []

    for item in data:
        text = item.get('text', '')
        if not text or len(text) < 10:
            continue

        # Check if item already has euphemisms marked
        euphemisms = item.get('euphemisms', [])
        if not euphemisms:
            continue

        # Create annotation candidate
        for euphemism in euphemisms:
            candidate = {
                'text': text,
                'euphemism_text': euphemism.get('pattern', ''),
                'start_pos': euphemism.get('start', 0),
                'end_pos': euphemism.get('end', 0),
                'euphemism_type': euphemism.get('type', 'unknown'),
                'resolved_entity': euphemism.get('entity', 'UNKNOWN'),
                'confidence': euphemism.get('confidence', 0.0),
                'source': item.get('source', 'unknown'),
                'needs_review': euphemism.get('confidence', 0.0) < 0.8
            }
            candidates.append(candidate)

    # Sample if too many
    if len(candidates) > max_samples:
        candidates = random.sample(candidates, max_samples)

    return candidates


def create_annotation_batches(candidates: List[Dict], batch_size: int = 100) -> List[List[Dict]]:
    """Split candidates into annotation batches"""
    batches = []
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        batches.append(batch)

    return batches


def generate_annotation_api_payloads(batches: List[List[Dict]], output_dir: Path):
    """Generate JSON files ready for annotation API upload"""
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, batch in enumerate(batches):
        # Convert to API format
        api_payload = {
            'annotations': [
                {
                    'text': item['text'],
                    'euphemism_text': item['euphemism_text'],
                    'start_pos': item['start_pos'],
                    'end_pos': item['end_pos'],
                    'euphemism_type': item['euphemism_type'],
                    'resolved_entity': item['resolved_entity'],
                    'confidence': item['confidence'],
                    'context_notes': f"Source: {item['source']}, Needs review: {item['needs_review']}"
                }
                for item in batch
            ],
            'dataset_name': f"gcs_import_batch_{idx + 1}"
        }

        output_file = output_dir / f"annotation_batch_{idx + 1}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(api_payload, f, ensure_ascii=False, indent=2)

        print(f"[OK] Created {output_file} with {len(batch)} annotations")


def generate_statistics(candidates: List[Dict], output_file: Path):
    """Generate statistics about annotation candidates"""
    stats = {
        'total_candidates': len(candidates),
        'by_type': {},
        'by_entity': {},
        'needs_review_count': 0,
        'avg_confidence': 0.0,
        'sources': {}
    }

    confidences = []
    for candidate in candidates:
        # Type distribution
        etype = candidate['euphemism_type']
        stats['by_type'][etype] = stats['by_type'].get(etype, 0) + 1

        # Entity distribution
        entity = candidate['resolved_entity']
        stats['by_entity'][entity] = stats['by_entity'].get(entity, 0) + 1

        # Needs review
        if candidate['needs_review']:
            stats['needs_review_count'] += 1

        # Confidence
        confidences.append(candidate['confidence'])

        # Sources
        source = candidate['source']
        stats['sources'][source] = stats['sources'].get(source, 0) + 1

    stats['avg_confidence'] = sum(confidences) / len(confidences) if confidences else 0.0

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n[STATS] Statistics:")
    print(f"  Total candidates: {stats['total_candidates']}")
    print(f"  Needs review: {stats['needs_review_count']}")
    print(f"  Avg confidence: {stats['avg_confidence']:.2f}")
    print(f"  Types: {len(stats['by_type'])}")
    print(f"  Entities: {len(stats['by_entity'])}")


def main():
    parser = argparse.ArgumentParser(description="Prepare annotation data from GCS")
    parser.add_argument(
        '--input',
        default='data/processed/final_training_dataset.json',
        help='Input training data file'
    )
    parser.add_argument(
        '--output-dir',
        default='data/annotations/batches',
        help='Output directory for annotation batches'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=1000,
        help='Maximum number of annotation samples'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for API uploads'
    )

    args = parser.parse_args()

    print("[START] Preparing annotation data...")

    # Load data
    print(f" Loading data from {args.input}")
    data = load_gcs_training_data(args.input)
    print(f"  Loaded {len(data)} items")

    # Extract candidates
    print(f"[SEARCH] Extracting annotation candidates...")
    candidates = extract_annotation_candidates(data, max_samples=args.max_samples)
    print(f"  Found {len(candidates)} candidates")

    # Create batches
    print(f"[BATCH] Creating batches (size={args.batch_size})...")
    batches = create_annotation_batches(candidates, batch_size=args.batch_size)
    print(f"  Created {len(batches)} batches")

    # Generate API payloads
    output_dir = Path(args.output_dir)
    print(f" Generating API payloads...")
    generate_annotation_api_payloads(batches, output_dir)

    # Generate statistics
    stats_file = output_dir / "annotation_stats.json"
    generate_statistics(candidates, stats_file)
    print(f"[OK] Statistics saved to {stats_file}")

    print(f"\n Done! Ready to upload to annotation API:")
    print(f"   POST /api/analyze/annotations/batch")
    print(f"   Files: {output_dir}/annotation_batch_*.json")


if __name__ == "__main__":
    main()
