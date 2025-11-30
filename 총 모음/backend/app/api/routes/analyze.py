"""
Analysis API routes
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List
import uuid
import json
from pathlib import Path
from datetime import datetime

from app.api.models.request import (
    AnalysisRequest,
    BatchAnalysisRequest,
    AnnotationRequest,
    BatchAnnotationRequest
)
from app.api.models.response import (
    AnalysisResponse,
    BatchAnalysisResponse,
    ErrorResponse,
    AnnotationResponse,
    BatchAnnotationResponse,
    AnnotationStatsResponse
)
from app.ml.detector import EuphemismDetector
from app.core.cache import cache
from app.core.config import settings

router = APIRouter(prefix="/analyze", tags=["analysis"])

# Global detector instance
detector = EuphemismDetector()


@router.post(
    "",
    response_model=AnalysisResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Analyze text for euphemisms",
    description="Detect and resolve Korean euphemisms in the provided text"
)
async def analyze_text(request: AnalysisRequest) -> AnalysisResponse:
    """
    Analyze single text for euphemism patterns
    
    Args:
        request: Analysis request with text
        
    Returns:
        Analysis results with detected euphemisms
    """
    try:
        # Check cache first
        cache_key = cache.generate_key(request.text, request.options)
        cached_result = cache.get(cache_key)
        
        if cached_result:
            return AnalysisResponse(**cached_result)
        
        # Perform analysis
        result = detector.detect_and_resolve(request.text)
        
        # Convert to response model
        response = AnalysisResponse(**result)
        
        # Cache result
        cache.set(cache_key, result)
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@router.post(
    "/batch",
    response_model=BatchAnalysisResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Batch analyze multiple texts",
    description="Analyze multiple texts in parallel for efficiency"
)
async def batch_analyze(request: BatchAnalysisRequest) -> BatchAnalysisResponse:
    """
    Batch analyze multiple texts
    
    Args:
        request: Batch analysis request with list of texts
        
    Returns:
        Batch analysis results
    """
    try:
        # Perform batch analysis
        results = detector.batch_analyze(request.texts)
        
        # Calculate totals
        total_detections = sum(r['total_detected'] for r in results)
        total_processing_time = sum(r['processing_time'] for r in results)
        
        # Convert to response models
        response_results = [AnalysisResponse(**r) for r in results]
        
        return BatchAnalysisResponse(
            results=response_results,
            total_texts=len(request.texts),
            total_detections=total_detections,
            processing_time=total_processing_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch analysis failed: {str(e)}"
        )


@router.get(
    "/stats",
    summary="Get detection statistics",
    description="Retrieve statistics about the detection system"
)
async def get_statistics() -> Dict:
    """
    Get detector statistics

    Returns:
        Statistics dictionary
    """
    return detector.get_statistics()


@router.post(
    "/explain",
    summary="Get AI explanation for euphemisms",
    description="Analyze text and provide detailed AI-generated explanations for detected euphemisms"
)
async def explain_euphemisms(request: AnalysisRequest) -> Dict:
    """
    Get detailed AI explanation for euphemisms in text

    This endpoint provides:
    - Pattern detection results
    - Context-aware entity resolution
    - AI-generated explanations (when Gemini API key is configured)

    Args:
        request: Analysis request with text

    Returns:
        Detailed analysis with AI explanations
    """
    try:
        result = detector.explain(request.text)

        # Add helpful metadata
        result['explanation_available'] = detector.gemini_detector is not None
        result['suggestion'] = (
            "AI    GEMINI_API_KEY   ."
            if not detector.gemini_detector else None
        )

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Explanation generation failed: {str(e)}"
        )


@router.post(
    "/context-analyze",
    summary="Context-aware euphemism analysis",
    description="Analyze euphemisms with detailed context keyword matching"
)
async def context_analyze(request: AnalysisRequest) -> Dict:
    """
    Perform context-aware analysis with detailed keyword matching

    Returns analysis including:
    - Detected euphemisms
    - Context keywords found
    - Entity candidates with confidence scores
    - Initial letter matching results

    Args:
        request: Analysis request with text

    Returns:
        Context-enhanced analysis results
    """
    try:
        # Get pattern detections
        pattern_detections = detector.detect_patterns(request.text)

        # Enhance with context
        enhanced = detector._enhance_with_context(pattern_detections, request.text)

        # Resolve entities
        resolved = detector.resolve_entities(enhanced, request.text)

        return {
            'text': request.text,
            'detections': resolved,
            'total_detected': len(resolved),
            'analysis_type': 'context_aware',
            'context_summary': {
                'keywords_analyzed': sum(len(d.get('context_keywords', [])) for d in resolved),
                'candidates_generated': sum(len(d.get('context_candidates', [])) for d in resolved),
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Context analysis failed: {str(e)}"
        )


# Annotation storage path
ANNOTATIONS_PATH = Path("data/annotations")
ANNOTATIONS_PATH.mkdir(parents=True, exist_ok=True)
ANNOTATIONS_FILE = ANNOTATIONS_PATH / "annotations.jsonl"


def save_annotation(annotation: Dict) -> str:
    """Save annotation to JSONL file and return ID"""
    annotation_id = f"ann_{uuid.uuid4().hex[:12]}"
    annotation['annotation_id'] = annotation_id
    annotation['timestamp'] = datetime.utcnow().isoformat()

    # Append to JSONL file
    with open(ANNOTATIONS_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(annotation, ensure_ascii=False) + '\n')

    return annotation_id


def load_annotations() -> List[Dict]:
    """Load all annotations from JSONL file"""
    if not ANNOTATIONS_FILE.exists():
        return []

    annotations = []
    with open(ANNOTATIONS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                annotations.append(json.loads(line))

    return annotations


@router.post(
    "/annotations",
    response_model=AnnotationResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Create manual annotation",
    description="Submit manual annotation for euphemism entity mapping"
)
async def create_annotation(request: AnnotationRequest) -> AnnotationResponse:
    """
    Create a single manual annotation

    Args:
        request: Annotation data

    Returns:
        Annotation response with ID
    """
    try:
        annotation_data = request.dict()
        annotation_id = save_annotation(annotation_data)

        return AnnotationResponse(
            annotation_id=annotation_id,
            text=request.text,
            euphemism_text=request.euphemism_text,
            resolved_entity=request.resolved_entity,
            status="saved"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save annotation: {str(e)}"
        )


@router.post(
    "/annotations/batch",
    response_model=BatchAnnotationResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Create batch annotations",
    description="Submit multiple annotations at once"
)
async def create_batch_annotations(request: BatchAnnotationRequest) -> BatchAnnotationResponse:
    """
    Create multiple annotations in batch

    Args:
        request: Batch annotation request

    Returns:
        Batch annotation response with saved count
    """
    annotation_ids = []
    errors = []
    failed_count = 0

    for idx, annotation in enumerate(request.annotations):
        try:
            annotation_data = annotation.dict()
            if request.dataset_name:
                annotation_data['dataset_name'] = request.dataset_name

            annotation_id = save_annotation(annotation_data)
            annotation_ids.append(annotation_id)

        except Exception as e:
            failed_count += 1
            errors.append({
                "index": idx,
                "error": str(e),
                "text": annotation.text[:50]
            })

    return BatchAnnotationResponse(
        saved_count=len(annotation_ids),
        failed_count=failed_count,
        annotation_ids=annotation_ids,
        errors=errors
    )


@router.get(
    "/annotations/stats",
    response_model=AnnotationStatsResponse,
    summary="Get annotation statistics",
    description="Retrieve statistics about annotation dataset"
)
async def get_annotation_stats() -> AnnotationStatsResponse:
    """
    Get statistics about annotations

    Returns:
        Annotation statistics
    """
    try:
        annotations = load_annotations()

        if not annotations:
            return AnnotationStatsResponse(
                total_annotations=0,
                by_type={},
                by_entity={},
                avg_confidence=0.0,
                annotators=[]
            )

        # Count by type
        by_type = {}
        by_entity = {}
        confidences = []
        annotators = set()

        for ann in annotations:
            # Type counts
            etype = ann.get('euphemism_type', 'unknown')
            by_type[etype] = by_type.get(etype, 0) + 1

            # Entity counts
            entity = ann.get('resolved_entity', 'unknown')
            by_entity[entity] = by_entity.get(entity, 0) + 1

            # Confidences
            if 'confidence' in ann:
                confidences.append(ann['confidence'])

            # Annotators
            if 'annotator_id' in ann:
                annotators.add(ann['annotator_id'])

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return AnnotationStatsResponse(
            total_annotations=len(annotations),
            by_type=by_type,
            by_entity=by_entity,
            avg_confidence=avg_confidence,
            annotators=list(annotators)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get annotation stats: {str(e)}"
        )
