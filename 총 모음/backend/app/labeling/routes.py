"""
Labeling API routes for data annotation workflow
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
import json
import uuid
from datetime import datetime
from google.cloud import firestore

from .models import (
    LabelingTask,
    LabelData,
    LabelingResponse,
    LabelingStatistics,
    TaskFilter
)

router = APIRouter(prefix="/api/labeling", tags=["labeling"])

# Firestore client for metadata and progress tracking
db = firestore.Client()


@router.get("/tasks/next", response_model=LabelingTask)
async def get_next_task(
    labeler_id: str,
    category_filter: Optional[str] = None
):
    """
    Get next unlabeled item for labeler

    Returns task with ML auto-labeling suggestions if available
    """
    try:
        # Query Firestore for unlabeled tasks
        tasks_ref = db.collection('labeling_tasks')
        query = tasks_ref.where('labeled', '==', False).where('assigned_to', '==', None)

        if category_filter:
            query = query.where('category', '==', category_filter)

        query = query.limit(1)
        tasks = list(query.stream())

        if not tasks:
            raise HTTPException(status_code=404, detail="No unlabeled tasks available")

        task_doc = tasks[0]
        task_data = task_doc.to_dict()

        # Assign task to labeler
        task_doc.reference.update({
            'assigned_to': labeler_id,
            'assigned_at': datetime.now()
        })

        # Return task with ML suggestions
        return LabelingTask(
            task_id=task_doc.id,
            text=task_data['text'],
            suggested_euphemism=task_data.get('ml_suggestion', {}).get('euphemism'),
            suggested_entity=task_data.get('ml_suggestion', {}).get('entity'),
            confidence=task_data.get('ml_suggestion', {}).get('confidence'),
            context=task_data.get('context'),
            category=task_data.get('category', 'unknown'),
            source=task_data.get('source', 'aihub'),
            created_at=task_data.get('created_at', datetime.now())
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch task: {str(e)}")


@router.post("/submit", response_model=LabelingResponse)
async def submit_label(label_data: LabelData):
    """
    Submit labeled data with validation

    Stores label in Firestore and marks task as completed
    """
    try:
        # Validate task exists
        task_ref = db.collection('labeling_tasks').document(label_data.task_id)
        task = task_ref.get()

        if not task.exists:
            raise HTTPException(status_code=404, detail="Task not found")

        # Store label
        label_ref = db.collection('labels').document(str(uuid.uuid4()))
        label_ref.set({
            'task_id': label_data.task_id,
            'labeler_id': label_data.labeler_id,
            'euphemism_text': label_data.euphemism_text,
            'start_pos': label_data.start_pos,
            'end_pos': label_data.end_pos,
            'euphemism_type': label_data.euphemism_type.value,
            'resolved_entity': label_data.resolved_entity,
            'confidence': label_data.confidence,
            'notes': label_data.notes,
            'labeling_time_seconds': label_data.labeling_time_seconds,
            'timestamp': datetime.now()
        })

        # Mark task as labeled
        task_ref.update({
            'labeled': True,
            'labeled_at': datetime.now(),
            'labeled_by': label_data.labeler_id
        })

        return LabelingResponse(
            success=True,
            task_id=label_data.task_id,
            message="Label submitted successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit label: {str(e)}")


@router.get("/stats", response_model=LabelingStatistics)
async def get_labeling_stats(labeler_id: Optional[str] = None):
    """
    Get labeling statistics

    Returns total progress, per-labeler stats, quality metrics
    """
    try:
        # Count total labeled and pending
        tasks_ref = db.collection('labeling_tasks')

        labeled_count = len(list(tasks_ref.where('labeled', '==', True).stream()))
        pending_count = len(list(tasks_ref.where('labeled', '==', False).stream()))

        # Get per-labeler stats
        labels_ref = db.collection('labels')
        labels = list(labels_ref.stream())

        by_labeler = {}
        by_category = {}

        for label_doc in labels:
            label = label_doc.to_dict()
            labeler = label.get('labeler_id', 'unknown')

            if labeler not in by_labeler:
                by_labeler[labeler] = 0
            by_labeler[labeler] += 1

        # Get category distribution
        for task_doc in tasks_ref.where('labeled', '==', True).stream():
            task = task_doc.to_dict()
            category = task.get('category', 'unknown')

            if category not in by_category:
                by_category[category] = 0
            by_category[category] += 1

        # Calculate estimated completion
        total_tasks = labeled_count + pending_count
        if labeled_count > 0 and total_tasks > 0:
            # Assuming 30s per task, 8 hours/day per labeler
            tasks_per_day_per_labeler = (8 * 3600) / 30  # 960 tasks/day/labeler
            num_labelers = len(by_labeler)

            if num_labelers > 0:
                estimated_days = pending_count / (tasks_per_day_per_labeler * num_labelers)
            else:
                estimated_days = None
        else:
            estimated_days = None

        return LabelingStatistics(
            total_labeled=labeled_count,
            total_pending=pending_count,
            by_labeler=by_labeler,
            by_category=by_category,
            quality_score=None,  # TODO: Implement Cohen's Kappa calculation
            estimated_completion_days=int(estimated_days) if estimated_days else None
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {str(e)}")


@router.post("/skip/{task_id}")
async def skip_task(task_id: str, labeler_id: str):
    """
    Skip a task (unassign it)

    Allows labeler to skip difficult or unclear tasks
    """
    try:
        task_ref = db.collection('labeling_tasks').document(task_id)
        task = task_ref.get()

        if not task.exists:
            raise HTTPException(status_code=404, detail="Task not found")

        # Unassign task
        task_ref.update({
            'assigned_to': None,
            'assigned_at': None,
            'skipped_by': firestore.ArrayUnion([labeler_id]),
            'skipped_at': datetime.now()
        })

        return {"success": True, "message": "Task skipped"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to skip task: {str(e)}")


@router.get("/export/{labeler_id}")
async def export_labels(labeler_id: Optional[str] = None):
    """
    Export labeled data in BIO format for ML training

    Returns JSONL format with tokens and BIO tags
    """
    try:
        labels_ref = db.collection('labels')

        if labeler_id:
            labels_query = labels_ref.where('labeler_id', '==', labeler_id)
        else:
            labels_query = labels_ref

        labels = list(labels_query.stream())

        bio_formatted = []
        for label_doc in labels:
            label = label_doc.to_dict()

            # Get original text from task
            task_ref = db.collection('labeling_tasks').document(label['task_id'])
            task = task_ref.get()

            if task.exists:
                task_data = task.to_dict()
                text = task_data['text']

                # Convert to BIO format
                tokens = list(text)
                tags = ['O'] * len(tokens)

                start = label['start_pos']
                end = label['end_pos']

                if start < len(tags):
                    tags[start] = 'B-EUPHEMISM'
                for i in range(start + 1, min(end, len(tags))):
                    tags[i] = 'I-EUPHEMISM'

                bio_formatted.append({
                    'text': text,
                    'tokens': tokens,
                    'tags': tags,
                    'euphemism_text': label['euphemism_text'],
                    'resolved_entity': label['resolved_entity'],
                    'type': label['euphemism_type']
                })

        return {
            "count": len(bio_formatted),
            "format": "BIO",
            "data": bio_formatted
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export: {str(e)}")
