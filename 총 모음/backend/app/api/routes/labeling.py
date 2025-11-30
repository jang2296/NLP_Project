"""
  API 

Firestore   ,  ,   .
"""

from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
from google.cloud import firestore
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/labeling", tags=["labeling"])

# Firestore  
db = firestore.Client(project='inspiring-list-465300-p6')

# ============================================================================
# Pydantic Models
# ============================================================================

class TaskRequest(BaseModel):
    """  """
    limit: int = 10
    category: Optional[str] = None
    labeler_id: Optional[str] = None

class LabelSubmission(BaseModel):
    """ """
    task_id: str
    euphemism_text: str
    entity: str
    confidence: float
    is_euphemism: bool
    labeler_id: str
    notes: Optional[str] = None

class BatchLabelSubmission(BaseModel):
    """  """
    labels: List[LabelSubmission]

class TaskAssignment(BaseModel):
    """ """
    task_ids: List[str]
    labeler_id: str

# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/tasks")
async def get_labeling_tasks(
    limit: int = 10,
    category: Optional[str] = None,
    labeler_id: Optional[str] = None
):
    """
      

    -   
    -   
    -      
    """
    try:
        collection_ref = db.collection('labeling_tasks')

        #  :  
        query = collection_ref.where('labeled', '==', False)

        #  
        if category:
            query = query.where('category', '==', category)

        #   (     )
        if labeler_id:
            query = query.where('assigned_to', 'in', [None, labeler_id])
        else:
            query = query.where('assigned_to', '==', None)

        # 
        query = query.limit(limit)

        # 
        docs = query.stream()

        tasks = []
        for doc in docs:
            task_data = doc.to_dict()
            task_data['task_id'] = doc.id
            tasks.append(task_data)

        return {
            'tasks': tasks,
            'count': len(tasks),
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to fetch labeling tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch tasks: {str(e)}")


@router.post("/tasks/assign")
async def assign_tasks(assignment: TaskAssignment):
    """
      

    -      
    -     
    """
    try:
        batch = db.batch()
        assigned_count = 0
        failed_tasks = []

        for task_id in assignment.task_ids:
            task_ref = db.collection('labeling_tasks').document(task_id)
            task_doc = task_ref.get()

            if not task_doc.exists:
                failed_tasks.append({'task_id': task_id, 'reason': 'Task not found'})
                continue

            task_data = task_doc.to_dict()

            #     
            if task_data.get('assigned_to') or task_data.get('labeled'):
                failed_tasks.append({
                    'task_id': task_id,
                    'reason': 'Already assigned or labeled'
                })
                continue

            #  
            batch.update(task_ref, {
                'assigned_to': assignment.labeler_id,
                'assigned_at': datetime.now()
            })
            assigned_count += 1

        #  
        batch.commit()

        return {
            'assigned_count': assigned_count,
            'failed_count': len(failed_tasks),
            'failed_tasks': failed_tasks,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to assign tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to assign tasks: {str(e)}")


@router.post("/labels")
async def submit_label(label: LabelSubmission):
    """
      

    -   
    - labels   
    - labeling_tasks 
    """
    try:
        # 1.   
        task_ref = db.collection('labeling_tasks').document(label.task_id)
        task_doc = task_ref.get()

        if not task_doc.exists:
            raise HTTPException(status_code=404, detail=f"Task {label.task_id} not found")

        task_data = task_doc.to_dict()

        #    
        if task_data.get('labeled'):
            raise HTTPException(status_code=400, detail="Task already labeled")

        # 2.  
        label_data = {
            'task_id': label.task_id,
            'text': task_data['text'],
            'euphemism_text': label.euphemism_text,
            'entity': label.entity,
            'confidence': label.confidence,
            'is_euphemism': label.is_euphemism,
            'labeler_id': label.labeler_id,
            'notes': label.notes,
            'ml_suggestion': task_data.get('ml_suggestion', {}),
            'source': task_data.get('source', 'unknown'),
            'category': task_data.get('category', 'unknown'),
            'timestamp': datetime.now()
        }

        label_ref = db.collection('labels').add(label_data)
        label_id = label_ref[1].id

        # 3.  
        task_ref.update({
            'labeled': True,
            'labeled_at': datetime.now(),
            'label_id': label_id
        })

        return {
            'label_id': label_id,
            'task_id': label.task_id,
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit label: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit label: {str(e)}")


@router.post("/labels/batch")
async def submit_batch_labels(batch_submission: BatchLabelSubmission):
    """
      

    -     
    -    
    """
    try:
        batch = db.batch()
        label_ids = []
        failed_labels = []

        for label in batch_submission.labels:
            try:
                #  
                task_ref = db.collection('labeling_tasks').document(label.task_id)
                task_doc = task_ref.get()

                if not task_doc.exists:
                    failed_labels.append({
                        'task_id': label.task_id,
                        'reason': 'Task not found'
                    })
                    continue

                task_data = task_doc.to_dict()

                if task_data.get('labeled'):
                    failed_labels.append({
                        'task_id': label.task_id,
                        'reason': 'Already labeled'
                    })
                    continue

                #  
                label_data = {
                    'task_id': label.task_id,
                    'text': task_data['text'],
                    'euphemism_text': label.euphemism_text,
                    'entity': label.entity,
                    'confidence': label.confidence,
                    'is_euphemism': label.is_euphemism,
                    'labeler_id': label.labeler_id,
                    'notes': label.notes,
                    'ml_suggestion': task_data.get('ml_suggestion', {}),
                    'timestamp': datetime.now()
                }

                label_ref = db.collection('labels').document()
                batch.set(label_ref, label_data)
                label_ids.append(label_ref.id)

                #  
                batch.update(task_ref, {
                    'labeled': True,
                    'labeled_at': datetime.now(),
                    'label_id': label_ref.id
                })

            except Exception as e:
                failed_labels.append({
                    'task_id': label.task_id,
                    'reason': str(e)
                })

        #  
        batch.commit()

        return {
            'saved_count': len(label_ids),
            'failed_count': len(failed_labels),
            'label_ids': label_ids,
            'failed_labels': failed_labels,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to submit batch labels: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit batch: {str(e)}")


@router.get("/stats")
async def get_labeling_stats(labeler_id: Optional[str] = None):
    """
      

    -     
    -  ,   
    """
    try:
        #   
        total_tasks = len(list(db.collection('labeling_tasks').stream()))

        #    
        labeled_tasks_query = db.collection('labeling_tasks').where('labeled', '==', True)
        labeled_tasks = len(list(labeled_tasks_query.stream()))

        #   
        assigned_tasks_query = db.collection('labeling_tasks').where('assigned_to', '!=', None)
        assigned_tasks = len(list(assigned_tasks_query.stream()))

        #  
        labels_query = db.collection('labels')
        if labeler_id:
            labels_query = labels_query.where('labeler_id', '==', labeler_id)

        labels_docs = labels_query.stream()

        labels_by_category = {}
        labels_by_type = {}
        total_confidence = 0
        label_count = 0

        for label_doc in labels_docs:
            label_data = label_doc.to_dict()

            # 
            category = label_data.get('category', 'unknown')
            labels_by_category[category] = labels_by_category.get(category, 0) + 1

            #  (euphemism vs non-euphemism)
            if label_data.get('is_euphemism'):
                labels_by_type['euphemism'] = labels_by_type.get('euphemism', 0) + 1
            else:
                labels_by_type['non_euphemism'] = labels_by_type.get('non_euphemism', 0) + 1

            #  
            total_confidence += label_data.get('confidence', 0)
            label_count += 1

        avg_confidence = total_confidence / label_count if label_count > 0 else 0

        return {
            'total_tasks': total_tasks,
            'labeled_tasks': labeled_tasks,
            'unlabeled_tasks': total_tasks - labeled_tasks,
            'assigned_tasks': assigned_tasks,
            'progress_percentage': (labeled_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            'labels_by_category': labels_by_category,
            'labels_by_type': labels_by_type,
            'avg_confidence': round(avg_confidence, 3),
            'labeler_id': labeler_id,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to fetch labeling stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {str(e)}")


@router.get("/export")
async def export_labels(
    format: str = "jsonl",
    labeler_id: Optional[str] = None,
    limit: Optional[int] = None
):
    """
      

    - JSONL  CSV 
    -   
    - ML   
    """
    try:
        #  
        query = db.collection('labels')

        if labeler_id:
            query = query.where('labeler_id', '==', labeler_id)

        if limit:
            query = query.limit(limit)

        #  
        labels = []
        for doc in query.stream():
            label_data = doc.to_dict()
            label_data['label_id'] = doc.id

            #  
            if 'timestamp' in label_data:
                label_data['timestamp'] = label_data['timestamp'].isoformat()

            labels.append(label_data)

        if format == "jsonl":
            # JSONL  (ML )
            import json
            jsonl_lines = [json.dumps(label, ensure_ascii=False) for label in labels]
            content = "\n".join(jsonl_lines)

            return {
                'format': 'jsonl',
                'count': len(labels),
                'data': content,
                'timestamp': datetime.now().isoformat()
            }

        elif format == "csv":
            # CSV  ()
            import csv
            from io import StringIO

            output = StringIO()
            if labels:
                writer = csv.DictWriter(output, fieldnames=labels[0].keys())
                writer.writeheader()
                writer.writerows(labels)

            return {
                'format': 'csv',
                'count': len(labels),
                'data': output.getvalue(),
                'timestamp': datetime.now().isoformat()
            }

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export labels: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export labels: {str(e)}")
