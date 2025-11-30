"""
KoELECTRA-based NER model for euphemism detection

Supports loading trained model from:
1. Local directory (HuggingFace format)
2. Google Cloud Storage (auto-download)
3. Fallback to base model if trained model unavailable
"""
import torch
from transformers import ElectraForTokenClassification, ElectraTokenizer
from typing import List, Dict, Optional
import numpy as np
import os
import logging
import subprocess
import shutil

from app.core.config import settings

logger = logging.getLogger(__name__)


class KoELECTRADetector:
    """KoELECTRA model for Named Entity Recognition of euphemisms"""

    # BIO tags (must match training labels)
    TAG_O = 0  # Outside
    TAG_B = 1  # Begin-EUPHEMISM
    TAG_I = 2  # Inside-EUPHEMISM

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize KoELECTRA model

        Args:
            model_path: Path to fine-tuned model (if None, uses settings.MODEL_PATH)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
        self._using_trained_model = False

        # Determine model path
        self.model_path = model_path or settings.MODEL_PATH

        try:
            # Load tokenizer (always from base model)
            self.tokenizer = ElectraTokenizer.from_pretrained(
                settings.KOELECTRA_MODEL
            )
            logger.info(f"Tokenizer loaded from {settings.KOELECTRA_MODEL}")

            # Try to load trained model
            model_loaded = self._load_trained_model()

            if not model_loaded:
                # Fallback to base model with 3 labels (O, B-EUPH, I-EUPH)
                logger.warning("Using base model (untrained) - detection quality may be limited")
                self.model = ElectraForTokenClassification.from_pretrained(
                    settings.KOELECTRA_MODEL,
                    num_labels=3
                )

            self.model.to(self.device)
            self.model.eval()
            self._is_loaded = True

            logger.info(f"KoELECTRA initialized on {self.device} (trained model: {self._using_trained_model})")

        except Exception as e:
            logger.error(f"KoELECTRA model initialization failed: {e}")
            self._is_loaded = False

    def _load_trained_model(self) -> bool:
        """
        Attempt to load trained model from local path or GCS

        Returns:
            True if trained model loaded successfully
        """
        # Step 1: Check local path
        if self._model_exists_local(self.model_path):
            try:
                self.model = ElectraForTokenClassification.from_pretrained(self.model_path)
                self._using_trained_model = True
                logger.info(f"Loaded trained model from local: {self.model_path}")
                return True
            except Exception as e:
                logger.warning(f"Failed to load local model: {e}")

        # Step 2: Try GCS download if auto-download enabled
        if settings.AUTO_DOWNLOAD_MODEL:
            gcs_path = f"gs://{settings.MODEL_GCS_BUCKET}/{settings.MODEL_GCS_PATH}"
            if self._download_from_gcs(gcs_path, self.model_path):
                try:
                    self.model = ElectraForTokenClassification.from_pretrained(self.model_path)
                    self._using_trained_model = True
                    logger.info(f"Loaded trained model from GCS: {gcs_path}")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to load downloaded model: {e}")

        return False

    def _model_exists_local(self, path: str) -> bool:
        """Check if model directory exists with required files"""
        if not os.path.exists(path):
            return False

        # HuggingFace model directory should contain config.json
        config_path = os.path.join(path, "config.json")
        if os.path.exists(config_path):
            return True

        # Also check for pytorch_model.bin or model.safetensors
        model_files = ["pytorch_model.bin", "model.safetensors"]
        for model_file in model_files:
            if os.path.exists(os.path.join(path, model_file)):
                return True

        return False

    def _download_from_gcs(self, gcs_path: str, local_path: str) -> bool:
        """
        Download model from Google Cloud Storage using Python SDK

        Args:
            gcs_path: GCS path (gs://bucket/path)
            local_path: Local destination path

        Returns:
            True if download successful
        """
        try:
            from google.cloud import storage

            # Parse GCS path
            # gs://bucket-name/path/to/model -> bucket_name, path/to/model
            if gcs_path.startswith("gs://"):
                gcs_path = gcs_path[5:]
            parts = gcs_path.split("/", 1)
            bucket_name = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""

            logger.info(f"Downloading model from GCS: gs://{bucket_name}/{prefix}")

            # Create local directory
            os.makedirs(local_path, exist_ok=True)

            # Initialize GCS client (uses default credentials in Cloud Run)
            client = storage.Client()
            bucket = client.bucket(bucket_name)

            # List and download all blobs with the prefix
            blobs = list(bucket.list_blobs(prefix=prefix))
            if not blobs:
                logger.warning(f"No files found in GCS path: gs://{bucket_name}/{prefix}")
                return False

            downloaded_count = 0
            for blob in blobs:
                # Get relative path (remove prefix)
                relative_path = blob.name[len(prefix):].lstrip("/")
                if not relative_path:
                    continue

                local_file_path = os.path.join(local_path, relative_path)

                # Create subdirectories if needed
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True) if os.path.dirname(relative_path) else None

                # Download file
                blob.download_to_filename(local_file_path)
                downloaded_count += 1
                logger.info(f"Downloaded: {relative_path}")

            logger.info(f"Model downloaded successfully: {downloaded_count} files to {local_path}")
            return downloaded_count > 0

        except ImportError:
            logger.warning("google-cloud-storage not installed - GCS download unavailable")
            return False
        except Exception as e:
            logger.warning(f"GCS download error: {e}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return self._is_loaded

    def predict(self, text: str) -> List[Dict]:
        """
        Predict euphemisms in text using NER

        Args:
            text: Input text to analyze

        Returns:
            List of detected euphemisms with positions
        """
        if not self._is_loaded:
            return []

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Extract entities
        entities = self._extract_entities(
            text,
            predictions[0].cpu().numpy(),
            inputs['input_ids'][0].cpu().numpy()
        )
        
        return entities
    
    def _extract_entities(
        self,
        text: str,
        predictions: np.ndarray,
        input_ids: np.ndarray
    ) -> List[Dict]:
        """
        Extract euphemism entities from BIO predictions
        
        Args:
            text: Original text
            predictions: Predicted labels
            input_ids: Token IDs
            
        Returns:
            List of entity dictionaries
        """
        entities = []
        current_entity = None
        current_tokens = []
        
        # Decode tokens and build entities
        for idx, (pred, token_id) in enumerate(zip(predictions, input_ids)):
            # Skip special tokens
            if token_id in [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id]:
                if current_entity:
                    entities.append(self._finalize_entity(current_entity, current_tokens, text))
                    current_entity = None
                    current_tokens = []
                continue
            
            token = self.tokenizer.decode([token_id])
            
            if pred == self.TAG_B:  # Begin new entity
                if current_entity:
                    entities.append(self._finalize_entity(current_entity, current_tokens, text))
                current_entity = {'start_idx': idx}
                current_tokens = [token]
            elif pred == self.TAG_I and current_entity:  # Continue entity
                current_tokens.append(token)
            elif pred == self.TAG_O and current_entity:  # End entity
                entities.append(self._finalize_entity(current_entity, current_tokens, text))
                current_entity = None
                current_tokens = []
        
        # Handle remaining entity
        if current_entity:
            entities.append(self._finalize_entity(current_entity, current_tokens, text))
        
        return entities
    
    def _finalize_entity(
        self,
        entity: Dict,
        tokens: List[str],
        original_text: str
    ) -> Dict:
        """
        Finalize entity with text and position
        
        Args:
            entity: Entity dictionary with start_idx
            tokens: List of tokens
            original_text: Original input text
            
        Returns:
            Complete entity dictionary
        """
        # Reconstruct text
        entity_text = ''.join(tokens).replace('##', '')
        
        # Find position in original text
        start_pos = original_text.find(entity_text)
        end_pos = start_pos + len(entity_text) if start_pos != -1 else -1
        
        return {
            'type': 'ml_detected',
            'text': entity_text,
            'start': start_pos,
            'end': end_pos,
            'confidence': 0.85,  # Default confidence for ML
            'method': 'koelectra_ner'
        }
    
    def batch_predict(self, texts: List[str]) -> List[List[Dict]]:
        """
        Batch prediction for multiple texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of detection results for each text
        """
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results
