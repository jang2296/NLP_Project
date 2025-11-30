"""
Model Manager for GCS-based model loading

Handles downloading and caching trained models from Google Cloud Storage
for use in Cloud Run deployments.
"""
import os
import logging
from pathlib import Path
from typing import Optional
import subprocess

logger = logging.getLogger(__name__)


class ModelManager:
    """Manage model loading from GCS"""

    def __init__(
        self,
        gcs_bucket: str = "inspiring-list-465300-p6-euphemism-data",
        local_cache_dir: str = "/app/models"
    ):
        """
        Initialize model manager

        Args:
            gcs_bucket: GCS bucket name containing models
            local_cache_dir: Local directory for caching downloaded models
        """
        self.gcs_bucket = gcs_bucket
        self.local_cache_dir = Path(local_cache_dir)
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)

        self.model_version = os.getenv("MODEL_VERSION", "latest")
        self.project_id = os.getenv("GCP_PROJECT_ID", "inspiring-list-465300-p6")

        logger.info(f"Model Manager initialized: bucket={gcs_bucket}, version={self.model_version}")

    def get_model_path(self, model_name: str = "kobert_euphemism.pt") -> Optional[Path]:
        """
        Get local path to model, downloading from GCS if needed

        Args:
            model_name: Name of model file

        Returns:
            Path to local model file, or None if download failed
        """
        # Check if model exists locally
        local_path = self.local_cache_dir / model_name

        if local_path.exists():
            logger.info(f"Model found in cache: {local_path}")
            return local_path

        # Download from GCS
        logger.info(f"Model not in cache, downloading from GCS...")
        success = self._download_from_gcs(model_name, local_path)

        if success and local_path.exists():
            logger.info(f"Model downloaded successfully: {local_path}")
            return local_path
        else:
            logger.warning(f"Failed to download model: {model_name}")
            return None

    def _download_from_gcs(self, model_name: str, local_path: Path) -> bool:
        """
        Download model from GCS using gcloud storage cp

        Args:
            model_name: Model filename in GCS
            local_path: Local destination path

        Returns:
            True if download successful, False otherwise
        """
        try:
            # Construct GCS path
            if self.model_version == "latest":
                gcs_path = f"gs://{self.gcs_bucket}/models/{model_name}"
            else:
                gcs_path = f"gs://{self.gcs_bucket}/models/v{self.model_version}/{model_name}"

            logger.info(f"Downloading from {gcs_path}")

            # Use gcloud storage cp command
            cmd = [
                "gcloud", "storage", "cp",
                gcs_path,
                str(local_path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                logger.info("Download successful")
                return True
            else:
                logger.error(f"Download failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("Download timed out (5 minutes)")
            return False
        except Exception as e:
            logger.error(f"Download error: {e}")
            return False

    def list_available_models(self) -> list:
        """
        List available models in GCS bucket

        Returns:
            List of model filenames available in GCS
        """
        try:
            gcs_dir = f"gs://{self.gcs_bucket}/models/"

            cmd = [
                "gcloud", "storage", "ls",
                gcs_dir
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                models = [
                    line.strip().split('/')[-1]
                    for line in result.stdout.strip().split('\n')
                    if line.strip() and not line.endswith('/')
                ]
                logger.info(f"Found {len(models)} models in GCS")
                return models
            else:
                logger.error(f"Failed to list models: {result.stderr}")
                return []

        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    def clear_cache(self):
        """Clear local model cache"""
        try:
            for model_file in self.local_cache_dir.glob("*.pt"):
                model_file.unlink()
                logger.info(f"Removed cached model: {model_file}")

            logger.info("Model cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

    def get_cache_info(self) -> dict:
        """
        Get information about cached models

        Returns:
            Dictionary with cache statistics
        """
        cached_models = list(self.local_cache_dir.glob("*.pt"))

        info = {
            'cache_dir': str(self.local_cache_dir),
            'cached_models': [m.name for m in cached_models],
            'total_size_mb': sum(m.stat().st_size for m in cached_models) / (1024 * 1024),
            'model_count': len(cached_models)
        }

        return info


# Global model manager instance
model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    """Get global model manager instance"""
    return model_manager
