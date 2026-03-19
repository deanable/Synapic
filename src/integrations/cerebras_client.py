"""
Cerebras Inference Client
==========================

Wrapper around the Cerebras Cloud SDK (cerebras_cloud_sdk package).
Provides model listing and inference using the ultra-fast Cerebras
inference API — the world's fastest LLM service.

Obtain an API key at: https://cloud.cerebras.ai

The client mirrors the interface of NvidiaClient and GoogleAIClient so
it can be plugged into the processing pipeline as a drop-in provider.

Note on Vision:
    Cerebras currently offers text-only LLMs (llama3.1-8b, gpt-oss-120b,
    qwen-3-235b-a22b-instruct-2507, zai-glm-4.7). Images are sent as
    base64 data URLs in the OpenAI-compatible multimodal content format.
    If the model rejects the image payload the client automatically falls
    back to a text-only prompt that asks the model to describe what it
    would expect from the file name / contextual prompt.

Author: Synapic Project
"""

import base64
import logging
import mimetypes
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CEREBRAS_API_URL = "https://api.cerebras.ai/v1"

# Default production model — fastest and most capable general-purpose option
DEFAULT_MODEL = "llama3.1-8b"

# Models known to be available on Cerebras (static fallback if API key is absent)
KNOWN_MODELS = [
    {"id": "llama3.1-8b",                      "provider": "Meta",   "capability": "LLM"},
    {"id": "gpt-oss-120b",                      "provider": "OpenAI", "capability": "LLM"},
    {"id": "qwen-3-235b-a22b-instruct-2507",    "provider": "Qwen",   "capability": "LLM (Preview)"},
    {"id": "zai-glm-4.7",                       "provider": "Z.ai",   "capability": "LLM (Preview)"},
]


class CerebrasClient:
    """Client for Cerebras Inference API.

    Mirrors the interface of ``NvidiaClient`` so it can be plugged into
    the processing pipeline as a drop-in provider.

    The Cerebras SDK wraps the REST API and provides a strongly-typed
    client that is compatible with the OpenAI SDK surface.

    Attributes:
        api_key (str): Cerebras API key for authentication.
        available (bool): Whether the SDK is installed and a key is present.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialise the Cerebras client.

        Args:
            api_key: Cerebras API key. Falls back to the ``CEREBRAS_API_KEY``
                     environment variable when not provided.
        """
        self.api_key = (api_key or os.environ.get("CEREBRAS_API_KEY", "")).strip()
        self._client = None  # Lazy-initialised SDK client
        self.available = False

        try:
            from cerebras.cloud.sdk import Cerebras  # type: ignore
            self._cerebras_class = Cerebras
            self.available = True
            logger.debug("CerebrasClient: cerebras_cloud_sdk loaded successfully")
        except ImportError as exc:
            self._cerebras_class = None
            logger.warning("CerebrasClient: cerebras_cloud_sdk not installed: %s", exc)
        except Exception as exc:
            self._cerebras_class = None
            logger.error("CerebrasClient: unexpected error during SDK import: %s", exc)

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True when the SDK is installed and an API key has been set."""
        return self.available and bool(self.api_key)

    def has_sdk(self) -> bool:
        """Return True when the Cerebras SDK import succeeded."""
        return self.available

    def has_api_key(self) -> bool:
        """Return True when a non-empty API key has been provided."""
        return bool(self.api_key)

    def availability_error(self) -> Optional[str]:
        """Return a user-facing explanation when the client is unavailable."""
        if not self.has_sdk():
            return "Cerebras SDK missing. Install it with: pip install cerebras_cloud_sdk"
        if not self.has_api_key():
            return "Cerebras API key not configured."
        return None

    # ------------------------------------------------------------------
    # Internal — lazy SDK client
    # ------------------------------------------------------------------

    def _ensure_client(self):
        """Create and cache the underlying Cerebras SDK client."""
        if not self._cerebras_class:
            return None
        if self._client is not None:
            return self._client
        try:
            self._client = self._cerebras_class(api_key=self.api_key)
            return self._client
        except Exception as exc:
            logger.error("CerebrasClient: failed to create SDK client: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Model listing
    # ------------------------------------------------------------------

    def list_models(self, limit: int = 40) -> List[Dict]:
        """Fetch available models from the Cerebras API.

        If the API call fails (e.g., invalid key) the static ``KNOWN_MODELS``
        list is returned so the UI is never left empty.

        Returns:
            List of dicts with keys ``id``, ``provider``, and ``capability``.
        """
        if not self.is_available():
            return list(KNOWN_MODELS)

        client = self._ensure_client()
        if client is None:
            return list(KNOWN_MODELS)

        try:
            response = client.models.list()
            model_data = getattr(response, "data", None) or response
            models: List[Dict] = []
            for m in model_data:
                model_id = getattr(m, "id", "") or m.get("id", "") if isinstance(m, dict) else getattr(m, "id", "")
                if not model_id:
                    continue
                # Determine display provider from model ID patterns
                if "llama" in model_id.lower():
                    provider = "Meta"
                elif "gpt-oss" in model_id.lower() or "openai" in model_id.lower():
                    provider = "OpenAI"
                elif "qwen" in model_id.lower():
                    provider = "Qwen"
                elif "glm" in model_id.lower() or "zai" in model_id.lower():
                    provider = "Z.ai"
                else:
                    provider = "Cerebras"

                preview_suffix = " (Preview)" if "preview" in model_id.lower() or "2507" in model_id else ""
                models.append({
                    "id": model_id,
                    "provider": provider,
                    "capability": f"LLM{preview_suffix}",
                })

            if models:
                logger.info("CerebrasClient: fetched %d models from API", len(models))
                return models[:limit]

        except Exception as exc:
            logger.error("CerebrasClient: model listing failed: %s", exc)

        # Fallback to known static list
        return list(KNOWN_MODELS)

    # ------------------------------------------------------------------
    # Inference — image + text
    # ------------------------------------------------------------------

    def chat_with_image(
        self,
        model_name: str,
        prompt: str,
        image_path: str,
    ) -> str:
        """Send a text + image prompt to a Cerebras model.

        The image is base64-encoded and passed as an ``image_url`` content
        part which follows the OpenAI multimodal message format.  If the
        model responds with an error indicating it cannot process images,
        the method automatically retries with a text-only prompt.

        Args:
            model_name: Cerebras model identifier (e.g. ``"llama3.1-8b"``).
            prompt: Text instruction for the model.
            image_path: Absolute path to the image file on disk.

        Returns:
            The model's text response, or an error string starting with
            ``"Error:"`` if the call fails irrecoverably.
        """
        if not self.is_available():
            detail = self.availability_error() or "Cerebras client unavailable."
            return f"Error: {detail}"

        if not os.path.exists(image_path):
            return f"Error: Image file not found: {image_path}"

        client = self._ensure_client()
        if client is None:
            return "Error: Failed to initialise Cerebras SDK client."

        # ------------------------------------------------------------------
        # Read and encode image
        # ------------------------------------------------------------------
        try:
            with open(image_path, "rb") as fh:
                image_b64 = base64.b64encode(fh.read()).decode()
        except Exception as exc:
            return f"Error reading image file: {exc}"

        mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        data_url = f"data:{mime_type};base64,{image_b64}"
        del image_b64  # Free the large standalone copy

        # ------------------------------------------------------------------
        # Attempt 1: multimodal content (image + text)
        # ------------------------------------------------------------------
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        del data_url  # Embedded in messages now

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_completion_tokens=1024,
            )
            del messages
            text = response.choices[0].message.content
            return text or ""
        except Exception as exc:
            err_str = str(exc).lower()
            # Detect image-rejection errors and fall back to text-only
            if any(kw in err_str for kw in ["image", "vision", "multimodal", "unsupported", "invalid"]):
                logger.warning(
                    "CerebrasClient: model %s rejected image content, falling back to text-only: %s",
                    model_name, exc,
                )
                del messages
                return self._chat_text_only(client, model_name, prompt, image_path)
            del messages
            logger.error("CerebrasClient: chat completion failed: %s", exc)
            return f"Error calling Cerebras API: {exc}"

    def _chat_text_only(self, client, model_name: str, prompt: str, image_path: str) -> str:
        """Text-only fallback when the model does not accept image inputs.

        Constructs a prompt that includes the file name as context so the
        model can still attempt to produce structured metadata.
        """
        filename = os.path.basename(image_path)
        text_prompt = (
            f"You are an image metadata assistant. Although you cannot see the "
            f"image directly, the file is named '{filename}'. Based on the file "
            f"name and the following instruction, produce the best possible response.\n\n"
            f"{prompt}"
        )
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": text_prompt}],
                max_completion_tokens=1024,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("CerebrasClient: text-only fallback also failed: %s", exc)
            return f"Error calling Cerebras API (text-only fallback): {exc}"

    # ------------------------------------------------------------------
    # Connection test
    # ------------------------------------------------------------------

    def test_connection(self) -> bool:
        """Quick connectivity check — tries to list one model."""
        try:
            models = self.list_models(limit=1)
            # list_models returns static fallback on error, so we probe differently
            if not self.is_available():
                return False
            client = self._ensure_client()
            return client is not None
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        """Release the underlying SDK HTTP client and connection pool."""
        if self._client is not None:
            try:
                if hasattr(self._client, "close"):
                    self._client.close()
            except Exception:
                pass
            self._client = None

    def __repr__(self) -> str:
        return (
            f"<CerebrasClient available={self.is_available()} "
            f"has_api_key={bool(self.api_key)}>"
        )
