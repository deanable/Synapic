"""
Image Processing and Metadata Management
========================================

This module handles all image processing operations, including:
- Image validation and safety checks
- AI model result parsing and tag extraction
- EXIF and IPTC metadata writing
- File system metadata persistence

The module supports three primary AI model tasks:
1. Image Classification: Generates keyword tags automatically
2. Zero-Shot Classification: Matches images to user-defined categories
3. Image-to-Text: Generates descriptive captions

Key Components:
- validate_image(): Pre-processing safety checks for image files
- extract_tags_from_result(): Parses AI model outputs into structured tags
- write_metadata(): Writes tags to image EXIF/IPTC fields
- write_metadata_with_retry(): Wrapper with retry logic for reliability

Metadata Formats:
- IPTC: Keywords written to 2:25 (Keywords) field
- EXIF: Description written to UserComment field
- Category: Written to both IPTC 2:15 (Category) and ObjectName

Dependencies:
- PIL (Pillow): Image loading and validation
- piexif: EXIF metadata manipulation
- iptcinfo3: IPTC metadata manipulation
- src.core.config: Application constants and thresholds

Author: Dean
"""

# ============================================================================
# IMPORTS
# ============================================================================

import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple, Any
from queue import Queue
from PIL import Image, UnidentifiedImageError
import piexif
from iptcinfo3 import IPTCInfo

from src.core import config
from src.utils.json_utils import extract_dict_from_text

# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================


class ImageValidationError(Exception):
    """Raised when image validation fails."""

    pass


# ============================================================================
# IMAGE VALIDATION
# ============================================================================


def validate_image(image_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Validate that an image file can be opened and processed.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        >>> valid, error = validate_image(Path("test.jpg"))
        >>> if valid:
        ...     print("Image is valid")
    """
    try:
        if not image_path.exists():
            return False, "File does not exist"

        if not image_path.is_file():
            return False, "Path is not a file"

        if image_path.stat().st_size == 0:
            return False, "File is empty"

        if image_path.stat().st_size > config.MAX_IMAGE_SIZE_MB * 1024 * 1024:
            return False, f"File exceeds {config.MAX_IMAGE_SIZE_MB}MB limit"

        with Image.open(image_path) as img:
            img.verify()

        with Image.open(image_path) as img:
            img.load()

        return True, None

    except UnidentifiedImageError:
        return False, "Cannot identify image file"
    except PermissionError:
        return False, "Permission denied"
    except Exception as e:
        return False, f"Validation failed: {str(e)}"


# ============================================================================
# METADATA WRITING
# ============================================================================


def write_metadata_with_retry(
    image_path: Path,
    category: str,
    keywords: List[str],
    description: str,
    q: Queue,
    max_retries: int = 3,
    retry_delay: float = 0.5,
) -> bool:
    """
    Write metadata to image with retry logic.

    Args:
        image_path: Path to the image file
        category: Category to write
        keywords: Keywords to write
        description: Description/Caption to write
        q: Queue for status messages
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
    """
    for attempt in range(max_retries):
        try:
            return write_metadata(image_path, category, keywords, description, q)
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(
                    f"Metadata write attempt {attempt + 1} failed for {image_path.name}: {e}. Retrying..."
                )
                time.sleep(retry_delay)
            else:
                logging.error(
                    f"All metadata write attempts failed for {image_path.name}"
                )
                return False
    return False


def write_metadata(
    image_path: Path,
    category: str,
    keywords: List[str],
    description: str,
    q: Optional[Queue] = None,
) -> bool:
    """
    Write category, keywords, and description to the image's IPTC and EXIF metadata.

    This function writes AI-generated tags to both IPTC and EXIF metadata fields for
    maximum compatibility across different platforms and applications:

    IPTC Fields:
    - Object Name (2:05): Category/label
    - Caption/Abstract (2:120): Description text
    - Keywords (2:25): List of keyword tags

    EXIF Fields:
    - XPTitle/XPSubject: Category (Windows-compatible)
    - ImageDescription: Description (cross-platform)
    - XPComment: Description (Windows-compatible)
    - XPKeywords: Semicolon-separated keywords (Windows-compatible)

    Args:
        image_path: Path to the image file to write metadata to
        category: Category or label text (empty string = no category)
        keywords: List of keyword strings to add (merges with existing)
        description: Descriptive caption text (empty string = no description)
        q: Optional queue for logging progress messages

    Returns:
        True if at least one metadata format (IPTC or EXIF) was written successfully,
        False if both formats failed

    Note:
        - Existing keywords are preserved and merged with new ones (no duplicates)
        - Temporary files created by iptcinfo3 are cleaned up automatically
        - Failures in one format don't prevent writing to the other
    """
    iptc_success = False
    exif_success = False

    try:
        logging.info(f"Writing IPTC metadata to {image_path.name}")
        info = IPTCInfo(image_path, force=True)

        if category:
            info["object name"] = category

        if description:
            # IPTC Caption/Abstract
            info["caption/abstract"] = description

        if keywords:
            existing_keywords = [
                k.decode("utf-8") if isinstance(k, bytes) else k
                for k in (info["keywords"] or [])
            ]
            # Use set for O(1) lookups instead of O(n)
            existing_set = set(existing_keywords)
            for k in keywords:
                if k not in existing_set:
                    existing_keywords.append(k)
                    existing_set.add(k)
            info["keywords"] = existing_keywords

        info.save()
        iptc_success = True
        logging.debug(f"IPTC metadata written successfully for {image_path.name}")

        # Cleanup temp file if created
        temp_file = image_path.with_name(image_path.name + "~")
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception as tmp_err:
                logging.warning(f"Failed to delete temp file {temp_file}: {tmp_err}")

    except Exception as e:
        logging.exception(f"Failed to write IPTC metadata for {image_path.name}")

    try:
        logging.info(f"Writing EXIF metadata to {image_path.name}")
        exif_dict = piexif.load(str(image_path))

        if category:
            # Map Category/Label to Windows 'Subject' and 'Title'
            exif_dict["0th"][piexif.ImageIFD.XPSubject] = category.encode("utf-16le")
            exif_dict["0th"][piexif.ImageIFD.XPTitle] = category.encode("utf-16le")

        if description:
            # EXIF ImageDescription - standard ascii for cross-platform compatibility
            exif_dict["0th"][piexif.ImageIFD.ImageDescription] = description.encode(
                "utf-8"
            )
            # Map long AI Caption/Description to Windows 'Comments'
            exif_dict["0th"][piexif.ImageIFD.XPComment] = description.encode("utf-16le")

        if keywords:
            existing_keywords_bytes = exif_dict["0th"].get(
                piexif.ImageIFD.XPKeywords, b""
            )

            # Piexif can sometimes return tuple of ints instead of bytes
            if isinstance(existing_keywords_bytes, tuple):
                try:
                    existing_keywords_bytes = bytes(existing_keywords_bytes)
                except Exception:
                    existing_keywords_bytes = b""

            existing_keywords_str = (
                existing_keywords_bytes.decode("utf-16le").rstrip("\x00")
                if existing_keywords_bytes
                else ""
            )
            existing_keywords = (
                existing_keywords_str.split(";") if existing_keywords_str else []
            )

            # Use set for O(1) lookups instead of O(n)
            existing_set = set(existing_keywords)
            for k in keywords:
                if k not in existing_set:
                    existing_keywords.append(k)
                    existing_set.add(k)

            exif_dict["0th"][piexif.ImageIFD.XPKeywords] = ";".join(
                existing_keywords
            ).encode("utf-16le")

        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, str(image_path))
        exif_success = True
        logging.debug(f"EXIF metadata written successfully for {image_path.name}")

    except Exception as e:
        logging.exception(f"Failed to write EXIF metadata for {image_path.name}")

    return iptc_success or exif_success


def process_single_image(
    image_path: Path,
    model: Any,
    model_task: str,
    categories: List[str],
    keywords: List[str],
    q: Queue,
) -> Tuple[bool, Optional[str]]:
    # ... (Logic mostly delegated to batch usually, but let's update if used)
    # NOTE: This function seems less used than the batch worker, but we should update it if it's called.
    # For now, I'll leave it as is or update it if I see it's used.
    # Actually, looking at gui_workers, it IS NOT used by the main batch loop.
    # But I should update extract_tags_from_result below.
    return False, "Function deprecated in favor of batch pipeline"


# ============================================================================
# AI MODEL RESULT PARSING
# ============================================================================


def to_title_case(text: str) -> str:
    """
    Convert text to proper Title Case.

    Handles:
    - Regular words: "hello world" -> "Hello World"
    - Compound words: "blue-sky" -> "Blue-Sky"
    - Underscores: "blue_sky" -> "Blue_Sky"
    - All-caps preserved: "3D", "AI", "JPEG", "HDR" stay as-is
    - Mixed case: "iPhone" stays as "iPhone"

    Args:
        text: Input string to convert

    Returns:
        Title-cased string
    """
    if not text or not isinstance(text, str):
        return text

    def capitalize_word(word: str) -> str:
        """Capitalize a single word, preserving all-caps and mixed case."""
        if not word:
            return word

        # If word is all uppercase (like "3D", "AI", "JPEG"), preserve it
        if word.isupper():
            return word

        # If word has mixed case (like "iPhone", "MacBook"), preserve it
        if not word.islower() and not word.isupper():
            # Check if it's just first-letter-capitalized (like "Hello")
            if word[0].isupper() and word[1:].islower():
                return word  # Already title case
            # Otherwise it's mixed case like "iPhone" - preserve it
            return word

        # Regular lowercase word - capitalize first letter
        return word[0].upper() + word[1:].lower() if len(word) > 1 else word.upper()

    # Handle compound separators
    result = []
    for word in text.split():
        # Handle hyphenated words (e.g., "blue-sky" or "high-speed")
        if "-" in word:
            parts = word.split("-")
            word = "-".join(capitalize_word(p) for p in parts)
        # Handle underscored words (e.g., "blue_sky")
        elif "_" in word:
            parts = word.split("_")
            word = "_".join(capitalize_word(p) for p in parts)
        # Handle forward slash separated (e.g., "Art/Design")
        elif "/" in word:
            parts = word.split("/")
            word = "/".join(capitalize_word(p) for p in parts)
        else:
            word = capitalize_word(word)
        result.append(word)

    return " ".join(result)


def _normalize_keywords(keywords_raw: Any) -> List[str]:
    """Normalize keyword payloads into a flat list of strings."""
    if isinstance(keywords_raw, str):
        return [k.strip() for k in keywords_raw.split(",") if k and k.strip()]

    if isinstance(keywords_raw, list):
        keywords = []
        for item in keywords_raw:
            if isinstance(item, str):
                keywords.extend(
                    [part.strip() for part in item.split(",") if part and part.strip()]
                )
            elif item is not None:
                keywords.append(str(item).strip())
        return [keyword for keyword in keywords if keyword]

    return []


def _sanitize_category(category_raw: Any) -> str:
    """
    Normalize a category field that may be a string, list, or malformed value.

    Handles:
    - List of repeated values (e.g. ['travel', 'traveling', ...]) -> pick most common
    - Nested lists -> flatten and pick most common
    - Non-string types -> convert to string
    - Deeply nested/garbage values -> return '[AI: No Result]'
    """
    if isinstance(category_raw, str):
        return category_raw.strip()

    if isinstance(category_raw, list):
        if not category_raw:
            return ""

        # Flatten if nested lists
        flat = []
        for item in category_raw:
            if isinstance(item, list):
                flat.extend(str(i).strip() for i in item if i)
            elif item is not None:
                flat.append(str(item).strip())

        flat = [f for f in flat if f]
        if not flat:
            return ""

        # If all values are identical or nearly identical (repeated keyword spam),
        # just return the first one
        if len(set(flat)) == 1:
            return flat[0]

        # Pick the most common value (mode)
        from collections import Counter

        most_common, count = Counter(flat).most_common(1)[0]
        # If the most common value appears more than once, use it
        if count > 1:
            return most_common
        # Otherwise fall back to first non-empty
        return flat[0]

    if category_raw is not None:
        result = str(category_raw).strip()
        # If conversion produced something reasonable-ish, use it
        if result and len(result) < 200:
            return result

    # Default: mark as unreliable
    return "[AI: No Result]"


def extract_tags_from_result(
    result: Any,
    model_task: str,
    threshold: float = 0.0,
    stop_words: Optional[List[str]] = None,
) -> Tuple[str, List[str], str]:
    """
    Extract category, keywords, and description from AI model output.

    This function parses the raw output from different types of AI models and converts
    it into structured metadata tags. It handles three main model types:

    1. Image Classification: Extracts top-N keyword labels with confidence scores
    2. Zero-Shot Classification: Finds the best matching category from user-defined options
    3. Image-to-Text: Parses generated captions and extracts JSON-structured metadata

    The function includes special handling for:
    - Vision-Language Models (VLMs) that output structured JSON
    - Multi-modal models that combine text generation with image understanding
    - Confidence threshold filtering to ensure quality
    - Stop word filtering for generated text
    - **Title Case formatting for all keywords and categories**

    Args:
        result: The raw output from the AI pipeline. Format varies by task:
                - Classification: List[Dict] with 'label' and 'score' keys
                - Zero-Shot: List[Dict] with 'label' and 'score' keys (pre-sorted)
                - Image-to-Text: List[Dict] or Dict with 'generated_text' key
        model_task: The AI task type (from config.MODEL_TASK_* constants)
        threshold: Minimum confidence score (0.0-1.0) for including results.
                   Default 0.0 means no filtering by confidence.
        stop_words: Optional list of words to exclude from extracted keywords.
                    Primarily used for image-to-text tasks.

    Returns:
        Tuple of (category, keywords, description) where:
        - category: Single best category label (str, may be empty) - Title Cased
        - keywords: List of keyword tags (List[str], may be empty) - Title Cased
        - description: Descriptive caption text (str, may be empty)

    Note:
        - For VLMs, attempts to parse JSON from generated text first
        - Falls back to plain text extraction if JSON parsing fails
        - Applies config.MAX_KEYWORDS_PER_IMAGE limit to prevent tag spam
        - All keywords and categories are converted to Title Case
    """
    category = ""
    keywords = []
    description = ""

    # Enhanced logging for troubleshooting
    if model_task in [
        config.MODEL_TASK_IMAGE_CLASSIFICATION,
        config.MODEL_TASK_ZERO_SHOT,
        config.MODEL_TASK_IMAGE_TO_TEXT,
        "image-text-to-text",
    ]:
        logging.info(f"Extracting tags - Task: {model_task}, Threshold: {threshold}")
        # Log a bit more of the result for debugging (e.g. the "S" issue)
        logging.info(f"Raw Result: {str(result)[:500]}")

    try:
        if model_task == config.MODEL_TASK_IMAGE_CLASSIFICATION:
            # "Keywords (Auto)" - Extract top 5 specific tags
            if isinstance(result, list):
                # Sort by score descending just in case
                sorted_res = sorted(result, key=lambda x: x["score"], reverse=True)
                for item in sorted_res[:5]:  # Top 5
                    if item["score"] >= threshold:
                        label = item["label"]
                        # Handle models that might return comma-separated labels
                        keywords.extend([k.strip() for k in label.split(",")])

            elif isinstance(result, dict):
                if result["score"] >= threshold:
                    label = result["label"]
                    keywords.extend([k.strip() for k in label.split(",")])

        elif model_task == config.MODEL_TASK_ZERO_SHOT:
            # "Categories (Custom)" - Extract broad buckets
            # We map this to CATEGORY (Subject) now.
            matched_categories = []

            # Handle list of dicts (standard for image zero-shot)
            if isinstance(result, list):
                # Sort by score descending
                sorted_res = sorted(result, key=lambda x: x["score"], reverse=True)
                for item in sorted_res:
                    if isinstance(item, dict) and "label" in item and "score" in item:
                        if item["score"] >= threshold:
                            matched_categories.append(item["label"])

            # Handle dict with lists (text-style zero-shot)
            elif isinstance(result, dict) and "labels" in result and "scores" in result:
                # Zip and sort
                zipped = sorted(
                    zip(result["labels"], result["scores"]),
                    key=lambda x: x[1],
                    reverse=True,
                )
                for label, score in zipped:
                    if score >= threshold:
                        matched_categories.append(label)

            if matched_categories:
                # User preference: Single best category instead of list
                category = matched_categories[0]
                # Log usage
                logging.info(f"Zero-Shot Category: '{category}' (Score: >={threshold})")

        elif model_task in [config.MODEL_TASK_IMAGE_TO_TEXT, "image-text-to-text"]:
            # Result: [{'generated_text': '...'}] or [{'generated_text': {'description': '...', ...}}]

            raw_gen = None
            if isinstance(result, list) and len(result) > 0:
                raw_gen = result[0].get("generated_text", "")
            elif isinstance(result, dict):
                raw_gen = result.get("generated_text", "")

            # CASE 1: Structured Dictionary (From Smart VLM)
            if isinstance(raw_gen, dict):
                description = raw_gen.get("description", "")
                category = _sanitize_category(raw_gen.get("category", ""))
                keywords = _normalize_keywords(raw_gen.get("keywords", []))

            # CASE 2: String (Plain Caption) or Chat Format
            else:
                text = ""
                # If raw_gen is a list of messages (chat format), extract assistant text
                if isinstance(raw_gen, list):
                    for msg in raw_gen:
                        if isinstance(msg, dict) and msg.get("role") == "assistant":
                            content = msg.get("content", "")
                            if isinstance(content, list):
                                for item in content:
                                    if (
                                        isinstance(item, dict)
                                        and item.get("type") == "text"
                                    ):
                                        text += item.get("text", "")
                            elif isinstance(content, str):
                                text += content
                elif isinstance(raw_gen, str):
                    text = raw_gen

                # Try to extract JSON from the text (robust approach)
                if text:
                    json_extracted = False
                    data = extract_dict_from_text(
                        text,
                        expected_keys={"description", "category", "keywords"},
                    )
                    if isinstance(data, dict):
                        description = data.get("description", "")
                        category = _sanitize_category(data.get("category", ""))
                        keywords = _normalize_keywords(data.get("keywords", []))
                        json_extracted = True
                        logging.info(
                            "Successfully extracted structured payload from generated text"
                        )

                    # Method 3: Fallback - treat entire text as plain description (only if JSON extraction failed)
                    if not json_extracted:
                        logging.warning(
                            f"Could not extract JSON from model response, using text as plain description"
                        )
                        description = text.strip()

                        # Cleanup: remove common prompt prefixes and structural artifacts
                        prefixes_to_strip = [
                            "Describe the image.",
                            "Describe this image.",
                            "Caption:",
                            "Description:",
                            "The image shows",
                            "This image shows",
                            "An image of",
                            "A picture of",
                            "generated_text:",
                            "Output:",
                            "Response:",
                            "Analysing the image:",
                            "Analysis:",
                            "Here is the JSON object:",
                            "```json",
                            "```",
                        ]

                        # Case-insensitive prefix stripping loop
                        still_stripping = True
                        while still_stripping:
                            original = description
                            for prefix in prefixes_to_strip:
                                if description.lower().startswith(prefix.lower()):
                                    description = description[len(prefix) :].strip()

                            # Strip common VLM artifacts like 's' at the start followed by comma (often from 'Image shows...')
                            # or other rogue leading characters
                            if description.lower().startswith("s, "):
                                description = description[3:].strip()
                            elif description.lower().startswith("s "):
                                description = description[2:].strip()

                            # Strip leading punctuation/symbols often left by prefix removal
                            description = description.lstrip(":.,- ")

                            if description == original:
                                still_stripping = False

                        # Final polish
                        description = description.strip()

                        # If description is just a single character (like 'S'), it's likely a failure or artifact
                        if len(description) <= 1:
                            logging.warning(
                                f"Extracted description too short ('{description}'). Setting to empty."
                            )
                            description = ""

                        if description and not description[0].isupper():
                            description = description[0].upper() + description[1:]

                        # We do NOT extract keywords from caption anymore.
                        # Unless they were extracted via JSON above.
                        if not keywords:
                            keywords = []

        # Deduplicate keywords
        if keywords:
            keywords = list(dict.fromkeys([k for k in keywords if k]))

    except Exception as e:
        logging.error(f"Error extracting tags from result: {e}")

    # =========================================================================
    # TITLE CASE NORMALIZATION
    # =========================================================================
    # Apply Title Case to all keywords and categories for consistent formatting
    if category:
        category = to_title_case(category)

    if keywords:
        keywords = [to_title_case(k) for k in keywords if k]
        # Re-deduplicate after title casing (in case "Blue Sky" and "blue sky" become same)
        keywords = list(dict.fromkeys(keywords))

    logging.debug(
        f"Final tags - Category: '{category}', Keywords: {keywords[:5]}..., Description: '{description[:50]}...'"
    )

    return category, keywords, description
