#!/usr/bin/env python3
"""
Synthetic PDF Document Generator

This script generates synthetic PDF documents by combining random images
from multiple sources in the OCR datasets directory.

Features:
- Randomly selects 1-5 sources from available datasets
- Creates documents with 1-50 pages
- Distributes pages randomly across selected sources
- Shuffles page order for randomness
- Uses PyMuPDF (fitz) to create PDFs without modifying original images
- Generates metadata tracking original image sources for each page
"""

import os
import sys
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF is required. Install with: pip install PyMuPDF")
    sys.exit(1)

try:
    from PIL import Image
    import io
except ImportError:
    print("ERROR: Pillow is required. Install with: pip install Pillow")
    sys.exit(1)


# Configuration
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
MIN_SOURCES = 1
MAX_SOURCES = 2
MIN_PAGES = 1
MAX_PAGES = 10
NUM_DOCUMENTS = 2
VERSION = "v0.0.0"


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def discover_sources(datasets_path: Path) -> Dict[str, List[Path]]:
    """
    Discover all available image sources and their images.

    A source is defined as a top-level directory within the datasets folder.
    Images are searched recursively within each source.

    Args:
        datasets_path: Path to the ocr-datasets directory

    Returns:
        Dictionary mapping source names to lists of image paths
    """
    sources = {}

    if not datasets_path.exists():
        raise FileNotFoundError(f"Datasets path does not exist: {datasets_path}")

    for source_dir in datasets_path.iterdir():
        if not source_dir.is_dir() or source_dir.name.startswith("."):
            continue

        images = []
        for root, _, files in os.walk(source_dir):
            for file in files:
                if Path(file).suffix.lower() in IMAGE_EXTENSIONS:
                    images.append(Path(root) / file)

        if images:
            sources[source_dir.name] = images
            print(f"  Found source '{source_dir.name}' with {len(images)} images")

    return sources


def select_random_sources(
    sources: Dict[str, List[Path]],
    min_count: int = MIN_SOURCES,
    max_count: int = MAX_SOURCES
) -> List[str]:
    """
    Randomly select between min_count and max_count sources.

    Args:
        sources: Dictionary of available sources
        min_count: Minimum number of sources to select
        max_count: Maximum number of sources to select

    Returns:
        List of selected source names
    """
    available = list(sources.keys())
    num_sources = random.randint(min_count, min(max_count, len(available)))
    return random.sample(available, num_sources)


def distribute_pages(
    total_pages: int,
    num_sources: int
) -> List[int]:
    """
    Randomly distribute total pages across sources.

    Uses a random partition algorithm to ensure the sum equals total_pages.
    Each source gets at least 1 page.

    Args:
        total_pages: Total number of pages to distribute
        num_sources: Number of sources to distribute among

    Returns:
        List of page counts for each source
    """
    if num_sources > total_pages:
        # More sources than pages: give 1 page to random sources
        distribution = [0] * num_sources
        for idx in random.sample(range(num_sources), total_pages):
            distribution[idx] = 1
        return distribution

    # Start with 1 page per source
    distribution = [1] * num_sources
    remaining = total_pages - num_sources

    # Randomly distribute remaining pages
    for _ in range(remaining):
        idx = random.randint(0, num_sources - 1)
        distribution[idx] += 1

    return distribution


def select_images_from_sources(
    sources: Dict[str, List[Path]],
    selected_sources: List[str],
    distribution: List[int]
) -> List[Tuple[Path, str]]:
    """
    Select random images from each source according to distribution.

    Args:
        sources: Dictionary of all available sources and their images
        selected_sources: List of source names to use
        distribution: Number of images to select from each source

    Returns:
        List of tuples (image_path, source_name)
    """
    selected_images = []

    for source_name, count in zip(selected_sources, distribution):
        if count == 0:
            continue

        available_images = sources[source_name]

        # Handle case where we need more images than available
        if count > len(available_images):
            # Sample with replacement
            chosen = random.choices(available_images, k=count)
        else:
            chosen = random.sample(available_images, count)

        for img_path in chosen:
            selected_images.append((img_path, source_name))

    return selected_images


def load_image_with_pillow(img_path: Path) -> Tuple[bytes, Tuple[int, int]]:
    """
    Load an image using Pillow preserving the original format when possible.

    This handles images with incorrect extensions (e.g., WebP saved as .jpg).
    Preserves the original format to minimize modifications and optimize size.

    Args:
        img_path: Path to the image file

    Returns:
        Tuple of (image bytes, image dimensions)
    """
    with Image.open(img_path) as pil_img:
        original_format = pil_img.format  # Detect actual format (not extension)
        size = pil_img.size

        # Convert to RGB if necessary for JPEG output (handles RGBA, P, etc.)
        if pil_img.mode in ("RGBA", "P", "LA"):
            # For formats that don't support transparency well in PDF, use PNG
            if original_format in ("PNG", "WEBP") and pil_img.mode == "RGBA":
                output_format = "PNG"
            else:
                pil_img = pil_img.convert("RGB")
                output_format = original_format if original_format in ("JPEG", "PNG") else "PNG"
        else:
            output_format = original_format if original_format in ("JPEG", "PNG", "TIFF") else "PNG"

        buffer = io.BytesIO()

        if output_format == "JPEG":
            # Preserve JPEG with high quality
            pil_img.save(buffer, format="JPEG", quality=95, optimize=True)
        elif output_format == "TIFF":
            # Convert TIFF to PNG for better PDF compatibility
            pil_img.save(buffer, format="PNG", optimize=True, compress_level=9)
        else:
            # PNG with maximum compression
            pil_img.save(buffer, format="PNG", optimize=True, compress_level=9)

        return buffer.getvalue(), size


def create_pdf_from_images(
    images: List[Tuple[Path, str]],
    output_path: Path,
    project_root: Path
) -> Dict:
    """
    Create a PDF from a list of images without modifying the originals.

    Uses Pillow as fallback for images that PyMuPDF cannot open directly
    (e.g., WebP files with wrong extension).

    Args:
        images: List of tuples (image_path, source_name)
        output_path: Path where to save the PDF
        project_root: Project root for computing relative paths

    Returns:
        Metadata dictionary with page information (using relative paths)
    """
    # Shuffle the images for random page order
    shuffled_images = images.copy()
    random.shuffle(shuffled_images)

    # Create PDF
    doc = fitz.open()
    metadata = {
        "output_file": str(output_path.relative_to(project_root)),
        "total_pages": len(shuffled_images),
        "pages": []
    }

    for page_num, (img_path, source_name) in enumerate(shuffled_images):
        try:
            # Try opening with PyMuPDF first
            try:
                img = fitz.open(str(img_path))
                if img.page_count > 0:
                    img_page = img[0]
                    rect = img_page.rect
                    width, height = rect.width, rect.height
                    img.close()

                    # Create page and insert image
                    pdf_page = doc.new_page(width=width, height=height)
                    pdf_page.insert_image(rect, filename=str(img_path))
                else:
                    img.close()
                    raise ValueError("No pages in image")

            except Exception:
                # Fallback: use Pillow to load and convert the image
                img_bytes, (width, height) = load_image_with_pillow(img_path)
                rect = fitz.Rect(0, 0, width, height)

                pdf_page = doc.new_page(width=width, height=height)
                pdf_page.insert_image(rect, stream=img_bytes)

            # Record metadata for this page
            metadata["pages"].append({
                "page_number": page_num + 1,
                "source": source_name,
                "original_image": str(img_path.relative_to(project_root))
            })

        except Exception as e:
            print(f"  Warning: Could not process image {img_path}: {e}")
            continue

    # Save the PDF with compression optimizations
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(
        str(output_path),
        deflate=True,      # Use deflate compression for streams
        garbage=4,         # Maximum garbage collection
        clean=True,        # Clean and sanitize content
    )
    doc.close()

    return metadata


def generate_document(
    doc_id: int,
    sources: Dict[str, List[Path]],
    output_dir: Path,
    project_root: Path
) -> Dict:
    """
    Generate a single synthetic PDF document.

    Args:
        doc_id: Unique document identifier
        sources: Dictionary of available sources
        output_dir: Directory to save the PDF
        project_root: Project root for computing relative paths

    Returns:
        Metadata dictionary for the document
    """
    # Step 1: Select random sources (1-5)
    selected_sources = select_random_sources(sources)

    # Step 2: Define random total pages (1-50)
    total_pages = random.randint(MIN_PAGES, MAX_PAGES)

    # Step 3: Distribute pages among sources
    distribution = distribute_pages(total_pages, len(selected_sources))

    # Step 4: Select random images from each source
    selected_images = select_images_from_sources(
        sources, selected_sources, distribution
    )

    # Step 5 & 6: Shuffle images and create PDF with metadata
    output_filename = f"synthetic_doc_{doc_id:04d}.pdf"
    output_path = output_dir / output_filename

    metadata = create_pdf_from_images(selected_images, output_path, project_root)

    # Add document-level metadata
    metadata["document_id"] = doc_id
    metadata["sources_used"] = selected_sources
    metadata["distribution"] = dict(zip(selected_sources, distribution))
    metadata["created_at"] = datetime.now().isoformat()

    return metadata


def main():
    """Main entry point for the synthetic PDF generator."""
    print("=" * 60)
    print("Synthetic PDF Document Generator")
    print("=" * 60)

    # Setup paths
    project_root = get_project_root()
    datasets_path = project_root / "datalake" / "ocr-datasets"
    output_dir = project_root / "datalake" / "datasets" / VERSION
    metadata_file = output_dir / "metadata.json"

    print(f"\nDatasets path: {datasets_path}")
    print(f"Output directory: {output_dir}")

    # Step 0: Discover all sources and their images
    print("\n[Step 0] Discovering image sources...")
    sources = discover_sources(datasets_path)

    if not sources:
        print("ERROR: No image sources found!")
        sys.exit(1)

    total_images = sum(len(imgs) for imgs in sources.values())
    print(f"\nTotal: {len(sources)} sources, {total_images} images")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate documents
    print(f"\n[Generating {NUM_DOCUMENTS} synthetic PDF documents...]")
    all_metadata = {
        "generated_at": datetime.now().isoformat(),
        "total_documents": NUM_DOCUMENTS,
        "sources_available": list(sources.keys()),
        "documents": []
    }

    for doc_id in range(1, NUM_DOCUMENTS + 1):
        if doc_id % 100 == 0 or doc_id == 1:
            print(f"  Generating document {doc_id}/{NUM_DOCUMENTS}...")

        try:
            doc_metadata = generate_document(doc_id, sources, output_dir, project_root)
            all_metadata["documents"].append(doc_metadata)
        except Exception as e:
            print(f"  ERROR generating document {doc_id}: {e}")
            continue

    # Step 7: Save metadata to JSON
    print(f"\n[Saving metadata to {metadata_file}...]")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Generation complete!")
    print(f"  Documents created: {len(all_metadata['documents'])}")
    print(f"  Output directory: {output_dir}")
    print(f"  Metadata file: {metadata_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
