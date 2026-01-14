#!/usr/bin/env python3
"""
Tesseract OCR Benchmark Script (pytesseract version)
Measures CPU, memory, and timing during OCR processing.
"""

import os
import sys
import time
import json
import psutil
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional

import pytesseract
from PIL import Image
from pdf2image import convert_from_path


@dataclass
class ResourceSample:
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float


@dataclass
class OCRResult:
    filename: str
    file_size_kb: float
    image_width: int
    image_height: int
    processing_time_ms: float
    text_length: int
    success: bool
    page_count: int = 1
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    run_id: str
    start_time: str
    end_time: str
    total_duration_s: float
    engine: str
    tesseract_version: str
    files_processed: int
    files_failed: int
    ocr_results: List[OCRResult]
    resource_samples: List[ResourceSample]
    peak_cpu_percent: float
    peak_memory_mb: float
    avg_cpu_percent: float
    avg_memory_mb: float


class ResourceMonitor:
    """Background thread that samples CPU and memory usage."""
    
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.samples: List[ResourceSample] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._process = psutil.Process()
    
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
    
    def _sample_loop(self):
        while self._running:
            try:
                cpu = self._process.cpu_percent()
                mem_info = self._process.memory_info()
                mem_mb = mem_info.rss / (1024 * 1024)
                mem_percent = self._process.memory_percent()
                
                self.samples.append(ResourceSample(
                    timestamp=time.time(),
                    cpu_percent=cpu,
                    memory_mb=mem_mb,
                    memory_percent=mem_percent
                ))
            except Exception as e:
                print(f"[Monitor] Error sampling: {e}", file=sys.stderr)
            
            time.sleep(self.interval)


def get_tesseract_version() -> str:
    """Get Tesseract version string."""
    try:
        return pytesseract.get_tesseract_version().public
    except Exception:
        return "unknown"


def process_file(file_path: Path, lang: str = "eng") -> OCRResult:
    """Process a single file (image or PDF) and return OCR result with timing."""
    filename = file_path.name

    try:
        # Get file info
        file_size_kb = file_path.stat().st_size / 1024

        # Check if PDF
        if file_path.suffix.lower() == ".pdf":
            return process_pdf(file_path, lang, file_size_kb)

        # Load image
        img = Image.open(file_path)
        width, height = img.size

        # Run OCR with timing
        start_time = time.perf_counter()
        text = pytesseract.image_to_string(img, lang=lang)
        end_time = time.perf_counter()

        processing_time_ms = (end_time - start_time) * 1000

        return OCRResult(
            filename=filename,
            file_size_kb=file_size_kb,
            image_width=width,
            image_height=height,
            processing_time_ms=processing_time_ms,
            text_length=len(text),
            success=True,
            page_count=1
        )

    except Exception as e:
        return OCRResult(
            filename=filename,
            file_size_kb=0,
            image_width=0,
            image_height=0,
            processing_time_ms=0,
            text_length=0,
            success=False,
            page_count=0,
            error=str(e)
        )


def process_pdf(pdf_path: Path, lang: str, file_size_kb: float) -> OCRResult:
    """Process a PDF file page by page and return aggregated OCR result."""
    filename = pdf_path.name

    try:
        start_time = time.perf_counter()

        # Convert PDF to images
        convert_start = time.perf_counter()
        images = convert_from_path(pdf_path)
        convert_end = time.perf_counter()
        convert_time_ms = (convert_end - convert_start) * 1000
        print(f"     [pdf2image: {convert_time_ms:.1f}ms]")

        page_count = len(images)

        # Process each page
        all_text = []
        max_width = 0
        max_height = 0

        ocr_loop_start = time.perf_counter()
        for page_num, img in enumerate(images, 1):
            page_start = time.perf_counter()

            width, height = img.size
            max_width = max(max_width, width)
            max_height = max(max_height, height)

            # Run OCR on page
            text = pytesseract.image_to_string(img, lang=lang)
            all_text.append(text)

            page_end = time.perf_counter()
            page_time_ms = (page_end - page_start) * 1000
            print(f"     [page {page_num}: {page_time_ms:.0f}ms]")

        ocr_loop_end = time.perf_counter()
        ocr_loop_time_ms = (ocr_loop_end - ocr_loop_start) * 1000
        print(f"     [OCR total: {ocr_loop_time_ms:.1f}ms]")

        end_time = time.perf_counter()
        processing_time_ms = (end_time - start_time) * 1000

        combined_text = "\n\n".join(all_text)

        return OCRResult(
            filename=filename,
            file_size_kb=file_size_kb,
            image_width=max_width,
            image_height=max_height,
            processing_time_ms=processing_time_ms,
            text_length=len(combined_text),
            success=True,
            page_count=page_count
        )

    except Exception as e:
        return OCRResult(
            filename=filename,
            file_size_kb=file_size_kb,
            image_width=0,
            image_height=0,
            processing_time_ms=0,
            text_length=0,
            success=False,
            page_count=0,
            error=str(e)
        )


def run_benchmark(
    input_dir: Path,
    output_dir: Path,
    lang: str = "eng",
    iterations: int = 1
) -> BenchmarkResult:
    """Run the full benchmark suite."""
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = datetime.now()
    
    print(f"[Benchmark] Starting run: {run_id}")
    print(f"[Benchmark] Tesseract version: {get_tesseract_version()}")
    print(f"[Benchmark] Input directory: {input_dir}")
    print(f"[Benchmark] Iterations per file: {iterations}")
    
    # Find all supported files (images and PDFs)
    supported_extensions = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".pdf"}
    input_files = sorted([
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in supported_extensions
    ])

    if not input_files:
        print("[Benchmark] ERROR: No supported files found in input directory!")
        print(f"[Benchmark] Supported formats: {', '.join(supported_extensions)}")
        sys.exit(1)

    print(f"[Benchmark] Found {len(input_files)} files to process")
    
    # Start resource monitoring
    monitor = ResourceMonitor(interval=0.25)
    monitor.start()
    
    # Process all images
    ocr_results: List[OCRResult] = []
    
    for iteration in range(iterations):
        print(f"\n[Benchmark] === Iteration {iteration + 1}/{iterations} ===")

        for file_path in input_files:
            print(f"[Benchmark] Processing: {file_path.name}")
            result = process_file(file_path, lang)
            ocr_results.append(result)

            if result.success:
                print(f"  -> OK: {result.processing_time_ms:.1f}ms, {result.text_length} chars, {result.page_count} page(s)")
            else:
                print(f"  -> FAILED: {result.error}")
    
    # Stop monitoring
    monitor.stop()
    
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    # Calculate statistics
    samples = monitor.samples
    
    if samples:
        peak_cpu = max(s.cpu_percent for s in samples)
        peak_memory = max(s.memory_mb for s in samples)
        avg_cpu = sum(s.cpu_percent for s in samples) / len(samples)
        avg_memory = sum(s.memory_mb for s in samples) / len(samples)
    else:
        peak_cpu = peak_memory = avg_cpu = avg_memory = 0
    
    files_processed = sum(1 for r in ocr_results if r.success)
    files_failed = sum(1 for r in ocr_results if not r.success)
    
    result = BenchmarkResult(
        run_id=run_id,
        start_time=start_time.isoformat(),
        end_time=end_time.isoformat(),
        total_duration_s=total_duration,
        engine="pytesseract",
        tesseract_version=get_tesseract_version(),
        files_processed=files_processed,
        files_failed=files_failed,
        ocr_results=ocr_results,
        resource_samples=[asdict(s) for s in samples],
        peak_cpu_percent=peak_cpu,
        peak_memory_mb=peak_memory,
        avg_cpu_percent=avg_cpu,
        avg_memory_mb=avg_memory
    )
    
    # Save results
    output_file = output_dir / f"benchmark_pytesseract_{run_id}.json"
    with open(output_file, "w") as f:
        json.dump(asdict(result), f, indent=2)
    
    print(f"\n[Benchmark] === Results ===")
    print(f"  Total duration: {total_duration:.2f}s")
    print(f"  Files processed: {files_processed}")
    print(f"  Files failed: {files_failed}")
    print(f"  Peak CPU: {peak_cpu:.1f}%")
    print(f"  Peak Memory: {peak_memory:.1f} MB")
    print(f"  Avg CPU: {avg_cpu:.1f}%")
    print(f"  Avg Memory: {avg_memory:.1f} MB")
    print(f"  Results saved to: {output_file}")
    
    return result


if __name__ == "__main__":
    input_dir = Path(os.environ.get("INPUT_DIR", "/app/input"))
    output_dir = Path(os.environ.get("OUTPUT_DIR", "/app/output"))
    lang = os.environ.get("TESSERACT_LANG", "eng")
    iterations = int(os.environ.get("ITERATIONS", "3"))
    
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    run_benchmark(input_dir, output_dir, lang, iterations)
