#!/bin/bash
# Tesseract OCR Benchmark Script (Native version)
# Measures CPU, memory, and timing during OCR processing.
# Designed to behave identically to the pytesseract benchmark.py

set -e

INPUT_DIR="${INPUT_DIR:-/app/input}"
OUTPUT_DIR="${OUTPUT_DIR:-/app/output}"
OCR_LANG="${TESSERACT_LANG:-eng}"
ITERATIONS="${ITERATIONS:-3}"

RUN_ID=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="${OUTPUT_DIR}/benchmark_native_${RUN_ID}.json"
SAMPLES_FILE="/tmp/samples_${RUN_ID}.csv"
OCR_RESULTS_FILE="/tmp/ocr_results_${RUN_ID}.json"
PDF_TEMP_DIR="/tmp/pdf_pages_${RUN_ID}"
FILE_TIMES_FILE="/tmp/file_times_${RUN_ID}.csv"
PAGE_METRICS_FILE="/tmp/page_metrics_${RUN_ID}.csv"

# Initialize metrics collection
echo "timestamp,cpu_percent,memory_mb,memory_percent" > "$SAMPLES_FILE"

# Initialize OCR results as CSV (will be converted to JSON at the end)
# Format: result_index,filename,file_size_kb,width,height,processing_time_ms,text_length,success,page_count,error
echo "result_index,filename,file_size_kb,width,height,processing_time_ms,text_length,success,page_count,error" > "$OCR_RESULTS_FILE"

# Initialize file times tracking (for average time per page calculation)
echo "filename,time_per_page_ms,page_count" > "$FILE_TIMES_FILE"

# Initialize page metrics tracking (for per-page resource correlation)
# Format: result_index,filename,page_number,start_timestamp,end_timestamp,processing_time_ms,width,height,text_length
echo "result_index,filename,page_number,start_timestamp,end_timestamp,processing_time_ms,width,height,text_length" > "$PAGE_METRICS_FILE"

# Counter for OCR results (to link page metrics to results)
OCR_RESULT_INDEX=0

# Get process info
PID=$$

# Get total system memory for percentage calculation (ps %mem often reports 0 in containers)
TOTAL_MEM_KB=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}' || echo "0")
TOTAL_MEM_MB=$(echo "scale=2; $TOTAL_MEM_KB / 1024" | bc)

# Resource monitoring function (runs in background)
monitor_resources() {
    while true; do
        if [ -f "/tmp/stop_monitor_${RUN_ID}" ]; then
            break
        fi

        # Get CPU and memory for our process tree
        CPU=$(ps -p $PID -o %cpu= 2>/dev/null | tr -d ' ' || echo "0")
        MEM_KB=$(ps -p $PID -o rss= 2>/dev/null | tr -d ' ' || echo "0")
        MEM_MB=$(awk "BEGIN {printf \"%.2f\", $MEM_KB / 1024}")
        # Calculate memory percent manually (ps %mem often reports 0 in containers)
        if [ "$TOTAL_MEM_MB" != "0" ] && [ "$TOTAL_MEM_MB" != "" ]; then
            # Use awk to ensure proper JSON number format (leading zero for decimals)
            MEM_PCT=$(awk "BEGIN {printf \"%.6f\", ($MEM_MB / $TOTAL_MEM_MB) * 100}")
        else
            MEM_PCT="0"
        fi
        TIMESTAMP=$(date +%s.%N)

        echo "$TIMESTAMP,$CPU,$MEM_MB,$MEM_PCT" >> "$SAMPLES_FILE"
        sleep 0.25
    done
}

# Get Tesseract version
TESSERACT_VERSION=$(tesseract --version 2>&1 | head -1 | cut -d' ' -f2)

echo "[Benchmark] Starting run: $RUN_ID"
echo "[Benchmark] Tesseract version: $TESSERACT_VERSION"
echo "[Benchmark] Input directory: $INPUT_DIR"
echo "[Benchmark] Iterations per file: $ITERATIONS"

# Supported extensions (matching Python version)
# .png, .jpg, .jpeg, .tiff, .tif, .bmp, .gif, .pdf
SUPPORTED_EXTENSIONS="png|jpg|jpeg|tiff|tif|bmp|gif|pdf"

# Find and sort files alphabetically (matching Python's sorted() behavior)
INPUT_FILES=$(find "$INPUT_DIR" -maxdepth 1 -type f | grep -iE "\.($SUPPORTED_EXTENSIONS)$" | sort)
FILE_COUNT=$(echo "$INPUT_FILES" | grep -c . || echo "0")

if [ "$FILE_COUNT" -eq 0 ]; then
    echo "[Benchmark] ERROR: No supported files found in input directory!"
    echo "[Benchmark] Supported formats: png, jpg, jpeg, tiff, tif, bmp, gif, pdf"
    exit 1
fi

echo "[Benchmark] Found $FILE_COUNT files to process"

# Function to append OCR result to CSV file
append_ocr_result() {
    local filename="$1"
    local file_size_kb="$2"
    local width="$3"
    local height="$4"
    local processing_time_ms="$5"
    local text_length="$6"
    local success="$7"
    local page_count="$8"
    local error="$9"

    # Append as CSV row
    echo "$OCR_RESULT_INDEX,$filename,$file_size_kb,$width,$height,$processing_time_ms,$text_length,$success,$page_count,$error" >> "$OCR_RESULTS_FILE"
}

# Function to process a single image file
process_image() {
    local image_path="$1"
    local filename=$(basename "$image_path")

    # Get file info (use stat for exact bytes, divide for float KB like Python)
    local file_size_bytes=$(stat -c%s "$image_path" 2>/dev/null || stat -f%z "$image_path" 2>/dev/null)
    local file_size_kb=$(echo "scale=6; $file_size_bytes / 1024" | bc)
    local dimensions=$(identify -format "%wx%h" "$image_path" 2>/dev/null || echo "0x0")
    local width=$(echo "$dimensions" | cut -d'x' -f1)
    local height=$(echo "$dimensions" | cut -d'x' -f2)

    # Run OCR with timing (record timestamps for correlation)
    local ocr_start=$(date +%s.%N)
    local output_file="${OUTPUT_DIR}/${filename%.*}.txt"
    local ocr_output

    if ocr_output=$(tesseract "$image_path" stdout -l "$OCR_LANG" 2>/dev/null); then
        local ocr_end=$(date +%s.%N)
        local processing_time=$(echo "scale=3; ($ocr_end - $ocr_start) * 1000" | bc)
        local text_length=${#ocr_output}

        echo "  -> OK: ${processing_time}ms, ${text_length} chars, 1 page(s)"
        append_ocr_result "$filename" "$file_size_kb" "$width" "$height" "$processing_time" "$text_length" "true" "1" ""
        # Track time per page for averaging
        echo "$filename,$processing_time,1" >> "$FILE_TIMES_FILE"
        # Track page metrics with timestamps for resource correlation
        echo "$OCR_RESULT_INDEX,$filename,1,$ocr_start,$ocr_end,$processing_time,$width,$height,$text_length" >> "$PAGE_METRICS_FILE"
        OCR_RESULT_INDEX=$((OCR_RESULT_INDEX + 1))
    else
        local ocr_end=$(date +%s.%N)
        local processing_time=$(echo "scale=3; ($ocr_end - $ocr_start) * 1000" | bc)
        echo "  -> FAILED: OCR error"
        append_ocr_result "$filename" "$file_size_kb" "0" "0" "$processing_time" "0" "false" "0" "OCR processing failed"
        OCR_RESULT_INDEX=$((OCR_RESULT_INDEX + 1))
    fi
}

# Function to process a PDF file using Poppler (pdftoppm)
# This matches pdf2image which pytesseract uses
process_pdf() {
    local pdf_path="$1"
    local filename=$(basename "$pdf_path")

    # Get file info (use stat for exact bytes, divide for float KB like Python)
    local file_size_bytes=$(stat -c%s "$pdf_path" 2>/dev/null || stat -f%z "$pdf_path" 2>/dev/null)
    local file_size_kb=$(echo "scale=6; $file_size_bytes / 1024" | bc)

    # Create temp directory for PDF pages
    local page_dir="${PDF_TEMP_DIR}/${filename%.*}"
    mkdir -p "$page_dir"

    local ocr_start=$(date +%s.%N)

    # Convert PDF to images using pdftoppm (same as pdf2image uses)
    # Using PPM format (uncompressed) and 200 DPI to match pdf2image defaults
    # PPM is much faster than PNG since no compression is needed
    local convert_start=$(date +%s.%N)
    if ! pdftoppm -r 200 "$pdf_path" "${page_dir}/page" 2>/dev/null; then
        local ocr_end=$(date +%s.%N)
        local processing_time=$(echo "scale=3; ($ocr_end - $ocr_start) * 1000" | bc)
        echo "  -> FAILED: PDF conversion error"
        append_ocr_result "$filename" "$file_size_kb" "0" "0" "$processing_time" "0" "false" "0" "PDF to image conversion failed"
        rm -rf "$page_dir"
        return
    fi
    local convert_end=$(date +%s.%N)
    local convert_time=$(echo "scale=1; ($convert_end - $convert_start) * 1000" | bc)
    echo "     [pdftoppm: ${convert_time}ms]"

    # Get list of generated page images (sorted) into an array
    # pdftoppm outputs as page-1.ppm, page-2.ppm, etc. (PPM format)
    local page_images=()
    while IFS= read -r img; do
        [ -n "$img" ] && page_images+=("$img")
    done < <(find "$page_dir" -name "page-*.ppm" | sort -V)

    local page_count=${#page_images[@]}

    if [ "$page_count" -eq 0 ]; then
        local ocr_end=$(date +%s.%N)
        local processing_time=$(echo "scale=3; ($ocr_end - $ocr_start) * 1000" | bc)
        echo "  -> FAILED: No pages extracted"
        append_ocr_result "$filename" "$file_size_kb" "0" "0" "$processing_time" "0" "false" "0" "No pages extracted from PDF"
        rm -rf "$page_dir"
        return
    fi

    # Process each page and collect results
    local max_width=0
    local max_height=0
    local text_file="${page_dir}/.all_text"
    local page_text_lengths_file="${page_dir}/.page_lengths"
    > "$text_file"  # Create/truncate file
    > "$page_text_lengths_file"

    local ocr_loop_start=$(date +%s.%N)
    local page_num=0
    for page_image in "${page_images[@]}"; do
        page_num=$((page_num + 1))
        # Record timestamps for correlation
        local page_start=$(date +%s.%N)

        # Get page dimensions from PPM header (much faster than ImageMagick identify)
        # PPM format: "P6\n[comments]\nwidth height\nmaxval\n" followed by binary data
        # Skip comment lines (starting with #) and get first line with two numbers
        local dimensions=$(head -c 200 "$page_image" | tr '\0' '\n' | grep -v '^#' | grep -v '^P' | head -1)
        local width=$(echo "$dimensions" | awk '{print $1}')
        local height=$(echo "$dimensions" | awk '{print $2}')

        # Track max dimensions
        [ "$width" -gt "$max_width" ] && max_width=$width
        [ "$height" -gt "$max_height" ] && max_height=$height

        # Run OCR on page, capture output for length measurement
        local page_text=$(tesseract "$page_image" stdout -l "$OCR_LANG" 2>/dev/null)
        echo "$page_text" >> "$text_file"
        echo "" >> "$text_file"
        local page_text_length=${#page_text}

        local page_end=$(date +%s.%N)
        local page_time=$(echo "scale=3; ($page_end - $page_start) * 1000" | bc)
        echo "     [page $page_num: ${page_time%.000}ms]"

        # Track page metrics with timestamps for resource correlation
        echo "$OCR_RESULT_INDEX,$filename,$page_num,$page_start,$page_end,$page_time,$width,$height,$page_text_length" >> "$PAGE_METRICS_FILE"
    done
    local ocr_loop_end=$(date +%s.%N)
    local ocr_loop_time=$(echo "scale=1; ($ocr_loop_end - $ocr_loop_start) * 1000" | bc)
    echo "     [OCR total: ${ocr_loop_time}ms]"

    local text_length=$(wc -c < "$text_file" | tr -d ' ')

    local ocr_end=$(date +%s.%N)
    local processing_time=$(echo "scale=3; ($ocr_end - $ocr_start) * 1000" | bc)

    echo "  -> OK: ${processing_time}ms, ${text_length} chars, ${page_count} page(s)"
    append_ocr_result "$filename" "$file_size_kb" "$max_width" "$max_height" "$processing_time" "$text_length" "true" "$page_count" ""
    # Track time per page for averaging
    local time_per_page=$(echo "scale=3; $processing_time / $page_count" | bc)
    echo "$filename,$time_per_page,$page_count" >> "$FILE_TIMES_FILE"
    # Increment result index (page metrics already written in loop)
    OCR_RESULT_INDEX=$((OCR_RESULT_INDEX + 1))

    # Cleanup page images
    rm -rf "$page_dir"
}

# Start resource monitoring in background
rm -f "/tmp/stop_monitor_${RUN_ID}"
monitor_resources &
MONITOR_PID=$!

START_TIME=$(date +%s.%N)
START_TIME_ISO=$(date -Iseconds)

# Build array of input files (to avoid subshell issues with while read)
INPUT_FILES_ARRAY=()
while IFS= read -r f; do
    [ -n "$f" ] && INPUT_FILES_ARRAY+=("$f")
done <<< "$INPUT_FILES"

# Process files
for iter in $(seq 1 $ITERATIONS); do
    echo ""
    echo "[Benchmark] === Iteration $iter/$ITERATIONS ==="

    for file_path in "${INPUT_FILES_ARRAY[@]}"; do
        filename=$(basename "$file_path")
        echo "[Benchmark] Processing: $filename"

        # Check if PDF or image
        extension="${filename##*.}"
        extension_lower=$(echo "$extension" | tr '[:upper:]' '[:lower:]')

        if [ "$extension_lower" = "pdf" ]; then
            process_pdf "$file_path"
        else
            process_image "$file_path"
        fi
    done
done

END_TIME=$(date +%s.%N)
END_TIME_ISO=$(date -Iseconds)
TOTAL_DURATION=$(echo "scale=2; $END_TIME - $START_TIME" | bc)

# Stop resource monitoring
touch "/tmp/stop_monitor_${RUN_ID}"
sleep 0.5
kill $MONITOR_PID 2>/dev/null || true

# Calculate statistics from samples
if [ -f "$SAMPLES_FILE" ]; then
    # Skip header line
    PEAK_CPU=$(tail -n +2 "$SAMPLES_FILE" | cut -d',' -f2 | sort -rn | head -1)
    PEAK_MEM=$(tail -n +2 "$SAMPLES_FILE" | cut -d',' -f3 | sort -rn | head -1)
    AVG_CPU=$(tail -n +2 "$SAMPLES_FILE" | cut -d',' -f2 | awk '{sum+=$1; count++} END {if(count>0) printf "%.1f", sum/count; else print "0"}')
    AVG_MEM=$(tail -n +2 "$SAMPLES_FILE" | cut -d',' -f3 | awk '{sum+=$1; count++} END {if(count>0) printf "%.1f", sum/count; else print "0"}')
else
    PEAK_CPU=0
    PEAK_MEM=0
    AVG_CPU=0
    AVG_MEM=0
fi

# Default values if empty
PEAK_CPU=${PEAK_CPU:-0}
PEAK_MEM=${PEAK_MEM:-0}
AVG_CPU=${AVG_CPU:-0}
AVG_MEM=${AVG_MEM:-0}

# Count results from OCR results CSV file
FILES_PROCESSED=$(tail -n +2 "$OCR_RESULTS_FILE" | grep -c ',true,' 2>/dev/null || true)
FILES_FAILED=$(tail -n +2 "$OCR_RESULTS_FILE" | grep -c ',false,' 2>/dev/null || true)
FILES_PROCESSED=${FILES_PROCESSED:-0}
FILES_FAILED=${FILES_FAILED:-0}

# Build resource_samples array from CSV
RESOURCE_SAMPLES="[]"
if [ -f "$SAMPLES_FILE" ]; then
    RESOURCE_SAMPLES=$(tail -n +2 "$SAMPLES_FILE" | awk -F',' '
        BEGIN { printf "[" }
        NF >= 4 && $1 != "" {
            if (NR > 1) printf ","
            printf "{\"timestamp\":%s,\"cpu_percent\":%s,\"memory_mb\":%s,\"memory_percent\":%s}", $1, $2, $3, $4
        }
        END { printf "]" }
    ')
fi

# Build OCR results with embedded page_metrics (matching Python structure)
OCR_RESULTS="[]"
if [ -f "$OCR_RESULTS_FILE" ] && [ -f "$PAGE_METRICS_FILE" ] && [ -f "$SAMPLES_FILE" ]; then
    OCR_RESULTS=$(awk -F',' '
    BEGIN {
        sample_count = 0
        ocr_count = 0
    }
    # First pass: read resource samples
    FILENAME == ARGV[1] && FNR > 1 {
        sample_count++
        sample_ts[sample_count] = $1
        sample_cpu[sample_count] = $2
        sample_mem[sample_count] = $3
    }
    # Second pass: read page metrics and correlate with samples
    FILENAME == ARGV[2] && FNR > 1 {
        result_idx = $1
        page_num = $3
        start_ts = $4
        end_ts = $5
        proc_time = $6
        p_width = $7
        p_height = $8
        text_len = $9

        # Find matching samples within time window
        peak_cpu = 0; sum_cpu = 0; peak_mem = 0; sum_mem = 0; match_count = 0
        for (i = 1; i <= sample_count; i++) {
            if (sample_ts[i] >= start_ts && sample_ts[i] <= end_ts) {
                match_count++
                sum_cpu += sample_cpu[i]
                sum_mem += sample_mem[i]
                if (sample_cpu[i] > peak_cpu) peak_cpu = sample_cpu[i]
                if (sample_mem[i] > peak_mem) peak_mem = sample_mem[i]
            }
        }
        avg_cpu = (match_count > 0) ? sum_cpu / match_count : 0
        avg_mem = (match_count > 0) ? sum_mem / match_count : 0

        # Build page metric JSON (output null for metrics when no samples matched)
        if (match_count > 0) {
            page_json = sprintf("{\"page_number\":%d,\"start_timestamp\":%s,\"end_timestamp\":%s,\"processing_time_ms\":%s,\"width\":%s,\"height\":%s,\"text_length\":%s,\"peak_cpu_percent\":%.1f,\"avg_cpu_percent\":%.1f,\"peak_memory_mb\":%.1f,\"avg_memory_mb\":%.1f,\"sample_count\":%d}",
                page_num, start_ts, end_ts, proc_time, p_width, p_height, text_len, peak_cpu, avg_cpu, peak_mem, avg_mem, match_count)
        } else {
            page_json = sprintf("{\"page_number\":%d,\"start_timestamp\":%s,\"end_timestamp\":%s,\"processing_time_ms\":%s,\"width\":%s,\"height\":%s,\"text_length\":%s,\"peak_cpu_percent\":null,\"avg_cpu_percent\":null,\"peak_memory_mb\":null,\"avg_memory_mb\":null,\"sample_count\":0}",
                page_num, start_ts, end_ts, proc_time, p_width, p_height, text_len)
        }

        pages[result_idx] = pages[result_idx] (pages[result_idx] ? "," : "") page_json
    }
    # Third pass: read OCR results and build final JSON
    FILENAME == ARGV[3] && FNR > 1 {
        ocr_count++
        result_idx = $1
        filename = $2
        file_size_kb = $3
        width = $4
        height = $5
        proc_time = $6
        text_len = $7
        success = $8
        page_count = $9
        error = $10

        # Build page_metrics array for this result
        page_metrics_json = pages[result_idx] ? pages[result_idx] : ""

        # Store OCR result with embedded page_metrics
        if (success == "true") {
            ocr_json[ocr_count] = sprintf("{\"filename\":\"%s\",\"file_size_kb\":%s,\"image_width\":%s,\"image_height\":%s,\"processing_time_ms\":%s,\"text_length\":%s,\"success\":true,\"page_count\":%s,\"error\":null,\"page_metrics\":[%s]}",
                filename, file_size_kb, width, height, proc_time, text_len, page_count, page_metrics_json)
        } else {
            ocr_json[ocr_count] = sprintf("{\"filename\":\"%s\",\"file_size_kb\":%s,\"image_width\":%s,\"image_height\":%s,\"processing_time_ms\":%s,\"text_length\":%s,\"success\":false,\"page_count\":%s,\"error\":\"%s\",\"page_metrics\":null}",
                filename, file_size_kb, width, height, proc_time, text_len, page_count, error)
        }
    }
    END {
        printf "["
        for (i = 1; i <= ocr_count; i++) {
            if (i > 1) printf ","
            printf "%s", ocr_json[i]
        }
        printf "]"
    }
    ' "$SAMPLES_FILE" "$PAGE_METRICS_FILE" "$OCR_RESULTS_FILE")
fi

# Calculate average time per page for each file across iterations
AVG_TIME_PER_PAGE="[]"
if [ -f "$FILE_TIMES_FILE" ]; then
    AVG_TIME_PER_PAGE=$(tail -n +2 "$FILE_TIMES_FILE" | awk -F',' '
        {
            filename[$1] = $1
            sum[$1] += $2
            count[$1]++
            pages[$1] = $3
        }
        END {
            printf "["
            first = 1
            for (f in filename) {
                if (!first) printf ","
                first = 0
                avg = sum[f] / count[f]
                printf "{\"filename\":\"%s\",\"avg_time_per_page_ms\":%.3f,\"total_pages\":%d,\"iterations\":%d}", f, avg, pages[f], count[f]
            }
            printf "]"
        }
    ')
fi

# Generate final JSON report (matching Python output structure)
cat > "$RESULTS_FILE" << EOF
{
  "run_id": "$RUN_ID",
  "start_time": "$START_TIME_ISO",
  "end_time": "$END_TIME_ISO",
  "total_duration_s": $TOTAL_DURATION,
  "engine": "tesseract-native",
  "tesseract_version": "$TESSERACT_VERSION",
  "files_processed": $FILES_PROCESSED,
  "files_failed": $FILES_FAILED,
  "ocr_results": $OCR_RESULTS,
  "resource_samples": $RESOURCE_SAMPLES,
  "peak_cpu_percent": $PEAK_CPU,
  "peak_memory_mb": $PEAK_MEM,
  "avg_cpu_percent": $AVG_CPU,
  "avg_memory_mb": $AVG_MEM,
  "avg_time_per_page": $AVG_TIME_PER_PAGE
}
EOF

echo ""
echo "[Benchmark] === Results ==="
echo "  Total duration: ${TOTAL_DURATION}s"
echo "  Files processed: $FILES_PROCESSED"
echo "  Files failed: $FILES_FAILED"
echo "  Peak CPU: ${PEAK_CPU}%"
echo "  Peak Memory: ${PEAK_MEM} MB"
echo "  Avg CPU: ${AVG_CPU}%"
echo "  Avg Memory: ${AVG_MEM} MB"
echo "  Results saved to: $RESULTS_FILE"

# Cleanup temp files
rm -f "$SAMPLES_FILE" "$OCR_RESULTS_FILE" "$FILE_TIMES_FILE" "$PAGE_METRICS_FILE" "/tmp/stop_monitor_${RUN_ID}"
rm -rf "$PDF_TEMP_DIR"
