# Moondream Performance Tester

A Docker-based Python application for testing Moondream vision model performance on CPU-only systems. This tool processes a batch of images and generates detailed performance metrics including processing times, captions, and system resource usage.

## Features

- **CPU-Optimized**: Designed specifically for CPU-only execution (no GPU required)
- **Batch Processing**: Process entire folders of images at once
- **Comprehensive Metrics**: Detailed timing and performance data for each image
- **Rich CLI Interface**: Beautiful progress bars and formatted output
- **CSV Export**: Structured data export for analysis
- **Docker Containerized**: Easy deployment on any system
- **Error Handling**: Robust error handling with detailed reporting

## System Requirements

- **RAM**: 8GB+ (16GB recommended)
- **Storage**: 5GB for model weights + space for your images
- **CPU**: Any modern x86_64 processor (Xeon optimized)
- **Docker**: Docker Engine installed

## Quick Start

### 1. Build the Docker Image

```bash
git clone <repository-url>
cd moondream-bot
docker build -t moondream-perf .
```

### 2. Prepare Your Images

Place your test images in a folder, for example:
```
/path/to/your/images/
├── image1.jpg
├── image2.png
├── subfolder/
│   └── image3.jpg
└── ...
```

### 3. Run Performance Test

```bash
docker run --rm \
  -v /path/to/your/images:/app/images:ro \
  -v /path/to/output:/app/output \
  moondream-perf \
  --input /app/images \
  --output /app/output/results.csv
```

## Usage Examples

### Basic Usage
```bash
docker run --rm \
  -v $(pwd)/test_images:/app/images:ro \
  -v $(pwd)/results:/app/output \
  moondream-perf -i /app/images -o /app/output/performance.csv
```

### Process Specific Directory Structure
```bash
# For nested directory structures
docker run --rm \
  -v /home/user/photos:/app/images:ro \
  -v /home/user/results:/app/output \
  moondream-perf \
  --input /app/images \
  --output /app/output/photo_analysis.csv
```

### Run with Resource Limits
```bash
# Limit memory usage to 8GB
docker run --rm \
  --memory=8g \
  -v /path/to/images:/app/images:ro \
  -v /path/to/output:/app/output \
  moondream-perf -i /app/images -o /app/output/results.csv
```

## Output Format

The application generates a CSV file with the following columns:

| Column | Description |
|--------|-------------|
| `filename` | Original image filename |
| `file_path` | Full path to the image file |
| `file_size_mb` | Image file size in megabytes |
| `image_dimensions` | Image dimensions (width x height) |
| `caption_short` | Short caption generated by Moondream |
| `caption_normal` | Detailed caption generated by Moondream |
| `processing_time_seconds` | Time to process this specific image |
| `memory_usage_mb` | Memory usage during processing |
| `timestamp` | When the image was processed |
| `status` | Processing status (success/error) |
| `error_message` | Error details if processing failed |
| `model_load_time_seconds` | Model loading time (first row only) |

## Performance Metrics

The application tracks and reports:

- **Model Loading Time**: Time to download and load Moondream
- **Per-Image Processing Time**: Individual image processing duration
- **Total Runtime**: Complete execution time
- **Average Processing Speed**: Images per second
- **Memory Usage**: RAM consumption during processing
- **Success/Failure Rates**: Processing statistics
- **File Size Analysis**: Performance correlation with image sizes

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)
- WebP (.webp)

## Troubleshooting

### Common Issues

**Out of Memory Errors**:
```bash
# Reduce batch size by limiting Docker memory
docker run --memory=6g --rm ...
```

**Permission Errors**:
```bash
# Ensure output directory is writable
chmod 755 /path/to/output
```

**No Images Found**:
- Check image formats are supported
- Verify directory path and permissions
- Ensure images are in mounted volume

**Slow Performance**:
- Model loads from internet on first run (5GB download)
- Subsequent runs use cached model
- Processing speed depends on image size and CPU cores

### Performance Optimization

1. **First Run**: Model download takes time - be patient
2. **Image Size**: Resize very large images for faster processing
3. **Memory**: Allocate 8GB+ RAM for optimal performance
4. **CPU**: More cores = better performance

## Docker Advanced Usage

### Custom Model Cache
```bash
# Persist model cache between runs
docker run --rm \
  -v model_cache:/root/.cache \
  -v /path/to/images:/app/images:ro \
  -v /path/to/output:/app/output \
  moondream-perf -i /app/images -o /app/output/results.csv
```

### Resource Monitoring
```bash
# Monitor resource usage
docker stats --no-stream
```

## Development

### Local Development (Non-Docker)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run directly
python main.py --input ./images --output ./results.csv
```

### Building Custom Image

```bash
# Build with custom tag
docker build -t my-moondream-perf:v1.0 .

# Build with specific Python version
docker build --build-arg PYTHON_VERSION=3.11 -t moondream-perf .
```

## License

This project is provided as-is for performance testing purposes.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify your Docker setup and image paths
3. Ensure sufficient system resources (RAM/storage)
4. Test with a small batch of images first