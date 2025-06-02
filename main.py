#!/usr/bin/env python3

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import psutil
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer

console = Console()


class MoondreamPerformanceTester:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_load_time = 0
        self.results: List[Dict] = []
        self.start_time = 0
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
    def load_model(self) -> bool:
        """Load the Moondream model and measure loading time."""
        console.print("[bold blue]Loading Moondream model...[/bold blue]")
        
        start_time = time.time()
        try:
            with console.status("[bold green]Downloading and loading model weights..."):
                self.model = AutoModelForCausalLM.from_pretrained(
                    "vikhyatk/moondream2",
                    revision="2025-01-09",
                    trust_remote_code=True
                )
                
            self.model_load_time = time.time() - start_time
            console.print(f"[green]✓[/green] Model loaded in {self.model_load_time:.2f} seconds")
            return True
            
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to load model: {str(e)}")
            return False
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def get_image_files(self, input_dir: Path) -> List[Path]:
        """Get all supported image files from the input directory."""
        image_files = []
        for file_path in input_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                image_files.append(file_path)
        return sorted(image_files)
    
    def process_image(self, image_path: Path) -> Dict:
        """Process a single image and return results with timing."""
        result = {
            'filename': image_path.name,
            'file_path': str(image_path),
            'file_size_mb': 0,
            'image_dimensions': '',
            'caption_short': '',
            'caption_normal': '',
            'processing_time_seconds': 0,
            'memory_usage_mb': 0,
            'timestamp': datetime.now().isoformat(),
            'status': 'error',
            'error_message': ''
        }
        
        try:
            # Get file size
            result['file_size_mb'] = round(image_path.stat().st_size / 1024 / 1024, 2)
            
            # Load image
            image = Image.open(image_path)
            result['image_dimensions'] = f"{image.width}x{image.height}"
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Start timing the actual processing
            start_time = time.time()
            memory_before = self.get_memory_usage()
            
            # Generate captions
            short_caption = self.model.caption(image, length="short")["caption"]
            normal_caption = self.model.caption(image, length="normal")["caption"]
            
            # End timing
            processing_time = time.time() - start_time
            memory_after = self.get_memory_usage()
            
            result.update({
                'caption_short': short_caption,
                'caption_normal': normal_caption,
                'processing_time_seconds': round(processing_time, 3),
                'memory_usage_mb': round(memory_after - memory_before, 2),
                'status': 'success'
            })
            
        except Exception as e:
            result['error_message'] = str(e)
            console.print(f"[red]Error processing {image_path.name}: {str(e)}[/red]")
        
        return result
    
    def process_images(self, input_dir: Path, output_file: Path) -> None:
        """Process all images in the input directory."""
        self.start_time = time.time()
        
        # Get all image files
        image_files = self.get_image_files(input_dir)
        
        if not image_files:
            console.print(f"[red]No supported image files found in {input_dir}[/red]")
            console.print(f"Supported formats: {', '.join(self.supported_formats)}")
            return
        
        console.print(f"[bold green]Found {len(image_files)} image files to process[/bold green]")
        
        # Process images with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            task = progress.add_task("Processing images...", total=len(image_files))
            
            for i, image_path in enumerate(image_files):
                progress.update(task, description=f"Processing {image_path.name}")
                
                result = self.process_image(image_path)
                if i == 0:  # Add model load time to first result
                    result['model_load_time_seconds'] = round(self.model_load_time, 3)
                
                self.results.append(result)
                progress.advance(task)
        
        # Save results and show summary
        self.save_results(output_file)
        self.show_summary()
    
    def save_results(self, output_file: Path) -> None:
        """Save results to CSV file."""
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Create DataFrame and save to CSV
            df = pd.DataFrame(self.results)
            df.to_csv(output_file, index=False)
            
            console.print(f"[green]✓[/green] Results saved to {output_file}")
            
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to save results: {str(e)}")
    
    def show_summary(self) -> None:
        """Display performance summary."""
        if not self.results:
            return
        
        total_time = time.time() - self.start_time
        successful_results = [r for r in self.results if r['status'] == 'success']
        failed_results = [r for r in self.results if r['status'] == 'error']
        
        # Calculate statistics
        processing_times = [r['processing_time_seconds'] for r in successful_results]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        total_processing_time = sum(processing_times)
        
        # Create summary table
        table = Table(title="Performance Summary", title_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Images", str(len(self.results)))
        table.add_row("Successful", str(len(successful_results)))
        table.add_row("Failed", str(len(failed_results)))
        table.add_row("Model Load Time", f"{self.model_load_time:.2f}s")
        table.add_row("Total Runtime", f"{total_time:.2f}s")
        table.add_row("Total Processing Time", f"{total_processing_time:.2f}s")
        table.add_row("Average Time per Image", f"{avg_processing_time:.3f}s")
        if avg_processing_time > 0:
            table.add_row("Images per Second", f"{1/avg_processing_time:.2f}")
        
        console.print()
        console.print(table)
        
        # Show failed images if any
        if failed_results:
            console.print()
            console.print("[bold red]Failed Images:[/bold red]")
            for result in failed_results:
                console.print(f"  • {result['filename']}: {result['error_message']}")


def main():
    parser = argparse.ArgumentParser(
        description="Test Moondream model performance on a batch of images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input /path/to/images --output results.csv
  python main.py -i ./images -o ./output/performance.csv
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=Path,
        required=True,
        help='Input directory containing images'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        required=True,
        help='Output CSV file path'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.input.exists():
        console.print(f"[red]Error: Input directory '{args.input}' does not exist[/red]")
        sys.exit(1)
    
    if not args.input.is_dir():
        console.print(f"[red]Error: '{args.input}' is not a directory[/red]")
        sys.exit(1)
    
    # Initialize tester
    console.print("[bold cyan]Moondream Performance Tester[/bold cyan]")
    console.print(f"Input directory: {args.input}")
    console.print(f"Output file: {args.output}")
    console.print()
    
    tester = MoondreamPerformanceTester()
    
    # Load model
    if not tester.load_model():
        sys.exit(1)
    
    # Process images
    try:
        tester.process_images(args.input, args.output)
        console.print("\n[bold green]✓ Performance testing completed successfully![/bold green]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Testing interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]✗ Unexpected error: {str(e)}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()