"""
Batch Processing Pipeline for Spectral Features
===============================================

Production-ready pipeline for processing entire dataset with:
- Parallel processing
- Error handling and recovery
- Progress tracking
- Resource monitoring
- Automatic checkpointing

Author: AI Research Team
Date: October 2025
"""

import numpy as np
import torch
from pathlib import Path
import json
import argparse
import multiprocessing as mp
from functools import partial
import time
import psutil
import logging
from tqdm import tqdm
from typing import List, Dict, Optional

from spectral_features import SpectralFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SpectralBatchProcessor:
    """
    Batch processor for computing spectral features with parallel processing.
    """
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        k: int = 16,
        normalization: str = 'symmetric',
        adaptive_k: bool = True,
        num_workers: int = None,
        checkpoint_interval: int = 100
    ):
        """
        Initialize batch processor.
        
        Args:
            data_dir: Directory containing graph JSON files
            output_dir: Directory to save spectral features
            k: Number of eigenpairs
            normalization: Laplacian normalization method
            num_workers: Number of parallel workers (None = CPU count)
            checkpoint_interval: Save progress every N graphs
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.k = k
        self.normalization = normalization
        self.adaptive_k = adaptive_k
        self.num_workers = num_workers or mp.cpu_count()
        self.checkpoint_interval = checkpoint_interval
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize extractor (will be recreated in worker processes)
        self.extractor = SpectralFeatureExtractor(
            k=k, normalization=normalization, adaptive_k=adaptive_k
        )
        
        # Load checkpoint if exists
        self.checkpoint_file = self.output_dir / "processing_checkpoint.json"
        self.processed_files = self._load_checkpoint()
        
        logger.info(f"Initialized with {self.num_workers} workers")
        logger.info(f"Already processed: {len(self.processed_files)} files")
    
    def _load_checkpoint(self) -> set:
        """Load checkpoint of already processed files."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
                return set(data.get('processed_files', []))
        return set()
    
    def _save_checkpoint(self, processed_files: set):
        """Save checkpoint of processed files."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump({
                'processed_files': list(processed_files),
                'timestamp': time.time(),
                'total': len(processed_files)
            }, f, indent=2)
    
    def _process_single_graph(
        self,
        json_file: Path,
        extractor: SpectralFeatureExtractor
    ) -> Dict:
        """
        Process a single graph file.
        
        Args:
            json_file: Path to graph JSON file
            extractor: SpectralFeatureExtractor instance
        
        Returns:
            Result dictionary with status and timing
        """
        instance_id = json_file.stem
        output_file = self.output_dir / f"{instance_id}_spectral.npz"
        
        result = {
            'instance_id': instance_id,
            'status': 'success',
            'error': None,
            'time': 0.0
        }
        
        try:
            start_time = time.time()
            
            # Load graph data
            with open(json_file, 'r') as f:
                graph_data = json.load(f)
            
            # Parse graph structure (adapt to your format)
            edges_data = graph_data.get('edges', [])
            num_nodes = len(graph_data.get('nodes', []))

            if not edges_data: # Handles empty list
                result['status'] = 'empty_graph'
                result['error'] = 'No edges in graph'
                return result

            # --- START FIX: Check for dict-based edge format ---
            if isinstance(edges_data[0], dict):
                # Format is [{"src": u, "dst": v}, ...]
                # Assuming undirected graph, add edges in both directions
                edges = []
                for edge in edges_data:
                    edges.append([edge['src'], edge['dst']])
                    edges.append([edge['dst'], edge['src']])
            else:
                # Format is already [[u, v], [v, u], ...]
                edges = edges_data
            # --- END FIX ---

            if len(edges) == 0:
                result['status'] = 'empty_graph'
                result['error'] = 'No edges in graph'
                return result
            
            # Convert to tensor
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            
            # Extract spectral features
            features = extractor.extract_features(
                edge_index,
                num_nodes,
                validate=True
            )
            
            # Check validation
            if not all(features['validation'].values()):
                result['status'] = 'validation_failed'
                result['error'] = f"Validation failed: {features['validation']}"
                logger.warning(f"Validation failed for {instance_id}")
            
            # Save features
            np.savez_compressed(
                output_file,
                eigenvalues=features['eigenvalues'],
                eigenvectors=features['eigenvectors'],
                num_nodes=num_nodes,
                normalization=self.normalization,
                k=self.k,
                instance_id=instance_id
            )
            
            result['time'] = time.time() - start_time
            
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            logger.error(f"Failed to process {instance_id}: {e}")
        
        return result
    
    def process_batch(
        self,
        max_files: Optional[int] = None,
        force_recompute: bool = False
    ) -> Dict:
        """
        Process entire dataset in parallel.
        
        Args:
            max_files: Maximum number of files to process (None = all)
            force_recompute: Recompute even if cache exists
        
        Returns:
            Statistics dictionary
        """
        # Find all JSON files
        json_files = list(self.data_dir.glob("**/*.json"))        
        if not force_recompute:
            # Filter out already processed files
            json_files = [
                f for f in json_files 
                if f.stem not in self.processed_files
            ]
        
        if max_files is not None:
            json_files = json_files[:max_files]
        
        logger.info(f"Processing {len(json_files)} graphs...")
        
        if len(json_files) == 0:
            logger.info("No files to process!")
            return {'total': 0}
        
        # Statistics
        stats = {
            'total': len(json_files),
            'success': 0,
            'failed': 0,
            'empty_graph': 0,
            'validation_failed': 0,
            'times': [],
            'errors': []
        }
        
        # Process with progress bar
        processed_count = 0
        
        # Create partial function with extractor
        # +++ MODIFIED (Phase 2.1) +++
        process_fn = partial(
            self._process_single_graph,
            extractor=SpectralFeatureExtractor(
                k=self.k,
                normalization=self.normalization,
                adaptive_k=self.adaptive_k
            )
        )
        # Use multiprocessing pool
        with mp.Pool(processes=self.num_workers) as pool:
            # Use imap for progress tracking
            results_iter = pool.imap(process_fn, json_files)
            
            for result in tqdm(results_iter, total=len(json_files), desc="Processing"):
                # Update statistics
                stats[result['status']] = stats.get(result['status'], 0) + 1
                
                if result['status'] == 'success':
                    stats['times'].append(result['time'])
                    self.processed_files.add(result['instance_id'])
                else:
                    stats['errors'].append({
                        'instance_id': result['instance_id'],
                        'error': result['error']
                    })
                
                processed_count += 1
                
                # Save checkpoint periodically
                if processed_count % self.checkpoint_interval == 0:
                    self._save_checkpoint(self.processed_files)
                    logger.info(f"Checkpoint saved: {processed_count}/{len(json_files)}")
        
        # Final checkpoint
        self._save_checkpoint(self.processed_files)
        
        # Compute timing statistics
        if stats['times']:
            stats['timing'] = {
                'mean': float(np.mean(stats['times'])),
                'std': float(np.std(stats['times'])),
                'min': float(np.min(stats['times'])),
                'max': float(np.max(stats['times'])),
                'total': float(np.sum(stats['times']))
            }
        
        # Save final statistics
        stats_file = self.output_dir / 'batch_processing_stats.json'
        with open(stats_file, 'w') as f:
            # Remove times array (too large)
            stats_save = {k: v for k, v in stats.items() if k != 'times'}
            json.dump(stats_save, f, indent=2)
        
        logger.info(f"Processing complete!")
        logger.info(f"  Success: {stats['success']}")
        logger.info(f"  Failed: {stats['failed']}")
        logger.info(f"  Empty graphs: {stats.get('empty_graph', 0)}")
        logger.info(f"  Validation failed: {stats.get('validation_failed', 0)}")
        
        if stats['times']:
            logger.info(f"  Average time: {stats['timing']['mean']:.3f}s per graph")
            logger.info(f"  Total time: {stats['timing']['total']:.1f}s")
        
        return stats


def monitor_resources():
    """Monitor system resources during processing."""
    process = psutil.Process()
    
    info = {
        'cpu_percent': process.cpu_percent(),
        'memory_mb': process.memory_info().rss / 1024 / 1024,
        'num_threads': process.num_threads()
    }
    
    return info


def main():
    """Main entry point for batch processing."""
    parser = argparse.ArgumentParser(
        description="Batch process spectral features for graph dataset"
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing graph JSON files'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save spectral features'
    )
    
    parser.add_argument(
        '--k',
        type=int,
        default=16,
        help='Number of eigenpairs to compute (default: 16)'
    )
    
    parser.add_argument(
        '--normalization',
        type=str,
        default='symmetric',
        choices=['symmetric', 'random_walk', 'unnormalized'],
        help='Laplacian normalization method (default: symmetric)'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: CPU count)'
    )
    
    parser.add_argument(
        '--max_files',
        type=int,
        default=None,
        help='Maximum number of files to process (default: all)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force recompute even if cache exists'
    )
    
    parser.add_argument(
        '--checkpoint_interval',
        type=int,
        default=100,
        help='Save checkpoint every N graphs (default: 100)'
    )
    parser.add_argument(
        '--adaptive-k',
        action='store_true',
        default=True,
        help='Use adaptive k based on graph size (default: True)'
    )
    parser.add_argument(
        '--no-adaptive-k',
        action='store_false',
        dest='adaptive_k',
        help='Disable adaptive k'
    )
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = SpectralBatchProcessor(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        k=args.k,
        normalization=args.normalization,
        adaptive_k=args.adaptive_k,
        num_workers=args.num_workers,
        checkpoint_interval=args.checkpoint_interval
    )
    
    # Monitor resources before
    logger.info("System resources before processing:")
    resources_before = monitor_resources()
    for key, value in resources_before.items():
        logger.info(f"  {key}: {value}")
    
    # Process batch
    start_time = time.time()
    stats = processor.process_batch(
        max_files=args.max_files,
        force_recompute=args.force
    )
    elapsed = time.time() - start_time
    
    # Monitor resources after
    logger.info("System resources after processing:")
    resources_after = monitor_resources()
    for key, value in resources_after.items():
        logger.info(f"  {key}: {value}")
    
    logger.info(f"\nTotal elapsed time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    
    if stats.get('times'):
        throughput = stats['success'] / elapsed
        logger.info(f"Throughput: {throughput:.2f} graphs/second")


if __name__ == "__main__":
    main()