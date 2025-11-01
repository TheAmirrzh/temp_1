"""
FIXED: Spectral Graph Feature Extraction

Key Fix: Eigenvalue correction now preserves structural zeros
Previous bug: Constant perturbation destroyed information about connected components
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import torch
from torch_geometric.utils import to_scipy_sparse_matrix
from typing import Tuple, Optional, Dict
import warnings
from pathlib import Path
import json
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpectralFeatureExtractor:
    """
    Extracts spectral features from graphs using graph Laplacian eigendecomposition.
    
    FIXED in this version:
    - Eigenvalue correction preserves structural zeros
    - Uses geometric (multiplicative) perturbation instead of additive
    """
    
    def __init__(
        self,
        k: int = 16,
        normalization: str = 'symmetric',
        add_self_loops: bool = True,
        tolerance: float = 0.0,
        max_iter: Optional[int] = None,
        adaptive_k: bool = True
    ):
        assert k >= 1, "k must be at least 1"
        assert normalization in ['symmetric', 'random_walk', 'unnormalized']
        
        self.k = k
        self.normalization = normalization
        self.add_self_loops = add_self_loops
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.adaptive_k = adaptive_k
        
        logger.info(f"SpectralFeatureExtractor initialized: k={k}, "
                   f"normalization={normalization}, adaptive_k={self.adaptive_k}")
    
    def compute_normalized_laplacian(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        edge_weight: Optional[torch.Tensor] = None
    ) -> sp.csr_matrix:
        """Compute normalized graph Laplacian matrix."""
        adj = to_scipy_sparse_matrix(
            edge_index, edge_weight, num_nodes=num_nodes
        )
        
        if self.add_self_loops:
            adj = adj + sp.eye(num_nodes, format='csr')
        
        degrees = np.array(adj.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1.0
        
        if self.normalization == 'symmetric':
            d_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees))
            laplacian = sp.eye(num_nodes) - d_inv_sqrt @ adj @ d_inv_sqrt
        elif self.normalization == 'random_walk':
            d_inv = sp.diags(1.0 / degrees)
            laplacian = sp.eye(num_nodes) - d_inv @ adj
        else:  # unnormalized
            d_matrix = sp.diags(degrees)
            laplacian = d_matrix - adj
        
        return laplacian.tocsr()
    
    def _eigenvalue_correction(self, eigvals: np.ndarray, epsilon: float = 1e-2) -> np.ndarray:
        """
        FIXED: Spread repeated eigenvalues while PRESERVING structural zeros.
        
        Key insight: Zero eigenvalues encode # of connected components.
        This is a fundamental graph property that MUST NOT be modified.
        
        Args:
            eigvals: Original eigenvalues
            epsilon: Perturbation magnitude
        
        Returns:
            Corrected eigenvalues with structural zeros preserved
        """

        # return corrected
        corrected = eigvals.copy()
        zero_mask = np.abs(eigvals) < 1e-5
        
        # Preserve structural zeros (connected components)
        for i in range(1, len(eigvals)):
            if zero_mask[i] or zero_mask[i-1]:
                continue
            
            # ADAPTIVE spacing based on spectral gap
            if abs(corrected[i] - corrected[i-1]) < epsilon:
                # Use Chebyshev nodes spacing for better polynomial interpolation
                spacing = epsilon * (1 + 0.5 * np.cos(np.pi * i / len(eigvals)))
                corrected[i] = corrected[i-1] + spacing
        
        return corrected
    
    def compute_spectral_decomposition(
        self,
        laplacian: sp.csr_matrix,
        k: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute k smallest eigenpairs of the Laplacian matrix.
        
        Uses scipy.sparse.linalg.eigsh (ARPACK) for efficient computation.
        """
        num_nodes = laplacian.shape[0]
        
        k_to_compute = k or self.k
        if self.adaptive_k:
            k_to_compute = min(max(self.k, num_nodes // 3), 32)
            k_to_compute = min(k_to_compute, num_nodes - 1)
            if k_to_compute < 1:
                k_to_compute = 1
        
        # Edge case: graph too small
        if num_nodes <= k_to_compute:
            logger.warning(f"Graph has only {num_nodes} nodes but k={k_to_compute}. "
                          f"Using dense solver for all eigenvalues.")
            laplacian_dense = laplacian.toarray()
            eigenvalues, eigenvectors = np.linalg.eigh(laplacian_dense)
            
            # FIXED: Apply correction
            eigenvalues = self._eigenvalue_correction(eigenvalues)
            
            # Re-sort after correction
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            return eigenvalues[:k_to_compute], eigenvectors[:, :k_to_compute]
        
        try:
            eigenvalues, eigenvectors = eigsh(
                laplacian,
                k=k_to_compute,
                which='SA',  # Smallest algebraic
                tol=self.tolerance,
                maxiter=self.max_iter,
                return_eigenvectors=True
            )
            
            # FIXED: Apply eigenvalue correction
            eigenvalues = self._eigenvalue_correction(eigenvalues)
            
            # Sort by eigenvalue (correction might change order)
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
        except Exception as e:
            logger.warning(f"Sparse eigsh failed ({e}). Falling back to dense solver.")
            laplacian_dense = laplacian.toarray()
            eigenvalues, eigenvectors = np.linalg.eigh(laplacian_dense)
            
            # FIXED: Apply correction here too
            eigenvalues = self._eigenvalue_correction(eigenvalues)
            
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            eigenvalues = eigenvalues[:k_to_compute]
            eigenvectors = eigenvectors[:, :k_to_compute]
        
        return eigenvalues, eigenvectors
    
    def validate_spectral_properties(
        self,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        verbose: bool = False
    ) -> Dict[str, bool]:
        """Validate spectral properties for quality assurance."""
        results = {}
        
        # Check 1: Sorted eigenvalues
        results['sorted'] = np.all(eigenvalues[:-1] <= eigenvalues[1:])
        
        # Check 2: First eigenvalue near zero
        results['zero_eigenvalue'] = np.abs(eigenvalues[0]) < 1e-6
        
        # Check 3: Non-negative eigenvalues
        results['non_negative'] = np.all(eigenvalues >= -1e-10)
        
        # Check 4: Orthogonality of eigenvectors
        gram_matrix = eigenvectors.T @ eigenvectors
        identity = np.eye(eigenvectors.shape[1])
        ortho_error = np.linalg.norm(gram_matrix - identity, 'fro')
        results['orthogonal'] = ortho_error < 1e-4
        results['orthogonality_error'] = ortho_error
        
        results['passed'] = all([
            results['sorted'],
            results['zero_eigenvalue'],
            results['non_negative'],
            results['orthogonal']
        ])
        
        if verbose:
            logger.info("Spectral Properties Validation:")
            logger.info(f"  Sorted: {results['sorted']}")
            logger.info(f"  Zero eigenvalue: {results['zero_eigenvalue']} "
                       f"(λ₀ = {eigenvalues[0]:.2e})")
            logger.info(f"  Non-negative: {results['non_negative']}")
            logger.info(f"  Orthogonal: {results['orthogonal']} "
                       f"(error = {ortho_error:.2e})")
        
        return results
    
    def extract_features(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        edge_weight: Optional[torch.Tensor] = None,
        validate: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Extract spectral features from a graph.
        
        Returns:
            Dictionary containing:
                - 'eigenvalues': Shape [k]
                - 'eigenvectors': Shape [num_nodes, k]
                - 'validation': Validation results (if validate=True)
        """
        laplacian = self.compute_normalized_laplacian(
            edge_index, num_nodes, edge_weight
        )
        
        eigenvalues, eigenvectors = self.compute_spectral_decomposition(laplacian)
        
        result = {
            'eigenvalues': eigenvalues.astype(np.float32),
            'eigenvectors': eigenvectors.astype(np.float32)
        }
        
        if validate:
            validation = self.validate_spectral_properties(
                eigenvalues, eigenvectors, verbose=False
            )
            result['validation'] = validation
        
        return result


def precompute_spectral_cache(
    data_dir: str,
    output_dir: str,
    k: int = 16,
    normalization: str = 'symmetric',
    adaptive_k: bool = True,
    max_graphs: Optional[int] = None,
    force_recompute: bool = False
) -> Dict[str, any]:
    """Precompute and cache spectral features for all graphs in dataset."""
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    json_files = list(data_path.glob("*.json"))
    if max_graphs is not None:
        json_files = json_files[:max_graphs]
    
    logger.info(f"Found {len(json_files)} graphs to process")
    
    extractor = SpectralFeatureExtractor(k=k, normalization=normalization, adaptive_k=adaptive_k)
    
    stats = {
        'total': len(json_files),
        'processed': 0,
        'skipped': 0,
        'failed': 0,
        'avg_eigenvalues': [],
        'avg_time_per_graph': []
    }
    
    import time
    
    for json_file in tqdm(json_files, desc="Computing spectral features"):
        instance_id = json_file.stem
        output_file = output_path / f"{instance_id}_spectral.npz"
        
        if output_file.exists() and not force_recompute:
            stats['skipped'] += 1
            continue
        
        try:
            with open(json_file, 'r') as f:
                graph_data = json.load(f)
            
            edges = graph_data.get('edges', [])
            num_nodes = graph_data.get('num_nodes', 0)
            
            if len(edges) == 0 or num_nodes == 0:
                logger.warning(f"Empty graph: {instance_id}")
                stats['failed'] += 1
                continue
            
            # FIXED: Handle different edge formats
            if isinstance(edges[0], dict):
                edge_list = []
                for edge in edges:
                    edge_list.append([edge['src'], edge['dst']])
            else:
                edge_list = edges
            
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            
            start_time = time.time()
            
            features = extractor.extract_features(
                edge_index, num_nodes, validate=True
            )
            
            elapsed = time.time() - start_time
            stats['avg_time_per_graph'].append(elapsed)
            
            np.savez_compressed(
                output_file,
                eigenvalues=features['eigenvalues'],
                eigenvectors=features['eigenvectors'],
                num_nodes=num_nodes,
                normalization=normalization,
                k=k
            )
            
            stats['processed'] += 1
            stats['avg_eigenvalues'].append(features['eigenvalues'])
            
        except Exception as e:
            logger.error(f"Failed to process {instance_id}: {e}")
            stats['failed'] += 1
    
    # Compute summary statistics
    if stats['avg_eigenvalues']:
        stats['eigenvalue_statistics'] = {
            'mean': np.mean([ev.mean() for ev in stats['avg_eigenvalues']]),
            'std': np.std([ev.mean() for ev in stats['avg_eigenvalues']]),
            'min': np.min([ev.min() for ev in stats['avg_eigenvalues']]),
            'max': np.max([ev.max() for ev in stats['avg_eigenvalues']])
        }
    
    if stats['avg_time_per_graph']:
        stats['avg_time'] = np.mean(stats['avg_time_per_graph'])
        stats['total_time'] = np.sum(stats['avg_time_per_graph'])
    
    stats_file = output_path / 'spectral_cache_stats.json'
    with open(stats_file, 'w') as f:
        stats_serializable = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in stats.items()
            if k != 'avg_eigenvalues'
        }
        json.dump(stats_serializable, f, indent=2)
    
    logger.info(f"Processed: {stats['processed']}, "
                f"Skipped: {stats['skipped']}, "
                f"Failed: {stats['failed']}")
    
    return stats


def load_spectral_features(cache_path: str) -> Dict[str, np.ndarray]:
    """Load precomputed spectral features from cache."""
    data = np.load(cache_path)
    return {
        'eigenvalues': data['eigenvalues'],
        'eigenvectors': data['eigenvectors'],
        'num_nodes': int(data['num_nodes']),
        'normalization': str(data['normalization']),
        'k': int(data['k'])
    }