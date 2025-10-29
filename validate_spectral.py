"""
Spectral Feature Validation and Analysis Tools
==============================================

Comprehensive validation and visualization tools for spectral features.
Ensures quality of computed eigendecompositions and provides insights
into spectral properties of the graph dataset.

Author: AI Research Team
Date: October 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for publication-quality figures
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


class SpectralValidator:
    """Validates and analyzes spectral features for quality assurance."""
    
    def __init__(self, cache_dir: str):
        """
        Initialize validator.
        
        Args:
            cache_dir: Directory containing cached spectral features
        """
        self.cache_dir = Path(cache_dir)
        assert self.cache_dir.exists(), f"Cache directory not found: {cache_dir}"
        
        self.cache_files = list(self.cache_dir.glob("*_spectral.npz"))
        logger.info(f"Found {len(self.cache_files)} cached spectral files")
    
    def validate_single_file(self, cache_file: Path) -> Dict:
        """
        Validate a single spectral feature cache file.
        
        Args:
            cache_file: Path to .npz file
        
        Returns:
            Validation results dictionary
        """
        data = np.load(cache_file)
        eigvals = data['eigenvalues']
        eigvecs = data['eigenvectors']
        
        results = {
            'file': cache_file.name,
            'num_nodes': int(data['num_nodes']),
            'k': len(eigvals)
        }
        
        # Check 1: Eigenvalue ordering
        results['sorted'] = np.all(eigvals[:-1] <= eigvals[1:])
        
        # Check 2: First eigenvalue near zero
        results['first_eigenvalue'] = float(eigvals[0])
        results['zero_eigenvalue'] = np.abs(eigvals[0]) < 1e-5
        
        # Check 3: All eigenvalues non-negative
        results['min_eigenvalue'] = float(eigvals.min())
        results['non_negative'] = eigvals.min() >= -1e-8
        
        # Check 4: Eigenvector orthogonality
        gram = eigvecs.T @ eigvecs
        identity = np.eye(len(eigvals))
        ortho_error = np.linalg.norm(gram - identity, 'fro')
        results['orthogonality_error'] = float(ortho_error)
        results['orthogonal'] = ortho_error < 1e-3
        
        # Check 5: Spectral gap (difference between first two non-zero eigenvalues)
        nonzero_eigvals = eigvals[eigvals > 1e-8]
        if len(nonzero_eigvals) >= 2:
            results['spectral_gap'] = float(nonzero_eigvals[1] - nonzero_eigvals[0])
        else:
            results['spectral_gap'] = None
        
        # Overall pass/fail
        results['passed'] = all([
            results['sorted'],
            results['zero_eigenvalue'],
            results['non_negative'],
            results['orthogonal']
        ])
        
        return results
    
    def validate_all(self, max_files: int = None) -> Dict:
        """
        Validate all cached spectral files.
        
        Args:
            max_files: Maximum number of files to validate (None = all)
        
        Returns:
            Summary statistics
        """
        files_to_check = self.cache_files[:max_files] if max_files else self.cache_files
        
        results = []
        failed_files = []
        
        logger.info(f"Validating {len(files_to_check)} files...")
        
        for cache_file in files_to_check:
            try:
                result = self.validate_single_file(cache_file)
                results.append(result)
                
                if not result['passed']:
                    failed_files.append(result['file'])
                    
            except Exception as e:
                logger.error(f"Failed to validate {cache_file.name}: {e}")
                failed_files.append(cache_file.name)
        
        # Compute summary statistics
        # Handle edge case: no results found
        if not results:
            logger.warning("No validation results to summarize. Returning empty stats.")
            summary = {
                'total_files': 0,
                'passed': 0,
                'failed': 0,
                'failed_files': [],
                'eigenvalue_stats': {
                    'first_eigenvalue_mean': np.nan,
                    'first_eigenvalue_std': np.nan,
                    'min_eigenvalue_mean': np.nan,
                    'spectral_gap_mean': np.nan,
                },
                'orthogonality_stats': {
                    'mean_error': np.nan,
                    'max_error': np.nan,
                    'std_error': np.nan,
                }
            }
            # Return early to avoid numpy errors
            return summary, results


        # Compute summary statistics (only if results exist)
        summary = {
            'total_files': len(files_to_check),
            'passed': sum(1 for r in results if r['passed']),
            'failed': len(failed_files),
            'failed_files': failed_files,
            
            'eigenvalue_stats': {
                'first_eigenvalue_mean': np.mean([r['first_eigenvalue'] for r in results]),
                'first_eigenvalue_std': np.std([r['first_eigenvalue'] for r in results]),
                'min_eigenvalue_mean': np.mean([r['min_eigenvalue'] for r in results]),
                'spectral_gap_mean': np.mean([r['spectral_gap'] for r in results if r['spectral_gap'] is not None]),
            },
            
            'orthogonality_stats': {
                'mean_error': np.mean([r['orthogonality_error'] for r in results]),
                'max_error': np.max([r['orthogonality_error'] for r in results]),
                'std_error': np.std([r['orthogonality_error'] for r in results]),
            }
        }
        
        logger.info(f"Validation complete: {summary['passed']}/{summary['total_files']} passed")
        
        return summary, results
    
    def plot_eigenvalue_distribution(
        self,
        output_path: str = "eigenvalue_distribution.pdf",
        max_samples: int = 100
    ):
        """
        Plot distribution of eigenvalues across dataset.
        
        Args:
            output_path: Path to save figure
            max_samples: Maximum number of graphs to sample
        """
        samples = np.random.choice(
            self.cache_files,
            size=min(max_samples, len(self.cache_files)),
            replace=False
        )
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        all_eigvals = []
        first_eigvals = []
        spectral_gaps = []
        
        for cache_file in samples:
            data = np.load(cache_file)
            eigvals = data['eigenvalues']
            all_eigvals.extend(eigvals)
            first_eigvals.append(eigvals[0])
            
            nonzero = eigvals[eigvals > 1e-8]
            if len(nonzero) >= 2:
                spectral_gaps.append(nonzero[1] - nonzero[0])
        
        # Plot 1: All eigenvalues histogram
        axes[0, 0].hist(all_eigvals, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Eigenvalue')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of All Eigenvalues')
        axes[0, 0].axvline(0, color='red', linestyle='--', label='Zero')
        axes[0, 0].legend()
        
        # Plot 2: First eigenvalue distribution
        axes[0, 1].hist(first_eigvals, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 1].set_xlabel('First Eigenvalue (λ₀)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('First Eigenvalue Distribution\n(Should be near zero)')
        axes[0, 1].axvline(0, color='red', linestyle='--')
        
        # Plot 3: Spectral gap distribution
        axes[1, 0].hist(spectral_gaps, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1, 0].set_xlabel('Spectral Gap')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Spectral Gap Distribution\n(Algebraic Connectivity)')
        
        # Plot 4: Eigenvalue spectrum (sample graphs)
        for i, cache_file in enumerate(samples[:10]):
            data = np.load(cache_file)
            eigvals = data['eigenvalues']
            axes[1, 1].plot(eigvals, alpha=0.5, linewidth=1)
        
        axes[1, 1].set_xlabel('Eigenvalue Index')
        axes[1, 1].set_ylabel('Eigenvalue')
        axes[1, 1].set_title('Eigenvalue Spectra (10 sample graphs)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved eigenvalue distribution plot to {output_path}")
        plt.close()
    
    def plot_eigenvector_examples(
        self,
        output_path: str = "eigenvector_examples.pdf",
        num_examples: int = 3
    ):
        """
        Visualize first few eigenvectors for example graphs.
        
        Args:
            output_path: Path to save figure
            num_examples: Number of example graphs
        """
        samples = np.random.choice(self.cache_files, size=num_examples, replace=False)
        
        fig, axes = plt.subplots(num_examples, 4, figsize=(16, 4 * num_examples))
        if num_examples == 1:
            axes = axes.reshape(1, -1)
        
        for i, cache_file in enumerate(samples):
            data = np.load(cache_file)
            eigvecs = data['eigenvectors']
            eigvals = data['eigenvalues']
            
            # Plot first 4 eigenvectors
            for j in range(min(4, eigvecs.shape[1])):
                axes[i, j].plot(eigvecs[:, j], linewidth=2)
                axes[i, j].axhline(0, color='red', linestyle='--', alpha=0.5)
                axes[i, j].set_title(f'λ_{j} = {eigvals[j]:.4f}')
                axes[i, j].set_xlabel('Node Index')
                axes[i, j].set_ylabel('Eigenvector Value')
                axes[i, j].grid(True, alpha=0.3)
            
            # Add graph ID to leftmost plot
            axes[i, 0].text(
                0.02, 0.98, cache_file.stem,
                transform=axes[i, 0].transAxes,
                verticalalignment='top',
                fontsize=8, color='gray'
            )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved eigenvector examples to {output_path}")
        plt.close()
    
    def generate_validation_report(self, output_dir: str):
        """
        Generate comprehensive validation report with plots and statistics.
        
        Args:
            output_dir: Directory to save report
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Generating validation report...")
        
        # Run validation
        summary, results = self.validate_all()
        
        # Save validation statistics
        stats_file = output_path / "validation_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved statistics to {stats_file}")
        
        # Generate plots
        self.plot_eigenvalue_distribution(
            output_path=str(output_path / "eigenvalue_distribution.pdf")
        )
        
        self.plot_eigenvector_examples(
            output_path=str(output_path / "eigenvector_examples.pdf")
        )
        
        # Generate markdown report
        report_file = output_path / "spectral_validation_report.md"
        with open(report_file, 'w') as f:
            f.write("# Spectral Feature Validation Report\n\n")
            f.write(f"**Date**: {np.datetime64('today')}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- **Total Files**: {summary['total_files']}\n")
            f.write(f"- **Passed**: {summary['passed']}\n")
            f.write(f"- **Failed**: {summary['failed']}\n")
            f.write(f"- **Success Rate**: {100*summary['passed']/summary['total_files']:.1f}%\n\n")
            
            f.write("## Eigenvalue Statistics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for key, value in summary['eigenvalue_stats'].items():
                f.write(f"| {key} | {value:.6f} |\n")
            
            f.write("\n## Orthogonality Statistics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for key, value in summary['orthogonality_stats'].items():
                f.write(f"| {key} | {value:.6e} |\n")
            
            if summary['failed_files']:
                f.write("\n## Failed Files\n\n")
                for fname in summary['failed_files']:
                    f.write(f"- {fname}\n")
        
        logger.info(f"Saved validation report to {report_file}")
        logger.info("Validation report generation complete!")


def compare_normalization_methods(
    edge_index,
    num_nodes: int,
    output_path: str = "normalization_comparison.pdf"
):
    """
    Compare different Laplacian normalization methods.
    
    Args:
        edge_index: Graph edge connectivity
        num_nodes: Number of nodes
        output_path: Path to save comparison plot
    """
    from spectral_features import SpectralFeatureExtractor
    
    methods = ['symmetric', 'random_walk', 'unnormalized']
    results = {}
    
    for method in methods:
        extractor = SpectralFeatureExtractor(k=16, normalization=method)
        features = extractor.extract_features(edge_index, num_nodes)
        results[method] = features
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, method in enumerate(methods):
        eigvals = results[method]['eigenvalues']
        axes[idx].plot(eigvals, 'o-', linewidth=2, markersize=6)
        axes[idx].set_xlabel('Eigenvalue Index')
        axes[idx].set_ylabel('Eigenvalue')
        axes[idx].set_title(f'{method.capitalize()} Laplacian')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].axhline(0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved normalization comparison to {output_path}")
    plt.close()


# Example usage
if __name__ == "__main__":
    # --- START FIX: Add argument parsing ---
    import argparse
    parser = argparse.ArgumentParser(description="Spectral Feature Validator")
    parser.add_argument(
        '--cache-dir', 
        type=str, 
        required=True, 
        help='Directory containing cached spectral features'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        required=True, 
        help='Directory to save validation reports'
    )
    args = parser.parse_args()
    # --- END FIX ---

    print("Spectral Feature Validator - Testing")
    print("=" * 60)
    
    # Example: validate a cache directory
    cache_dir = args.cache_dir # <-- FIX: Use parsed argument
    
    if Path(cache_dir).exists():
        validator = SpectralValidator(cache_dir)
        
        # Generate full validation report
        validator.generate_validation_report(output_dir=args.output_dir) # <-- FIX: Use parsed argument
        
        print(f"\nValidation complete! Check {args.output_dir}/ for reports.")
    else:
        print(f"Cache directory not found: {cache_dir}")
        print("Run spectral feature extraction first!")
    
    print("=" * 60)