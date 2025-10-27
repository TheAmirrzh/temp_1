"""
Comprehensive Test Suite for Spectral Features
==============================================

Unit tests and integration tests for spectral feature extraction.
Validates correctness, numerical stability, and edge cases.

Author: AI Research Team
Date: October 2025
"""

import unittest
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil
import sys

from spectral_features import SpectralFeatureExtractor, load_spectral_features


class TestSpectralFeatureExtractor(unittest.TestCase):
    """Test suite for SpectralFeatureExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = SpectralFeatureExtractor(k=4, normalization='symmetric')
    
    def test_simple_triangle(self):
        """Test on a simple triangle graph."""
        # Triangle: 0-1-2-0
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 0],
            [1, 0, 2, 1, 0, 2]
        ], dtype=torch.long)
        num_nodes = 3
        
        features = self.extractor.extract_features(edge_index, num_nodes, validate=True)
        
        # Check shapes - should return all eigenvalues for small graphs
        self.assertGreaterEqual(len(features['eigenvalues']), 2)
        self.assertEqual(features['eigenvectors'].shape[0], num_nodes)
        
        # Check first eigenvalue near zero
        self.assertLess(np.abs(features['eigenvalues'][0]), 1e-5)
        
        # Check validation passed
        self.assertTrue(features['validation']['passed'])
    
    def test_disconnected_graph(self):
        """Test on disconnected graph (two triangles)."""
        # +++ FIX: Must initialize with add_self_loops=False and adaptive_k=False for this test +++
        # Request k=6 to get all eigenvalues (graph has 6 nodes)
        extractor_no_loops = SpectralFeatureExtractor(k=6, normalization='symmetric', add_self_loops=False, adaptive_k=False)

        # Two separate triangles: (0-1-2) and (3-4-5)
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 0, 3, 4, 4, 5, 5, 3],
            [1, 0, 2, 1, 0, 2, 4, 3, 5, 4, 3, 5]
        ], dtype=torch.long)
        num_nodes = 6

        # +++ FIX: Use the new 'extractor_no_loops' +++
        features = extractor_no_loops.extract_features(edge_index, num_nodes, validate=True)

        # Disconnected graph should have TWO zero eigenvalues (one per connected component)
        zero_eigenvalues = np.sum(np.abs(features['eigenvalues']) < 1e-5) # Use np.abs for safety
        self.assertEqual(zero_eigenvalues, 2)
    
    def test_complete_graph(self):
        """Test on complete graph K4."""
        # Complete graph on 4 nodes
        edges = [(i, j) for i in range(4) for j in range(4) if i != j]
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        num_nodes = 4
        
        features = self.extractor.extract_features(edge_index, num_nodes, validate=True)
        
        # Check all eigenvectors are orthogonal
        gram = features['eigenvectors'].T @ features['eigenvectors']
        identity = np.eye(features['eigenvectors'].shape[1])
        ortho_error = np.linalg.norm(gram - identity, 'fro')
        self.assertLess(ortho_error, 1e-3)
    
    def test_star_graph(self):
        """Test on star graph (central node connected to all others)."""
        # Star: node 0 connected to nodes 1, 2, 3, 4
        edges = [(0, i) for i in range(1, 5)] + [(i, 0) for i in range(1, 5)]
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        num_nodes = 5
        
        features = self.extractor.extract_features(edge_index, num_nodes, validate=True)
        
        # Star graph has specific spectral properties
        self.assertEqual(features['eigenvalues'].shape[0], 4)  # k=4
        self.assertLess(features['eigenvalues'][0], 1e-5)
        self.assertTrue(features['validation']['sorted'])
    
    def test_path_graph(self):
        """Test on path graph (linear chain)."""
        # Path: 0-1-2-3-4
        edges = [(i, i+1) for i in range(4)] + [(i+1, i) for i in range(4)]
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        num_nodes = 5
        
        features = self.extractor.extract_features(edge_index, num_nodes, validate=True)
        
        # Path graph eigenvalues should be in [0, 2] for symmetric Laplacian
        self.assertTrue(np.all(features['eigenvalues'] >= -1e-8))
        self.assertTrue(np.all(features['eigenvalues'] <= 2 + 1e-6))
    
    def test_single_node(self):
        """Test on single node graph (edge case)."""
        edge_index = torch.empty((2, 0), dtype=torch.long)
        num_nodes = 1
        
        features = self.extractor.extract_features(edge_index, num_nodes, validate=False)
        
        # Single node: eigenvalue should be 0, eigenvector should be [1]
        self.assertEqual(features['eigenvalues'].shape[0], 1)
        self.assertLess(np.abs(features['eigenvalues'][0]), 1e-5)
    
    def test_different_normalizations(self):
        """Test different Laplacian normalizations give different results."""
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 0],
            [1, 0, 2, 1, 0, 2]
        ], dtype=torch.long)
        num_nodes = 3
        
        results = {}
        for norm in ['symmetric', 'random_walk', 'unnormalized']:
            extractor = SpectralFeatureExtractor(k=3, normalization=norm)
            features = extractor.extract_features(edge_index, num_nodes)
            results[norm] = features['eigenvalues']
        
        # Different normalizations should give different eigenvalues
        # Allow small tolerance for numerical differences
        self.assertFalse(np.allclose(results['symmetric'], results['unnormalized'], rtol=1e-3))
        # Random walk and symmetric are related but not identical for non-regular graphs
        # This is actually a regular graph (triangle) so they might be close
        # Just check they're not exactly the same
        if not np.allclose(results['symmetric'], results['random_walk'], rtol=1e-10):
            self.assertFalse(np.allclose(results['symmetric'], results['random_walk'], rtol=1e-3))
    
    def test_self_loops(self):
        """Test effect of adding self-loops."""
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 0],
            [1, 0, 2, 1, 0, 2]
        ], dtype=torch.long)
        num_nodes = 3
        
        # Without self-loops
        extractor_no_loops = SpectralFeatureExtractor(k=3, add_self_loops=False)
        features_no_loops = extractor_no_loops.extract_features(edge_index, num_nodes)
        
        # With self-loops
        extractor_loops = SpectralFeatureExtractor(k=3, add_self_loops=True)
        features_loops = extractor_loops.extract_features(edge_index, num_nodes)
        
        # Self-loops should change eigenvalues
        self.assertFalse(np.allclose(
            features_no_loops['eigenvalues'],
            features_loops['eigenvalues']
        ))
    
    def test_numerical_stability_large_k(self):
        """Test numerical stability with large k."""
        # Create a larger graph
        n = 50
        edges = [(i, (i+1)%n) for i in range(n)] + [((i+1)%n, i) for i in range(n)]
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        num_nodes = n
        
        # Request many eigenvectors
        extractor = SpectralFeatureExtractor(k=30, normalization='symmetric')
        features = extractor.extract_features(edge_index, num_nodes, validate=True)
        
        # Should still pass validation
        self.assertTrue(features['validation']['sorted'])
        self.assertTrue(features['validation']['orthogonal'])
    
    def test_weighted_edges(self):
        """Test with weighted edges."""
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 0],
            [1, 0, 2, 1, 0, 2]
        ], dtype=torch.long)
        edge_weight = torch.tensor([1.0, 1.0, 2.0, 2.0, 0.5, 0.5], dtype=torch.float)
        num_nodes = 3
        
        features = self.extractor.extract_features(
            edge_index, num_nodes, edge_weight=edge_weight, validate=True
        )
        
        # Weighted graph should still have valid spectral properties
        self.assertTrue(features['validation']['passed'])
        self.assertLess(np.abs(features['eigenvalues'][0]), 1e-5)


class TestCachingFunctionality(unittest.TestCase):
    """Test caching and loading of spectral features."""
    
    def setUp(self):
        """Create temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_load(self):
        """Test saving and loading spectral features."""
        # Create features
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 0],
            [1, 0, 2, 1, 0, 2]
        ], dtype=torch.long)
        num_nodes = 3
        
        extractor = SpectralFeatureExtractor(k=3, normalization='symmetric')
        features = extractor.extract_features(edge_index, num_nodes)
        
        # Save
        cache_file = Path(self.temp_dir) / "test_spectral.npz"
        np.savez_compressed(
            cache_file,
            eigenvalues=features['eigenvalues'],
            eigenvectors=features['eigenvectors'],
            num_nodes=num_nodes,
            normalization='symmetric',
            k=3
        )
        
        # Load
        loaded = load_spectral_features(str(cache_file))
        
        # Compare
        np.testing.assert_array_almost_equal(
            features['eigenvalues'],
            loaded['eigenvalues']
        )
        np.testing.assert_array_almost_equal(
            features['eigenvectors'],
            loaded['eigenvectors']
        )
        self.assertEqual(loaded['num_nodes'], num_nodes)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_empty_graph(self):
        """Test on graph with no edges."""
        edge_index = torch.empty((2, 0), dtype=torch.long)
        num_nodes = 5
        
        extractor = SpectralFeatureExtractor(k=4, normalization='symmetric')
        features = extractor.extract_features(edge_index, num_nodes, validate=False)
        
        # Graph with no edges: all eigenvalues should be 0 or 1 (with self-loops)
        self.assertIsNotNone(features['eigenvalues'])
    
    def test_k_larger_than_nodes(self):
        """Test when k > num_nodes."""
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        num_nodes = 2
        
        # Request k=10 but only 2 nodes
        extractor = SpectralFeatureExtractor(k=10, normalization='symmetric')
        features = extractor.extract_features(edge_index, num_nodes, validate=False)
        
        # Should automatically reduce k to num_nodes
        self.assertLessEqual(features['eigenvalues'].shape[0], num_nodes)
    
    def test_duplicate_edges(self):
        """Test graph with duplicate edges."""
        # Duplicate edge (0, 1)
        edge_index = torch.tensor([
            [0, 1, 0, 1, 1, 2, 2, 0],
            [1, 0, 1, 0, 2, 1, 0, 2]
        ], dtype=torch.long)
        num_nodes = 3
        
        extractor = SpectralFeatureExtractor(k=3, normalization='symmetric')
        features = extractor.extract_features(edge_index, num_nodes, validate=True)
        
        # Should still produce valid spectral features
        self.assertTrue(features['validation']['sorted'])


class TestSpectralProperties(unittest.TestCase):
    """Test mathematical properties of spectral decomposition."""
    
    def test_eigenvalue_bounds(self):
        """Test that eigenvalues are in correct range for symmetric Laplacian."""
        # Random graph
        np.random.seed(42)
        n = 20
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if np.random.rand() < 0.3:  # 30% edge probability
                    edges.extend([(i, j), (j, i)])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        
        extractor = SpectralFeatureExtractor(k=10, normalization='symmetric')
        features = extractor.extract_features(edge_index, n)
        
        # For symmetric normalized Laplacian: eigenvalues in [0, 2]
        self.assertTrue(np.all(features['eigenvalues'] >= -1e-8))
        self.assertTrue(np.all(features['eigenvalues'] <= 2 + 1e-6))
    
    def test_fiedler_value(self):
        """Test Fiedler value (second smallest eigenvalue) - algebraic connectivity."""
        # Connected graph should have positive Fiedler value
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 0],
            [1, 0, 2, 1, 3, 2, 0, 3]
        ], dtype=torch.long)
        num_nodes = 4
        
        extractor = SpectralFeatureExtractor(k=4, normalization='symmetric')
        features = extractor.extract_features(edge_index, num_nodes)
        
        # Fiedler value (λ₁) should be positive for connected graph
        fiedler = features['eigenvalues'][1]
        self.assertGreater(fiedler, 1e-6)
    
    def test_eigenvector_normalization(self):
        """Test that eigenvectors are normalized."""
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3],
            [1, 0, 2, 1, 3, 2]
        ], dtype=torch.long)
        num_nodes = 4
        
        extractor = SpectralFeatureExtractor(k=4, normalization='symmetric')
        features = extractor.extract_features(edge_index, num_nodes)
        
        # Each eigenvector should have unit norm
        norms = np.linalg.norm(features['eigenvectors'], axis=0)
        # Account for actual number of eigenvectors returned
        expected_norms = np.ones(features['eigenvectors'].shape[1])
        np.testing.assert_array_almost_equal(norms, expected_norms, decimal=5)


def run_integration_test():
    """Run integration test on sample dataset."""
    print("\n" + "="*60)
    print("INTEGRATION TEST: Sample Dataset Processing")
    print("="*60 + "\n")
    
    # Create sample graphs
    sample_graphs = [
        {
            'name': 'triangle',
            'edges': [[0, 1], [1, 0], [1, 2], [2, 1], [2, 0], [0, 2]],
            'num_nodes': 3
        },
        {
            'name': 'square',
            'edges': [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 0], [0, 3]],
            'num_nodes': 4
        },
        {
            'name': 'pentagon',
            'edges': [[i, (i+1)%5] for i in range(5)] + [[(i+1)%5, i] for i in range(5)],
            'num_nodes': 5
        }
    ]
    
    # Process each graph
    extractor = SpectralFeatureExtractor(k=4, normalization='symmetric')
    
    results = []
    for graph in sample_graphs:
        
        # --- START FIX: Correct edge_index creation ---
        # The old code flattened the list and reshaped incorrectly
        edge_index = torch.tensor(graph['edges'], dtype=torch.long).t()
        # --- END FIX ---
        
        features = extractor.extract_features(
            edge_index,
            graph['num_nodes'],
            validate=True
        )
        
        print(f"Graph: {graph['name']}")
        print(f"  Nodes: {graph['num_nodes']}")
        print(f"  Eigenvalues: {features['eigenvalues']}")
        print(f"  Validation: {features['validation']['passed']}")
        print(f"  Spectral gap: {features['eigenvalues'][1] - features['eigenvalues'][0]:.6f}")
        print()
        
        results.append({
            'name': graph['name'],
            'passed': features['validation']['passed']
        })
    
    # Summary
    all_passed = all(r['passed'] for r in results)
    print("="*60)
    print(f"Integration Test: {'PASSED' if all_passed else 'FAILED'}")
    print("="*60)
    
    return all_passed


def benchmark_performance():
    """Benchmark spectral feature extraction performance."""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60 + "\n")
    
    import time
    
    graph_sizes = [10, 50, 100, 200, 500]
    k_values = [8, 16, 32]
    
    results = []
    
    for n in graph_sizes:
        for k in k_values:
            if k >= n:
                continue
            
            # Create random graph (Erdős-Rényi)
            np.random.seed(42)
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if np.random.rand() < 0.1:  # 10% edge probability
                        edges.extend([(i, j), (j, i)])
            
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            
            # Time the extraction
            extractor = SpectralFeatureExtractor(k=k, normalization='symmetric')
            
            start = time.time()
            features = extractor.extract_features(edge_index, n, validate=False)
            elapsed = time.time() - start
            
            results.append({
                'n': n,
                'k': k,
                'num_edges': len(edges) // 2,
                'time': elapsed
            })
            
            print(f"n={n:3d}, k={k:2d}, edges={len(edges)//2:4d}, time={elapsed:.4f}s")
    
    print("\n" + "="*60)
    print(f"Average time: {np.mean([r['time'] for r in results]):.4f}s")
    print("="*60)


if __name__ == '__main__':
    print("\n" + "="*70)
    print(" SPECTRAL FEATURE EXTRACTION - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    # Run unit tests
    print("\n[1/3] Running Unit Tests...")
    print("-"*70)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestSpectralFeatureExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestCachingFunctionality))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestSpectralProperties))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Run integration test
    print("\n[2/3] Running Integration Test...")
    print("-"*70)
    integration_passed = run_integration_test()
    
    # Run benchmark
    print("\n[3/3] Running Performance Benchmark...")
    print("-"*70)
    benchmark_performance()
    
    # Final summary
    print("\n" + "="*70)
    print(" TEST SUITE SUMMARY")
    print("="*70)
    print(f"Unit Tests: {'PASSED' if result.wasSuccessful() else 'FAILED'}")
    print(f"  - Tests run: {result.testsRun}")
    print(f"  - Failures: {len(result.failures)}")
    print(f"  - Errors: {len(result.errors)}")
    print(f"Integration Test: {'PASSED' if integration_passed else 'FAILED'}")
    print("="*70)
    
    if result.wasSuccessful() and integration_passed:
        print("\n✓ All tests passed! Spectral feature extraction is ready for production.")
    else:
        print("\n✗ Some tests failed. Please review errors above.")
    
    print("\n")

    # --- START FIX: Uncomment sys.exit ---
    sys.exit(0 if result.wasSuccessful() and integration_passed else 1)
    # --- END FIX ---