"""Tests for eval_metrics functions correctness.

This module tests that eval_metrics functions:
1. Return correct values for known inputs
2. Handle edge cases properly
3. Return correct data types
"""

import pytest

from src.modules.metrics.eval_metrics import accuracy, auprc, auroc


class TestAccuracyEval:
    """Tests for accuracy function from eval_metrics."""

    def test_perfect_predictions(self):
        """Test that perfect predictions give accuracy of 1.0."""
        targets = [0, 1, 2, 0, 1, 2]
        preds = [0, 1, 2, 0, 1, 2]
        
        result = accuracy(targets, preds, verbose=False)
        
        assert result == 1.0, "Perfect predictions should give accuracy of 1.0"

    def test_all_wrong_predictions(self):
        """Test that all wrong predictions give accuracy of 0.0."""
        targets = [0, 0, 0, 0]
        preds = [1, 1, 1, 1]
        
        result = accuracy(targets, preds, verbose=False)
        
        assert result == 0.0, "All wrong predictions should give accuracy of 0.0"

    def test_partial_accuracy(self):
        """Test partial accuracy calculation."""
        targets = [0, 0, 1, 1]
        preds = [0, 1, 1, 1]
        
        result = accuracy(targets, preds, verbose=False)
        
        assert result == 0.75, "3 out of 4 correct should give accuracy of 0.75"

    def test_return_type(self):
        """Test that function returns a float."""
        targets = [0, 1, 2]
        preds = [0, 1, 2]
        
        result = accuracy(targets, preds, verbose=False)
        
        assert isinstance(result, float), "Result should be a float"

    def test_different_lengths_error(self):
        """Test that function handles different lengths (should use sklearn's handling)."""
        targets = [0, 1, 2]
        preds = [0, 1, 2, 0]  # Different length
        
        # sklearn's accuracy_score should handle this, but let's check
        # It might raise an error or use the minimum length
        try:
            result = accuracy(targets, preds, verbose=False)
            assert isinstance(result, float)
        except ValueError:
            # If sklearn raises an error for different lengths, that's acceptable
            pass


class TestAUROC:
    """Tests for auroc function from eval_metrics."""

    def test_perfect_separation(self):
        """Test that perfect separation gives AUROC of 1.0."""
        targets = [0, 0, 0, 1, 1, 1]
        probs = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]  # Perfect separation
        
        result = auroc(targets, probs, plot=False, verbose=False)
        
        assert result == 1.0, "Perfect separation should give AUROC of 1.0"

    def test_random_classifier(self):
        """Test that random classifier gives AUROC around 0.5."""
        targets = [0, 0, 0, 0, 1, 1, 1, 1]
        probs = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # Random
        
        result = auroc(targets, probs, plot=False, verbose=False)
        
        # Random classifier should give AUROC close to 0.5
        assert 0.4 <= result <= 0.6, \
            f"Random classifier should give AUROC around 0.5, got {result}"

    def test_return_type(self):
        """Test that function returns a float."""
        targets = [0, 0, 1, 1]
        probs = [0.1, 0.2, 0.7, 0.8]
        
        result = auroc(targets, probs, plot=False, verbose=False)
        
        assert isinstance(result, float), "Result should be a float"

    def test_output_range(self):
        """Test that AUROC is in [0, 1] range."""
        targets = [0, 0, 1, 1]
        probs = [0.3, 0.4, 0.5, 0.6]
        
        result = auroc(targets, probs, plot=False, verbose=False)
        
        assert 0.0 <= result <= 1.0, f"AUROC should be in [0, 1], got {result}"

    def test_reversed_separation(self):
        """Test that reversed separation gives low AUROC."""
        targets = [0, 0, 0, 1, 1, 1]
        probs = [0.7, 0.8, 0.9, 0.1, 0.2, 0.3]  # Reversed
        
        result = auroc(targets, probs, plot=False, verbose=False)
        
        assert result < 0.5, "Reversed separation should give AUROC < 0.5"

    def test_single_class_warning(self):
        """Test handling of single class (should be handled by sklearn)."""
        # Single class might cause issues, sklearn should handle it
        targets = [0, 0, 0, 0]
        probs = [0.1, 0.2, 0.3, 0.4]
        
        try:
            result = auroc(targets, probs, plot=False, verbose=False)
            # If it works, result should be a float
            assert isinstance(result, float)
        except ValueError:
            # sklearn might raise ValueError for single class, which is acceptable
            pass


class TestAUPRC:
    """Tests for auprc function from eval_metrics."""

    def test_perfect_separation(self):
        """Test that perfect separation gives high AUPRC."""
        targets = [0, 0, 0, 1, 1, 1]
        probs = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]  # Perfect separation
        
        result = auprc(targets, probs, plot=False, verbose=False)
        
        assert result > 0.9, "Perfect separation should give high AUPRC"

    def test_return_type(self):
        """Test that function returns a float."""
        targets = [0, 0, 1, 1]
        probs = [0.1, 0.2, 0.7, 0.8]
        
        result = auprc(targets, probs, plot=False, verbose=False)
        
        assert isinstance(result, float), "Result should be a float"

    def test_output_range(self):
        """Test that AUPRC is in [0, 1] range."""
        targets = [0, 0, 1, 1]
        probs = [0.3, 0.4, 0.5, 0.6]
        
        result = auprc(targets, probs, plot=False, verbose=False)
        
        assert 0.0 <= result <= 1.0, f"AUPRC should be in [0, 1], got {result}"

    def test_better_separation_higher_auprc(self):
        """Test that better separation gives higher AUPRC."""
        targets = [0, 0, 0, 1, 1, 1]
        
        # Good separation (almost perfect)
        probs_good = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        result_good = auprc(targets, probs_good, plot=False, verbose=False)
        
        # Poor separation (some overlap)
        probs_poor = [0.5, 0.52, 0.48, 0.55, 0.51, 0.49]  # All close to 0.5
        result_poor = auprc(targets, probs_poor, plot=False, verbose=False)
        
        assert result_good > result_poor, \
            f"Better separation should give higher AUPRC (good: {result_good}, poor: {result_poor})"

    def test_single_class_warning(self):
        """Test handling of single class (should be handled by sklearn)."""
        targets = [0, 0, 0, 0]
        probs = [0.1, 0.2, 0.3, 0.4]
        
        try:
            result = auprc(targets, probs, plot=False, verbose=False)
            assert isinstance(result, float)
        except ValueError:
            # sklearn might raise ValueError for single class, which is acceptable
            pass
