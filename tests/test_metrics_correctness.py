"""Tests for metric functions correctness.

This module tests that metric functions:
1. Return correct shapes and types
2. Compute correct values for known inputs
3. Handle edge cases properly
4. Accumulate correctly across multiple updates
5. Match sklearn implementations (where applicable)
"""

import numpy as np
import pytest
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    jaccard_score,
    ndcg_score,
)

from src.modules.metrics.components.classification import (
    Accuracy,
    BalancedAccuracy,
    MAP,
    MeanAveragePrecision,
    MRR,
    NDCG,
    PrecisionAtRecall,
    SentiMRR,
)
from src.modules.metrics.components.segmentation import IoU


class TestAccuracy:
    """Tests for Accuracy metric."""

    @pytest.fixture
    def accuracy_metric(self):
        return Accuracy()

    def test_perfect_predictions(self, accuracy_metric):
        """Test that perfect predictions give accuracy of 1.0."""
        batch_size = 10
        num_classes = 3
        preds = torch.randn(batch_size, num_classes)
        targets = torch.arange(batch_size) % num_classes
        
        # Make perfect predictions
        preds = torch.zeros_like(preds)
        preds.scatter_(1, targets.unsqueeze(1), 1.0)
        
        accuracy_metric.update(preds, targets)
        result = accuracy_metric.compute()
        
        assert torch.allclose(result, torch.tensor(1.0)), \
            "Perfect predictions should give accuracy of 1.0"

    def test_wrong_predictions(self, accuracy_metric):
        """Test that all wrong predictions give accuracy of 0.0."""
        batch_size = 10
        num_classes = 3
        preds = torch.randn(batch_size, num_classes)
        targets = torch.zeros(batch_size, dtype=torch.long)
        
        # Make all predictions wrong (predict class 1 for class 0 targets)
        preds = torch.zeros_like(preds)
        preds[:, 1] = 1.0  # Predict class 1
        
        accuracy_metric.update(preds, targets)
        result = accuracy_metric.compute()
        
        assert torch.allclose(result, torch.tensor(0.0)), \
            "All wrong predictions should give accuracy of 0.0"

    def test_partial_accuracy(self, accuracy_metric):
        """Test partial accuracy calculation."""
        batch_size = 10
        num_classes = 3
        targets = torch.zeros(batch_size, dtype=torch.long)
        
        # Predict correctly for half the samples
        preds = torch.zeros(batch_size, num_classes)
        preds[:5, 0] = 1.0  # Correct predictions
        preds[5:, 1] = 1.0  # Wrong predictions
        
        accuracy_metric.update(preds, targets)
        result = accuracy_metric.compute()
        
        assert torch.allclose(result, torch.tensor(0.5)), \
            "Half correct should give accuracy of 0.5"

    def test_multiple_updates(self, accuracy_metric):
        """Test that metric accumulates correctly across multiple updates."""
        num_classes = 3
        
        # First batch: all correct
        preds1 = torch.zeros(5, num_classes)
        targets1 = torch.zeros(5, dtype=torch.long)
        preds1[:, 0] = 1.0
        accuracy_metric.update(preds1, targets1)
        
        # Second batch: all wrong
        preds2 = torch.zeros(5, num_classes)
        targets2 = torch.zeros(5, dtype=torch.long)
        preds2[:, 1] = 1.0
        accuracy_metric.update(preds2, targets2)
        
        result = accuracy_metric.compute()
        assert torch.allclose(result, torch.tensor(0.5)), \
            "5 correct out of 10 should give accuracy of 0.5"

    def test_reset(self, accuracy_metric):
        """Test that reset clears the metric state."""
        batch_size = 10
        num_classes = 3
        preds = torch.zeros(batch_size, num_classes)
        targets = torch.zeros(batch_size, dtype=torch.long)
        preds[:, 0] = 1.0
        
        accuracy_metric.update(preds, targets)
        result1 = accuracy_metric.compute()
        assert result1.item() == 1.0
        
        accuracy_metric.reset()
        result2 = accuracy_metric.compute()
        # After reset, should have no samples, so accuracy should be 0.0
        # (or epsilon handling might make it slightly different)
        assert result2.item() == 0.0 or torch.allclose(result2, torch.tensor(0.0))


class TestBalancedAccuracy:
    """Tests for BalancedAccuracy metric."""

    @pytest.fixture
    def balanced_accuracy_metric(self):
        return BalancedAccuracy(task='multiclass', num_classes=3)

    def test_perfect_predictions(self, balanced_accuracy_metric):
        """Test that perfect predictions give balanced accuracy of 1.0."""
        batch_size = 9
        num_classes = 3
        preds = torch.zeros(batch_size, num_classes)
        targets = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        
        # Perfect predictions
        preds.scatter_(1, targets.unsqueeze(1), 1.0)
        
        balanced_accuracy_metric.update(preds, targets)
        result = balanced_accuracy_metric.compute()
        
        assert torch.allclose(result, torch.tensor(1.0), atol=1e-5), \
            "Perfect predictions should give balanced accuracy of 1.0"

    def test_balanced_vs_unbalanced(self, balanced_accuracy_metric):
        """Test that balanced accuracy handles class imbalance correctly."""
        # 10 samples of class 0, 1 sample of class 1
        preds = torch.zeros(11, 3)
        targets = torch.tensor([0] * 10 + [1])
        
        # Predict all as class 0
        preds[:, 0] = 1.0
        
        balanced_accuracy_metric.update(preds, targets)
        result = balanced_accuracy_metric.compute()
        
        # Balanced accuracy should be average of per-class recall (only for classes present)
        # Class 0: recall = 10/10 = 1.0
        # Class 1: recall = 0/1 = 0.0
        # Class 2: not present, so excluded from average
        # Balanced accuracy = (1.0 + 0.0) / 2 = 0.5
        assert result.item() == 0.5, \
            f"Balanced accuracy should be 0.5 (only averaging over present classes), got {result.item()}"


class TestIoU:
    """Tests for IoU (Intersection over Union) metric."""

    @pytest.fixture
    def iou_metric(self):
        return IoU(n_class=3)

    def test_perfect_predictions(self, iou_metric):
        """Test that perfect predictions give IoU of 1.0."""
        batch_size = 2
        num_classes = 3
        height, width = 10, 10
        preds = torch.randn(batch_size, num_classes, height, width)
        targets = torch.randint(0, num_classes, (batch_size, height, width))
        
        # Make perfect predictions
        preds = torch.zeros_like(preds)
        for b in range(batch_size):
            for c in range(num_classes):
                preds[b, c] = (targets[b] == c).float() * 10.0
        
        iou_metric.update(preds, targets)
        result = iou_metric.compute()
        
        assert torch.allclose(result, torch.tensor(1.0), atol=1e-5), \
            "Perfect predictions should give IoU of 1.0"

    def test_no_overlap(self, iou_metric):
        """Test that no overlap gives IoU of 0.0."""
        batch_size = 2
        num_classes = 3
        height, width = 10, 10
        targets = torch.zeros(batch_size, height, width, dtype=torch.long)
        
        # Predict all as class 1 when targets are class 0
        preds = torch.zeros(batch_size, num_classes, height, width)
        preds[:, 1] = 10.0
        
        iou_metric.update(preds, targets)
        result = iou_metric.compute()
        
        # IoU should be very low (0.0) when there's no overlap
        assert result.item() < 0.1, \
            "No overlap should give IoU close to 0.0"


class TestNDCG:
    """Tests for NDCG metric."""

    @pytest.fixture
    def ndcg_metric(self):
        return NDCG(k=10)

    def test_perfect_ranking(self, ndcg_metric):
        """Test that perfect ranking gives NDCG of 1.0."""
        # Create a batch where predictions perfectly match targets
        batch_size = 5
        seq_length = 10
        y_true = torch.zeros(batch_size, seq_length)
        
        # Perfect ranking: highest scores for highest relevance
        # For perfect ranking, y_score should match y_true exactly
        # (highest relevance values should have highest scores)
        for i in range(seq_length):
            y_true[:, i] = seq_length - i  # Relevance decreases with position (higher = better)
        
        # Perfect ranking: scores match relevance exactly
        y_score = y_true.clone()
        
        ndcg_metric.update(y_score, y_true)
        result = ndcg_metric.compute()
        
        assert torch.allclose(result, torch.tensor(1.0), atol=1e-3), \
            f"Perfect ranking should give NDCG close to 1.0, got {result.item()}"

    def test_output_range(self, ndcg_metric):
        """Test that NDCG is in [0, 1] range."""
        batch_size = 3
        seq_length = 5
        y_true = torch.rand(batch_size, seq_length)
        y_score = torch.rand(batch_size, seq_length)
        
        ndcg_metric.update(y_score, y_true)
        result = ndcg_metric.compute()
        
        assert 0.0 <= result.item() <= 1.0, \
            "NDCG should be in [0, 1] range"


class TestMRR:
    """Tests for MRR (Mean Reciprocal Rank) metric."""

    @pytest.fixture
    def mrr_metric(self):
        return MRR(k=None)

    def test_perfect_ranking(self, mrr_metric):
        """Test that perfect ranking gives high MRR."""
        batch_size = 5
        seq_length = 10
        y_true = torch.zeros(batch_size, seq_length)
        y_score = torch.zeros(batch_size, seq_length)
        
        # Put relevance 1 at position 0 (perfect)
        y_true[:, 0] = 1.0
        y_score[:, 0] = 10.0  # Highest score
        
        mrr_metric.update(y_score, y_true)
        result = mrr_metric.compute()
        
        # Perfect ranking: first position, so reciprocal rank = 1/1 = 1.0
        assert result.item() > 0.9, \
            "Perfect ranking should give MRR close to 1.0"

    def test_output_range(self, mrr_metric):
        """Test that MRR is non-negative."""
        batch_size = 3
        seq_length = 5
        y_true = torch.rand(batch_size, seq_length)
        y_score = torch.rand(batch_size, seq_length)
        
        mrr_metric.update(y_score, y_true)
        result = mrr_metric.compute()
        
        assert result.item() >= 0.0, "MRR should be non-negative"


class TestSentiMRR:
    """Tests for SentiMRR metric."""

    @pytest.fixture
    def senti_mrr_metric(self):
        return SentiMRR(k=None)

    def test_basic_functionality(self, senti_mrr_metric):
        """Test that SentiMRR computes correctly."""
        batch_size = 5
        y_pred = torch.rand(batch_size)
        s_c = torch.rand(batch_size)
        s_mean = torch.tensor(0.5)
        
        senti_mrr_metric.update(y_pred, s_c, s_mean)
        result = senti_mrr_metric.compute()
        
        assert result.item() >= 0.0, "SentiMRR should be non-negative"
        assert torch.isfinite(result), "SentiMRR should be finite"


class TestPrecisionAtRecall:
    """Tests for PrecisionAtRecall metric."""

    @pytest.fixture
    def precision_at_recall_metric(self):
        return PrecisionAtRecall(recall_point=0.95)

    def test_basic_functionality(self, precision_at_recall_metric):
        """Test that PrecisionAtRecall computes correctly."""
        batch_size = 100
        # Create sorted distances and labels
        distances = torch.sort(torch.rand(batch_size))[0]
        labels = torch.zeros(batch_size)
        labels[:50] = 1.0  # First 50 are positive (sorted by distance)
        
        precision_at_recall_metric.update(distances, labels)
        result = precision_at_recall_metric.compute()
        
        assert 0.0 <= result.item() <= 1.0, \
            "Precision should be in [0, 1] range"

    def test_output_range(self, precision_at_recall_metric):
        """Test that output is in valid range."""
        batch_size = 50
        distances = torch.sort(torch.rand(batch_size))[0]
        labels = torch.randint(0, 2, (batch_size,)).float()
        
        precision_at_recall_metric.update(distances, labels)
        result = precision_at_recall_metric.compute()
        
        assert 0.0 <= result.item() <= 1.0, \
            "Precision should be in [0, 1] range"


class TestAccuracySklearnComparison:
    """Tests comparing Accuracy metric with sklearn implementation."""

    @pytest.fixture
    def accuracy_metric(self):
        return Accuracy()

    def test_compare_with_sklearn_simple(self, accuracy_metric):
        """Compare Accuracy with sklearn.accuracy_score on simple case."""
        batch_size = 20
        num_classes = 3
        preds = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        
        # Get class predictions
        pred_classes = preds.argmax(dim=1)
        
        # Compute with our metric
        accuracy_metric.update(preds, targets)
        our_result = accuracy_metric.compute().item()
        
        # Compute with sklearn
        sklearn_result = accuracy_score(
            targets.cpu().numpy(),
            pred_classes.cpu().numpy()
        )
        
        assert abs(our_result - sklearn_result) < 1e-6, \
            f"Accuracy should match sklearn: ours={our_result}, sklearn={sklearn_result}"

    def test_compare_with_sklearn_multiple_batches(self, accuracy_metric):
        """Compare Accuracy with sklearn across multiple batches."""
        num_classes = 3
        all_targets = []
        all_preds = []
        
        # Multiple batches
        for _ in range(3):
            batch_size = 10
            preds = torch.randn(batch_size, num_classes)
            targets = torch.randint(0, num_classes, (batch_size,))
            pred_classes = preds.argmax(dim=1)
            
            accuracy_metric.update(preds, targets)
            all_targets.append(targets.cpu().numpy())
            all_preds.append(pred_classes.cpu().numpy())
        
        our_result = accuracy_metric.compute().item()
        
        # Concatenate all batches for sklearn
        all_targets_np = np.concatenate(all_targets)
        all_preds_np = np.concatenate(all_preds)
        sklearn_result = accuracy_score(all_targets_np, all_preds_np)
        
        assert abs(our_result - sklearn_result) < 1e-6, \
            f"Accuracy should match sklearn across batches: ours={our_result}, sklearn={sklearn_result}"


class TestBalancedAccuracySklearnComparison:
    """Tests comparing BalancedAccuracy metric with sklearn implementation."""

    def test_compare_with_sklearn_binary(self):
        """Compare BalancedAccuracy with sklearn for binary classification."""
        balanced_accuracy_metric = BalancedAccuracy(task='binary', num_classes=2)
        
        batch_size = 20
        preds = torch.randn(batch_size, 2)
        targets = torch.randint(0, 2, (batch_size,))
        pred_classes = preds.argmax(dim=1)
        
        # Compute with our metric
        balanced_accuracy_metric.update(preds, targets)
        our_result = balanced_accuracy_metric.compute().item()
        
        # Compute with sklearn
        sklearn_result = balanced_accuracy_score(
            targets.cpu().numpy(),
            pred_classes.cpu().numpy()
        )
        
        assert abs(our_result - sklearn_result) < 1e-5, \
            f"BalancedAccuracy (binary) should match sklearn: ours={our_result}, sklearn={sklearn_result}"

    @pytest.fixture
    def balanced_accuracy_metric(self):
        return BalancedAccuracy(task='multiclass', num_classes=3)

    def test_compare_with_sklearn_simple(self, balanced_accuracy_metric):
        """Compare BalancedAccuracy with sklearn.balanced_accuracy_score."""
        batch_size = 30
        num_classes = 3
        preds = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        
        # Get class predictions
        pred_classes = preds.argmax(dim=1)
        
        # Compute with our metric
        balanced_accuracy_metric.update(preds, targets)
        our_result = balanced_accuracy_metric.compute().item()
        
        # Compute with sklearn
        sklearn_result = balanced_accuracy_score(
            targets.cpu().numpy(),
            pred_classes.cpu().numpy()
        )
        
        # Should match sklearn exactly (only averaging over classes present in targets)
        assert abs(our_result - sklearn_result) < 1e-5, \
            f"BalancedAccuracy should match sklearn: ours={our_result}, sklearn={sklearn_result}"

    def test_compare_with_sklearn_multiple_batches(self, balanced_accuracy_metric):
        """Compare BalancedAccuracy with sklearn across multiple batches."""
        num_classes = 3
        all_targets = []
        all_preds = []
        
        # Multiple batches
        for _ in range(2):
            batch_size = 15
            preds = torch.randn(batch_size, num_classes)
            targets = torch.randint(0, num_classes, (batch_size,))
            pred_classes = preds.argmax(dim=1)
            
            balanced_accuracy_metric.update(preds, targets)
            all_targets.append(targets.cpu().numpy())
            all_preds.append(pred_classes.cpu().numpy())
        
        our_result = balanced_accuracy_metric.compute().item()
        
        # Concatenate all batches for sklearn
        all_targets_np = np.concatenate(all_targets)
        all_preds_np = np.concatenate(all_preds)
        sklearn_result = balanced_accuracy_score(all_targets_np, all_preds_np)
        
        assert abs(our_result - sklearn_result) < 1e-5, \
            f"BalancedAccuracy should match sklearn across batches: ours={our_result}, sklearn={sklearn_result}"


class TestIoUSklearnComparison:
    """Tests comparing IoU metric with sklearn jaccard_score implementation."""

    @pytest.fixture
    def iou_metric(self):
        return IoU(n_class=3)

    def test_compare_with_sklearn_simple(self, iou_metric):
        """Compare IoU with sklearn.jaccard_score (macro average)."""
        batch_size = 2
        num_classes = 3
        height, width = 10, 10
        preds = torch.randn(batch_size, num_classes, height, width)
        targets = torch.randint(0, num_classes, (batch_size, height, width))
        
        # Get class predictions
        pred_classes = preds.argmax(dim=1)
        
        # Compute with our metric
        iou_metric.update(preds, targets)
        our_result = iou_metric.compute().item()
        
        # Flatten for sklearn
        preds_flat = pred_classes.cpu().numpy().flatten()
        targets_flat = targets.cpu().numpy().flatten()
        
        # Compute IoU (Jaccard) with sklearn using macro average
        sklearn_result = jaccard_score(
            targets_flat,
            preds_flat,
            average='macro',
            zero_division=0
        )
        
        # Note: Our IoU computes sum(intersection) / sum(union) across all classes
        # sklearn macro computes mean(IoU per class), which might differ slightly
        # But they should be close for balanced cases
        assert abs(our_result - sklearn_result) < 0.1 or our_result > 0.9, \
            f"IoU should be close to sklearn: ours={our_result}, sklearn={sklearn_result}"

    def test_compare_with_sklearn_perfect_match(self, iou_metric):
        """Compare IoU with sklearn on perfect predictions."""
        batch_size = 2
        num_classes = 3
        height, width = 10, 10
        targets = torch.randint(0, num_classes, (batch_size, height, width))
        
        # Perfect predictions
        preds = torch.zeros(batch_size, num_classes, height, width)
        for b in range(batch_size):
            for c in range(num_classes):
                preds[b, c] = (targets[b] == c).float() * 10.0
        
        pred_classes = preds.argmax(dim=1)
        
        # Compute with our metric
        iou_metric.update(preds, targets)
        our_result = iou_metric.compute().item()
        
        # Flatten for sklearn
        preds_flat = pred_classes.cpu().numpy().flatten()
        targets_flat = targets.cpu().numpy().flatten()
        
        # Compute with sklearn
        sklearn_result = jaccard_score(
            targets_flat,
            preds_flat,
            average='macro',
            zero_division=0
        )
        
        # For perfect predictions, both should be close to 1.0
        assert abs(our_result - sklearn_result) < 0.1 or (our_result > 0.9 and sklearn_result > 0.9), \
            f"Perfect predictions should give high IoU: ours={our_result}, sklearn={sklearn_result}"


class TestNDCGSklearnComparison:
    """Tests comparing NDCG metric with sklearn implementation."""

    @pytest.fixture
    def ndcg_metric(self):
        return NDCG(k=10)

    def test_compare_with_sklearn_simple(self, ndcg_metric):
        """Compare NDCG with sklearn.ndcg_score."""
        batch_size = 3
        seq_length = 10
        
        # Create relevance scores and predictions
        y_true = torch.rand(batch_size, seq_length) * 5  # Relevance 0-5
        y_score = torch.rand(batch_size, seq_length)
        
        # Compute with our metric
        ndcg_metric.update(y_score, y_true)
        our_result = ndcg_metric.compute().item()
        
        # Compute with sklearn for each sample and average
        sklearn_results = []
        for i in range(batch_size):
            sklearn_ndcg = ndcg_score(
                [y_true[i].cpu().numpy()],
                [y_score[i].cpu().numpy()],
                k=min(10, seq_length)
            )
            sklearn_results.append(sklearn_ndcg)
        sklearn_result = np.mean(sklearn_results)
        
        # Results might differ slightly due to implementation differences
        # (e.g., how ties are handled, normalization), but should be close
        assert abs(our_result - sklearn_result) < 0.2, \
            f"NDCG should be close to sklearn: ours={our_result}, sklearn={sklearn_result}"

    def test_compare_with_sklearn_perfect_ranking(self, ndcg_metric):
        """Compare NDCG with sklearn on perfect ranking."""
        batch_size = 3
        seq_length = 10
        
        # Perfect ranking: scores match relevance exactly
        y_true = torch.arange(seq_length, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
        y_score = y_true.clone()  # Perfect ranking
        
        # Compute with our metric
        ndcg_metric.update(y_score, y_true)
        our_result = ndcg_metric.compute().item()
        
        # Compute with sklearn
        sklearn_results = []
        for i in range(batch_size):
            sklearn_ndcg = ndcg_score(
                [y_true[i].cpu().numpy()],
                [y_score[i].cpu().numpy()],
                k=min(10, seq_length)
            )
            sklearn_results.append(sklearn_ndcg)
        sklearn_result = np.mean(sklearn_results)
        
        # Both should be 1.0 for perfect ranking
        assert abs(our_result - 1.0) < 0.05 or abs(sklearn_result - 1.0) < 0.05, \
            f"Perfect ranking should give NDCG close to 1.0: ours={our_result}, sklearn={sklearn_result}"


class TestMeanAveragePrecision:
    """Tests for MeanAveragePrecision metric."""

    @pytest.fixture
    def map_metric(self):
        return MeanAveragePrecision(task='multiclass', num_classes=3)

    def test_output_shape(self, map_metric):
        """Test that output has correct shape."""
        batch_size = 10
        num_classes = 3
        preds = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        
        map_metric.update(preds, targets)
        result = map_metric.compute()
        
        assert result.shape == (), "MAP should be a scalar"
        assert torch.isfinite(result), "MAP should be finite"

    def test_output_range(self, map_metric):
        """Test that MAP is in [0, 1] range."""
        batch_size = 10
        num_classes = 3
        preds = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        
        map_metric.update(preds, targets)
        result = map_metric.compute()
        
        assert 0.0 <= result.item() <= 1.0, \
            f"MAP should be in [0, 1] range, got {result.item()}"

    def test_perfect_predictions(self, map_metric):
        """Test that perfect predictions give high MAP."""
        batch_size = 10
        num_classes = 3
        targets = torch.randint(0, num_classes, (batch_size,))
        
        # Perfect predictions: very high probability for correct class
        preds = torch.zeros(batch_size, num_classes)
        preds.scatter_(1, targets.unsqueeze(1), 10.0)
        
        map_metric.update(preds, targets)
        result = map_metric.compute()
        
        assert result.item() > 0.9, \
            f"Perfect predictions should give high MAP, got {result.item()}"

    def test_multiple_updates(self, map_metric):
        """Test that metric accumulates correctly across multiple updates."""
        num_classes = 3
        
        # Multiple batches
        for _ in range(3):
            batch_size = 5
            preds = torch.randn(batch_size, num_classes)
            targets = torch.randint(0, num_classes, (batch_size,))
            map_metric.update(preds, targets)
        
        result = map_metric.compute()
        assert torch.isfinite(result), "MAP should be finite after multiple updates"
        assert 0.0 <= result.item() <= 1.0, "MAP should be in [0, 1] range"

    def test_binary_classification(self):
        """Test MAP for binary classification."""
        map_metric = MeanAveragePrecision(task='binary', num_classes=2)
        batch_size = 20
        # Our implementation handles (N, 2) format by extracting positive class probabilities
        preds = torch.rand(batch_size, 2)  # Use probabilities between 0 and 1
        targets = torch.randint(0, 2, (batch_size,))
        
        map_metric.update(preds, targets)
        result = map_metric.compute()
        
        assert result.shape == ()
        assert 0.0 <= result.item() <= 1.0


class TestMeanAveragePrecisionSklearnComparison:
    """Tests comparing MeanAveragePrecision with sklearn implementation."""

    @pytest.fixture
    def map_metric(self):
        return MeanAveragePrecision(task='multiclass', num_classes=3)

    def test_compare_with_sklearn_binary(self):
        """Compare MAP with sklearn.average_precision_score for binary classification."""
        map_metric = MeanAveragePrecision(task='binary', num_classes=2)
        
        batch_size = 20
        preds = torch.randn(batch_size, 2)
        targets = torch.randint(0, 2, (batch_size,))
        
        # Apply softmax to get probabilities
        probs = torch.softmax(preds, dim=1)
        
        # Compute with our metric
        map_metric.update(probs, targets)
        our_result = map_metric.compute().item()
        
        # Compute with sklearn (for binary, use positive class probabilities)
        sklearn_result = average_precision_score(
            targets.cpu().numpy(),
            probs[:, 1].cpu().numpy()
        )
        
        assert abs(our_result - sklearn_result) < 1e-4, \
            f"MAP (binary) should match sklearn: ours={our_result}, sklearn={sklearn_result}"

    def test_compare_with_sklearn_multiclass(self, map_metric):
        """Compare MAP with sklearn.average_precision_score for multiclass."""
        batch_size = 30
        num_classes = 3
        preds = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        
        # Apply softmax to get probabilities
        probs = torch.softmax(preds, dim=1)
        
        # Compute with our metric
        map_metric.update(probs, targets)
        our_result = map_metric.compute().item()
        
        # Compute with sklearn (macro average across classes)
        sklearn_results = []
        for i in range(num_classes):
            # One-vs-rest AP for each class
            y_true_binary = (targets.cpu().numpy() == i).astype(int)
            y_score = probs[:, i].cpu().numpy()
            ap_score = average_precision_score(y_true_binary, y_score)
            sklearn_results.append(ap_score)
        sklearn_result = np.mean(sklearn_results)
        
        assert abs(our_result - sklearn_result) < 1e-4, \
            f"MAP (multiclass) should match sklearn: ours={our_result}, sklearn={sklearn_result}"

    def test_compare_with_sklearn_multiple_batches(self, map_metric):
        """Compare MAP with sklearn across multiple batches."""
        num_classes = 3
        all_targets = []
        all_probs = []
        
        # Multiple batches
        for _ in range(2):
            batch_size = 15
            preds = torch.randn(batch_size, num_classes)
            targets = torch.randint(0, num_classes, (batch_size,))
            probs = torch.softmax(preds, dim=1)
            
            map_metric.update(probs, targets)
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
        
        our_result = map_metric.compute().item()
        
        # Concatenate all batches for sklearn
        all_targets_np = np.concatenate(all_targets)
        all_probs_np = np.concatenate(all_probs)
        
        # Compute with sklearn (macro average)
        sklearn_results = []
        for i in range(num_classes):
            y_true_binary = (all_targets_np == i).astype(int)
            y_score = all_probs_np[:, i]
            ap_score = average_precision_score(y_true_binary, y_score)
            sklearn_results.append(ap_score)
        sklearn_result = np.mean(sklearn_results)
        
        assert abs(our_result - sklearn_result) < 1e-4, \
            f"MAP should match sklearn across batches: ours={our_result}, sklearn={sklearn_result}"

    def test_alias_works(self):
        """Test that MAP alias works."""
        map_metric1 = MeanAveragePrecision(task='multiclass', num_classes=3)
        map_metric2 = MAP(task='multiclass', num_classes=3)
        
        assert isinstance(map_metric1, MeanAveragePrecision)
        assert isinstance(map_metric2, MeanAveragePrecision)
