"""Tests for AI screening classifier architecture and feature shapes."""

import torch
from pipeline.screen.ai_classifier import (
    CLASS_NAMES,
    EMBED_DIM,
    NUM_CLASSES,
    NUM_FRAMES,
    ScreeningClassifier,
)


class TestScreeningClassifier:
    """Verify classifier architecture, forward pass, and tensor shapes."""

    def test_forward_pass_shape(self):
        """Classifier should output (B, NUM_CLASSES) logits."""
        model = ScreeningClassifier()
        model.eval()

        B = 4
        embeddings = torch.randn(B, NUM_FRAMES, EMBED_DIM)
        scanner_scores = torch.randn(B, NUM_FRAMES)
        otb_scores = torch.randn(B, NUM_FRAMES)

        with torch.no_grad():
            logits = model(embeddings, scanner_scores, otb_scores)

        assert logits.shape == (B, NUM_CLASSES)

    def test_single_sample_forward(self):
        """Single sample batch should work."""
        model = ScreeningClassifier()
        model.eval()

        embeddings = torch.randn(1, NUM_FRAMES, EMBED_DIM)
        scanner_scores = torch.randn(1, NUM_FRAMES)
        otb_scores = torch.randn(1, NUM_FRAMES)

        with torch.no_grad():
            logits = model(embeddings, scanner_scores, otb_scores)

        assert logits.shape == (1, NUM_CLASSES)
        # Verify softmax produces valid probabilities
        probs = torch.softmax(logits, dim=-1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-5)

    def test_class_names_match_num_classes(self):
        """CLASS_NAMES length should match NUM_CLASSES."""
        assert len(CLASS_NAMES) == NUM_CLASSES

    def test_model_is_deterministic_in_eval_mode(self):
        """Same input should produce same output in eval mode (dropout off)."""
        model = ScreeningClassifier()
        model.eval()

        embeddings = torch.randn(1, NUM_FRAMES, EMBED_DIM)
        scanner_scores = torch.randn(1, NUM_FRAMES)
        otb_scores = torch.randn(1, NUM_FRAMES)

        with torch.no_grad():
            out1 = model(embeddings, scanner_scores, otb_scores)
            out2 = model(embeddings, scanner_scores, otb_scores)

        assert torch.allclose(out1, out2)

    def test_gradient_flows_in_training_mode(self):
        """Verify gradients flow through the classifier during training."""
        model = ScreeningClassifier()
        model.train()

        embeddings = torch.randn(2, NUM_FRAMES, EMBED_DIM, requires_grad=True)
        scanner_scores = torch.randn(2, NUM_FRAMES, requires_grad=True)
        otb_scores = torch.randn(2, NUM_FRAMES, requires_grad=True)

        logits = model(embeddings, scanner_scores, otb_scores)
        loss = logits.sum()
        loss.backward()

        assert embeddings.grad is not None
        assert scanner_scores.grad is not None
