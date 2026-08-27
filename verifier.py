from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

from PIL import Image, ImageStat
import torch
import torch.nn.functional as F
from torchvision import transforms

from model import FruitClassifier


@dataclass
class VerificationResult:
    label: str
    verified: bool
    confidence: float
    dataset_similarity: float
    class_margin: float
    verification_score: float
    probabilities: Dict[str, float]
    reasons: List[str]


class FruitVerifier:
    """Closed-set fruit classifier with a conservative unknown-image gateway."""

    # Conservative runtime floors. The checkpoint's calibrated thresholds are
    # still used, but the child-facing app prefers rejecting an uncertain image
    # over forcing it into one of the five known fruit classes.
    GATEWAY_CONFIDENCE_FLOOR = 0.70
    GATEWAY_PROBABILITY_MARGIN_FLOOR = 0.12
    GATEWAY_CLASS_MARGIN_FLOOR = 0.035
    GATEWAY_SIMILARITY_BUFFER = 0.02
    MIN_IMAGE_SIDE = 96
    MIN_IMAGE_CONTRAST = 8.0

    def __init__(self, checkpoint_path: Union[str, Path], device: str | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.checkpoint_path}")

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        self.class_names = list(checkpoint["class_names"])
        self.image_size = int(checkpoint.get("image_size", 224))
        self.model = FruitClassifier(len(self.class_names), pretrained=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device).eval()

        verification = checkpoint["verification"]
        self.centroids = F.normalize(
            torch.tensor(verification["centroids"], dtype=torch.float32, device=self.device),
            dim=1,
        )
        self.confidence_threshold = float(verification["confidence_threshold"])
        self.margin_threshold = float(verification["margin_threshold"])
        self.similarity_thresholds = {
            key: float(value) for key, value in verification["similarity_thresholds"].items()
        }
        self.metadata = checkpoint.get("metadata", {})

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _quality_reasons(self, image: Image.Image) -> List[str]:
        reasons: List[str] = []
        width, height = image.size
        if min(width, height) < self.MIN_IMAGE_SIDE:
            reasons.append("The image is too small for reliable recognition.")

        grayscale = image.convert("L")
        stats = ImageStat.Stat(grayscale)
        contrast = float(stats.stddev[0]) if stats.stddev else 0.0
        if contrast < self.MIN_IMAGE_CONTRAST:
            reasons.append("The image has too little visual detail or contrast.")

        return reasons

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> VerificationResult:
        image = image.convert("RGB")
        reasons = self._quality_reasons(image)

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits, embedding = self.model(tensor, return_embedding=True)
        probabilities_tensor = torch.softmax(logits, dim=1)[0]
        embedding = F.normalize(embedding, dim=1)
        similarities = torch.matmul(embedding, self.centroids.T)[0]

        top_probabilities = torch.topk(probabilities_tensor, k=min(2, len(self.class_names)))
        class_index = int(top_probabilities.indices[0].item())
        label = self.class_names[class_index]
        confidence = float(top_probabilities.values[0].item())
        second_probability = (
            float(top_probabilities.values[1].item()) if len(self.class_names) > 1 else 0.0
        )
        probability_margin = confidence - second_probability

        nearest_centroid_index = int(torch.argmax(similarities).item())
        nearest_centroid_label = self.class_names[nearest_centroid_index]

        similarity = float(similarities[class_index].item())
        other_similarities = torch.cat(
            [similarities[:class_index], similarities[class_index + 1 :]]
        )
        class_margin = (
            float(similarity - torch.max(other_similarities).item())
            if len(other_similarities)
            else 1.0
        )

        calibrated_similarity_threshold = self.similarity_thresholds[label]
        gateway_confidence_threshold = max(
            self.confidence_threshold,
            self.GATEWAY_CONFIDENCE_FLOOR,
        )
        gateway_similarity_threshold = min(
            0.98,
            calibrated_similarity_threshold + self.GATEWAY_SIMILARITY_BUFFER,
        )
        gateway_class_margin_threshold = max(
            self.margin_threshold,
            self.GATEWAY_CLASS_MARGIN_FLOOR,
        )

        if confidence < gateway_confidence_threshold:
            reasons.append("Model confidence is below the Unknown Gateway threshold.")
        if probability_margin < self.GATEWAY_PROBABILITY_MARGIN_FLOOR:
            reasons.append("The top fruit predictions are too close to each other.")
        if similarity < gateway_similarity_threshold:
            reasons.append("The image does not match this fruit's dataset profile closely enough.")
        if class_margin < gateway_class_margin_threshold:
            reasons.append("The image is too similar to multiple dataset fruit profiles.")
        if nearest_centroid_label != label:
            reasons.append(
                "The classifier and the nearest dataset fruit profile do not agree."
            )

        verified = not reasons

        confidence_gate = min(
            1.0, confidence / max(gateway_confidence_threshold, 1e-6)
        )
        probability_margin_gate = min(
            1.0,
            max(0.0, probability_margin)
            / max(self.GATEWAY_PROBABILITY_MARGIN_FLOOR, 1e-6),
        )
        similarity_gate = min(
            1.0,
            max(0.0, similarity) / max(gateway_similarity_threshold, 1e-6),
        )
        margin_gate = min(
            1.0,
            max(0.0, class_margin) / max(gateway_class_margin_threshold, 1e-6),
        )
        verification_score = float(
            100.0
            * min(
                confidence_gate,
                probability_margin_gate,
                similarity_gate,
                margin_gate,
            )
        )

        probabilities = {
            name: float(probabilities_tensor[index].item())
            for index, name in enumerate(self.class_names)
        }
        return VerificationResult(
            label=label if verified else "Unknown / Not Verified",
            verified=verified,
            confidence=confidence,
            dataset_similarity=similarity,
            class_margin=class_margin,
            verification_score=verification_score,
            probabilities=probabilities,
            reasons=reasons,
        )
