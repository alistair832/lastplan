from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

from PIL import Image
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
    def __init__(self, checkpoint_path: Union[str, Path], device: str | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.checkpoint_path}")

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
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

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> VerificationResult:
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits, embedding = self.model(tensor, return_embedding=True)
        probabilities_tensor = torch.softmax(logits, dim=1)[0]
        embedding = F.normalize(embedding, dim=1)
        similarities = torch.matmul(embedding, self.centroids.T)[0]

        class_index = int(torch.argmax(probabilities_tensor).item())
        label = self.class_names[class_index]
        confidence = float(probabilities_tensor[class_index].item())
        similarity = float(similarities[class_index].item())

        other_similarities = torch.cat(
            [similarities[:class_index], similarities[class_index + 1 :]]
        )
        class_margin = float(similarity - torch.max(other_similarities).item())
        similarity_threshold = self.similarity_thresholds[label]

        reasons: List[str] = []
        if confidence < self.confidence_threshold:
            reasons.append("Model confidence is below the calibrated threshold.")
        if similarity < similarity_threshold:
            reasons.append("Image features are not similar enough to this fruit's dataset profile.")
        if class_margin < self.margin_threshold:
            reasons.append("The image is too similar to multiple fruit classes.")

        verified = not reasons
        confidence_gate = min(1.0, confidence / max(self.confidence_threshold, 1e-6))
        similarity_gate = min(1.0, max(0.0, similarity) / max(similarity_threshold, 1e-6))
        margin_gate = min(1.0, max(0.0, class_margin) / max(self.margin_threshold, 1e-6))
        verification_score = float(100.0 * min(confidence_gate, similarity_gate, margin_gate))

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
