from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from verifier import FruitVerifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify and verify one fruit image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("artifacts/fruit_classifier.pt"))
    args = parser.parse_args()

    verifier = FruitVerifier(args.model)
    with Image.open(args.image) as image:
        result = verifier.predict(image)

    print(f"Result: {result.label}")
    print(f"Verified: {result.verified}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Dataset similarity: {result.dataset_similarity:.2%}")
    print(f"Verification score: {result.verification_score:.1f}/100")
    print("Top probabilities:")
    for name, probability in sorted(result.probabilities.items(), key=lambda item: item[1], reverse=True):
        print(f"  {name:12s} {probability:.2%}")
    if result.reasons:
        print("Rejected because:")
        for reason in result.reasons:
            print(f"  - {reason}")


if __name__ == "__main__":
    main()
