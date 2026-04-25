"""
Ensemble predictor that combines the geometric measurement pipeline and the
trained ResNet-18 classifier into one decision.

Design:
  - Measurement is precise but only fires on perpendicular shots with a
    findable hex+aperture combo.
  - Classifier fires on any image but doesn't give measurements.
  - Run both. If they agree, very high confidence. If they disagree, the
    classifier "wins" for naming the class (it has visual evidence of every
    angle, the measurement may have latched onto the wrong size bucket),
    but we surface the disagreement so the caller can prompt for recapture.
  - If only one fires, return its prediction with reduced confidence.

Public surface:

    predictor = EnsemblePredictor.load(model_dir)   # or EnsemblePredictor() for measurement-only
    result = predictor.predict(image_rgb)
    # result.class_name, result.confidence, result.agreement, result.measurement, result.classifier
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rfconnectorai.classifier.predict import ClassifierPrediction, ConnectorClassifier
from rfconnectorai.measurement.class_predictor import Prediction, predict_class


@dataclass
class EnsembleResult:
    class_name: str
    confidence: float                       # 0..1, blended across the two signals
    agreement: str                          # "agree" | "disagree" | "measurement_only" | "classifier_only" | "neither"
    measurement: Prediction | None = None
    classifier: ClassifierPrediction | None = None
    reason: str = ""


class EnsemblePredictor:
    """
    Holds an optional ConnectorClassifier alongside the measurement pipeline.
    Stateless after construction. If no classifier is loaded, the predictor
    runs the measurement pipeline alone — useful for environments without
    PyTorch or when no model has been trained yet.
    """

    def __init__(self, classifier: ConnectorClassifier | None = None):
        self.classifier = classifier

    @classmethod
    def load(
        cls,
        classifier_dir: Path | None,
    ) -> "EnsemblePredictor":
        """Load with a trained classifier from `classifier_dir`. Pass None for
        measurement-only mode."""
        if classifier_dir is None:
            return cls(classifier=None)
        return cls(classifier=ConnectorClassifier.load(classifier_dir))

    def predict(
        self,
        image: np.ndarray,
        require_aruco: bool = False,
        aruco_marker_size_mm: float | None = 25.0,
    ) -> EnsembleResult:
        """Run both predictors on `image` and combine."""
        meas: Prediction = predict_class(
            image,
            aruco_marker_size_mm=aruco_marker_size_mm,
            require_aruco=require_aruco,
        )
        clf: ClassifierPrediction | None = None
        if self.classifier is not None:
            clf = self.classifier.predict(image)

        meas_fired = meas.class_name != "Unknown"
        clf_fired = clf is not None

        if meas_fired and clf_fired:
            if meas.class_name == clf.class_name:
                # Both agree — confidence is the blended average, leaning on
                # the classifier's softmax probability since it's a true
                # probability while the measurement is a hard decision.
                blended = 0.5 + 0.5 * clf.confidence
                return EnsembleResult(
                    class_name=meas.class_name,
                    confidence=blended,
                    agreement="agree",
                    measurement=meas,
                    classifier=clf,
                )
            else:
                # Disagree — surface it. Pick the classifier's class as the
                # headline (more robust across viewing angles) but with
                # low confidence so the caller knows to prompt.
                return EnsembleResult(
                    class_name=clf.class_name,
                    confidence=min(clf.confidence, 0.5),
                    agreement="disagree",
                    measurement=meas,
                    classifier=clf,
                    reason=(
                        f"measurement says {meas.class_name}, "
                        f"classifier says {clf.class_name} ({clf.confidence:.0%})"
                    ),
                )

        if meas_fired:
            # Only measurement fired (no classifier loaded, or classifier failed).
            base_conf = 0.7 if self.classifier is None else 0.5
            return EnsembleResult(
                class_name=meas.class_name,
                confidence=base_conf,
                agreement="measurement_only",
                measurement=meas,
                classifier=clf,
            )

        if clf_fired:
            # Only classifier fired — measurement bailed out (probably
            # non-perpendicular view or unfindable hex).
            return EnsembleResult(
                class_name=clf.class_name,
                confidence=clf.confidence * 0.85,   # discount: no measurement to confirm
                agreement="classifier_only",
                measurement=meas,
                classifier=clf,
                reason=meas.reason,
            )

        # Neither fired.
        return EnsembleResult(
            class_name="Unknown",
            confidence=0.0,
            agreement="neither",
            measurement=meas,
            classifier=clf,
            reason=meas.reason or "no signals available",
        )
