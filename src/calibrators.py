import numpy as np
from sklearn.linear_model import LogisticRegression


class SigmoidCalibrator:

    def __init__(self):
        self.model = LogisticRegression(
            solver="lbfgs"
        )

    def fit(
        self,
        probabilities,
        y,
    ):
        probabilities = np.asarray(
            probabilities
        )

        probabilities = np.clip(
            probabilities,
            1e-6,
            1 - 1e-6,
        )

        logits = np.log(
            probabilities
            / (
                1
                - probabilities
            )
        ).reshape(
            -1,
            1,
        )

        self.model.fit(
            logits,
            y,
        )

        return self

    def predict(
        self,
        probabilities,
    ):
        probabilities = np.asarray(
            probabilities
        )

        probabilities = np.clip(
            probabilities,
            1e-6,
            1 - 1e-6,
        )

        logits = np.log(
            probabilities
            / (
                1
                - probabilities
            )
        ).reshape(
            -1,
            1,
        )

        return (
            self.model
            .predict_proba(
                logits
            )[:, 1]
        )