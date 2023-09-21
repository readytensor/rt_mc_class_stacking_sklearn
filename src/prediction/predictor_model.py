from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
import joblib
import os
import pandas as pd
import numpy as np


PREDICTOR_FILE_NAME = "classifier_model.pkl"


class Classifier:
    """A wrapper class for the Stacking Classifier.

    This class provides a consistent interface that can be used with other
    classifier models.

    Attributes:
        model_name (str): Name of the classifier model.
    """

    model_name = "Stacking Classifier"

    def __init__(self, passthrough=False):
        """Construct a new Stacking Classifier.

        Args:
            passthrough (bool): Whether to pass original training data to final
                                estimator.
        """
        self.passthrough = passthrough
        self.model = self.build_model()
        self._is_trained = False

    def build_model(self) -> StackingClassifier:
        """Build a new stacking classifier.

        Returns:
            StackingClassifier: Initialized stacking classifier.
        """
        base_learners = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('svc', SVC(kernel='rbf', C=1, degree=2, probability=True)),
            ('logistic', LogisticRegression()),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(100,),
                activation="relu",
                solver="adam",
                learning_rate="adaptive",
                max_iter=500
            )),
            ('knn', KNeighborsClassifier(n_neighbors=5))
        ]
        model = StackingClassifier(
            estimators=base_learners,
            final_estimator=LogisticRegression(),
            passthrough=self.passthrough,
        )
        return model

    def fit(self, train_inputs: pd.DataFrame, train_targets: pd.Series) -> None:
        """Fit the classifier to the training data.

        Args:
            train_inputs (pd.DataFrame): Training input data.
            train_targets (pd.Series): Training target data.
        """
        # Column-vector
        if isinstance(train_targets, pd.DataFrame) and train_targets.shape[1] == 1:
            y = train_targets.values.ravel()
        else:
            y = train_targets
        self.model.fit(train_inputs.values, y)
        self._is_trained = True

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict classification labels for the given data.

        Args:
            inputs (pd.DataFrame): Input data for prediction.

        Returns:
            np.ndarray: Predicted classification labels.
        """
        return self.model.predict(inputs.values)

    def predict_proba(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted class probabilities.
        """
        return self.model.predict_proba(inputs.values)

    def evaluate(self, test_inputs: pd.DataFrame, test_targets: pd.Series) -> float:
        """Evaluate the classifier and return the accuracy score.

        Args:
            test_inputs (pd.DataFrame): Test input data.
            test_targets (pd.Series): Test target data.

        Returns:
            float: Accuracy score of the classifier.

        Raises:
            NotFittedError: If the model is not trained yet.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        predictions = self.predict(test_inputs)
        return accuracy_score(test_targets, predictions)

    def save(self, model_dir_path: str) -> None:
        """Save the classifier to disk.

        Args:
            model_dir_path (str): Directory path to save the model.

        Raises:
            NotFittedError: If the model is not trained yet.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Classifier":
        """Load the classifier from disk.

        Args:
            model_dir_path (str): Directory path from where to load the model.

        Returns:
            Classifier: Loaded classifier model.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    def __str__(self):
        """String representation of the Classifier.

        Returns:
            str: Information about the classifier.
        """
        return (
            f"Model name: {self.model_name} ("
            f"passthrough: {self.passthrough})"
        )


def train_predictor_model(
    train_inputs: pd.DataFrame, train_targets: pd.Series, hyperparameters: dict
) -> Classifier:
    """
    Instantiate and train the predictor model.

    Args:
        train_X (pd.DataFrame): The training data inputs.
        train_y (pd.Series): The training data labels.
        hyperparameters (dict): Hyperparameters for the classifier.

    Returns:
        'Classifier': The classifier model
    """
    classifier = Classifier(**hyperparameters)
    classifier.fit(train_inputs=train_inputs, train_targets=train_targets)
    return classifier


def predict_with_model(
    classifier: Classifier, data: pd.DataFrame, return_probs=False
) -> np.ndarray:
    """
    Predict class probabilities for the given data.

    Args:
        classifier (Classifier): The classifier model.
        data (pd.DataFrame): The input data.
        return_probs (bool): Whether to return class probabilities or labels.
            Defaults to True.

    Returns:
        np.ndarray: The predicted classes or class probabilities.
    """
    if return_probs:
        return classifier.predict_proba(data)
    return classifier.predict(data)


def save_predictor_model(model: Classifier, predictor_dir_path: str) -> None:
    """
    Save the classifier model to disk.

    Args:
        model (Classifier): The classifier model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Classifier:
    """
    Load the classifier model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Classifier: A new instance of the loaded classifier model.
    """
    return Classifier.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Classifier, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the classifier model and return the accuracy.

    Args:
        model (Classifier): The classifier model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.

    Returns:
        float: The accuracy of the classifier model.
    """
    return model.evaluate(x_test, y_test)
