import numpy as np

from app import train_and_predict, get_accuracy


def test_predictions_not_none():
    """
    Test 1: Sprawdza, czy otrzymujemy jakąkolwiek predykcję.
    """
    preds, _ = train_and_predict()
    assert preds is not None, "Predictions should not be None."


def test_predictions_length():
    """
    Test 2 (na maksymalną ocenę 5): Sprawdza, czy długość listy predykcji jest większa od 0 
    i czy odpowiada przewidywanej liczbie próbek testowych.
    """
    preds, y_test = train_and_predict()
    assert len(preds) > 0, "Predictions list should not be empty."
    assert len(preds) == len(y_test), f"Predictions length ({len(preds)}) should match test labels length ({len(y_test)})."


def test_predictions_value_range():
    """
    Test 3 (na maksymalną ocenę 5): Sprawdza, czy wartości w predykcjach mieszczą się 
    w spodziewanym zakresie: Dla zbioru Iris mamy 3 klasy (0, 1, 2).
    """ 
    preds, _ = train_and_predict()
    unique_preds = np.unique(preds)
    for p in unique_preds:
        assert p in [0, 1, 2], f"Unexpected prediction value: {p}"


def test_model_accuracy():
    """
    Test 4: Sprawdza, czy dokładność modelu wynosi co najmniej 70%.
    """
    accuracy = get_accuracy()
    assert accuracy >= 0.7, f"Model accuracy should be at least 70%, but got {accuracy * 100:.2f}%"
