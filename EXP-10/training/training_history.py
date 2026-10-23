
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
HISTORY_PATH = BASE_DIR / "models" / "training_history.json"


def save_training_history(history):
    """
    Save genuine history recorded during model training.
    Expected keys may include:
    accuracy, val_accuracy, loss, val_loss, epochs.
    """

    HISTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(HISTORY_PATH, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)


def load_training_history():
    """Return saved training history, or None if unavailable."""

    if not HISTORY_PATH.exists():
        return None

    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as file:
            history = json.load(file)

        if not isinstance(history, dict):
            return None

        return history

    except (OSError, json.JSONDecodeError):
        return None
