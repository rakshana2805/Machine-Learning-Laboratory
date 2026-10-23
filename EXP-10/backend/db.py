# backend/db.py

import json
import sqlite3
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "eyecare_history.db"


def get_connection():
    """Create a SQLite database connection."""
    connection = sqlite3.connect(str(DB_PATH))
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    """Create the prediction history table if it does not exist."""
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                prediction TEXT,
                confidence REAL,
                image_path TEXT,
                probabilities TEXT,
                preprocessing_stages TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.commit()


def save_prediction(
    prediction=None,
    confidence=None,
    filename=None,
    image_path=None,
    probabilities=None,
    preprocessing_stages=None,
    **kwargs,
):
    """Save a prediction and its optional metadata."""

    # Support common alternative keyword names.
    prediction = prediction or kwargs.get("predicted_class") or kwargs.get("result")
    confidence = (
        confidence
        if confidence is not None
        else kwargs.get("confidence_score", 0.0)
    )
    filename = filename or kwargs.get("image_name") or kwargs.get("file_name")
    image_path = image_path or kwargs.get("uploaded_image_path")

    if prediction is None:
        raise ValueError("A prediction value is required.")

    initialize_database()

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO prediction_history (
                filename,
                prediction,
                confidence,
                image_path,
                probabilities,
                preprocessing_stages,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(filename) if filename is not None else None,
                str(prediction),
                float(confidence or 0.0),
                str(image_path) if image_path is not None else None,
                json.dumps(probabilities, default=str)
                if probabilities is not None else None,
                json.dumps(preprocessing_stages, default=str)
                if preprocessing_stages is not None else None,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        connection.commit()
        return cursor.lastrowid


def get_prediction_history(limit=100):
    """Return the latest predictions as dictionaries."""
    initialize_database()

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM prediction_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (max(1, min(int(limit), 1000)),),
        ).fetchall()

    history = []
    for row in rows:
        item = dict(row)

        for field in ("probabilities", "preprocessing_stages"):
            if item.get(field):
                try:
                    item[field] = json.loads(item[field])
                except (TypeError, json.JSONDecodeError):
                    pass

        history.append(item)

    return history


# Initialize the database when this module is imported.
initialize_database()