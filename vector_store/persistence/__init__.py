"""Persistence strategies package."""
from .json_persistence import JsonPersistence, PicklePersistence

__all__ = [
    "JsonPersistence",
    "PicklePersistence",
]
