#!/usr/bin/env python3
"""
Types and enums for the crossword solver
"""

from enum import Enum


class DifficultyLevel(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    CRYPTIC = "cryptic"
