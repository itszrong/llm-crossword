"""
Agentic Crossword Solver

A sophisticated multi-agent system for solving crossword puzzles using
intelligent reasoning, constraint satisfaction, and self-correction.
"""

__version__ = "1.0.0"
__author__ = "Agentic Design Patterns Team"

from .solver.main_solver import AgenticCrosswordSolver
from .crossword.utils import load_puzzle

__all__ = ["AgenticCrosswordSolver", "load_puzzle"]
