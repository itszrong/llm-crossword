#!/usr/bin/env python3
"""
Main Crossword Solver Integration

This module provides a simple interface to the agentic crossword solver
for integration with the existing crossword framework.
"""

import logging
from typing import Optional
from src.solver.agents import CoordinatorAgent
from src.crossword.crossword import CrosswordPuzzle

logger = logging.getLogger(__name__)


class AgenticCrosswordSolver:
    """
    Main interface for the agentic crossword solver
    
    Usage:
        solver = AgenticCrosswordSolver()
        success = solver.solve(puzzle)
    """
    
    def __init__(self):
        self.coordinator = CoordinatorAgent()
    
    def solve(self, puzzle: CrosswordPuzzle, verbose: bool = True, puzzle_name: str = "unknown") -> bool:
        """
        Solve a crossword puzzle using agentic design patterns
        
        Args:
            puzzle: The CrosswordPuzzle instance to solve
            verbose: Whether to log detailed progress
            puzzle_name: Name identifier for the puzzle
            
        Returns:
            True if puzzle was successfully solved, False otherwise
        """
        if verbose:
            logger.info(f"Starting agentic solver for {puzzle.width}x{puzzle.height} puzzle")
            logger.info(f"Puzzle has {len(puzzle.clues)} clues to solve")
        
        try:
            # Use the coordinator agent to solve the puzzle
            success = self.coordinator.solve_puzzle(puzzle, puzzle_name)
            
            if verbose:
                if success:
                    logger.info("✅ Puzzle solved successfully!")
                else:
                    logger.warning("❌ Puzzle solving incomplete")
                    
                # Log final state
                solved_clues = sum(1 for clue in puzzle.clues if clue.answered)
                logger.info(f"Final state: {solved_clues}/{len(puzzle.clues)} clues solved")
            
            return success
            
        except Exception as e:
            logger.error(f"Error during puzzle solving: {e}")
            return False
    
    def solve_with_stats(self, puzzle: CrosswordPuzzle, puzzle_name: str = "unknown", log_path: str = None) -> dict:
        """
        Solve puzzle and return detailed statistics
        
        Args:
            puzzle: The CrosswordPuzzle instance to solve
            puzzle_name: Name identifier for the puzzle
            log_path: Optional path to save detailed JSON log
            
        Returns:
            Dictionary with solving statistics
        """
        import time
        
        start_time = time.time()
        initial_solved = sum(1 for clue in puzzle.clues if clue.answered)
        
        success = self.solve(puzzle, verbose=False, puzzle_name=puzzle_name)
        
        end_time = time.time()
        final_solved = sum(1 for clue in puzzle.clues if clue.answered)
        
        # Save detailed log if path provided
        if log_path:
            self.coordinator.save_log(log_path)
        
        return {
            "success": success,
            "total_clues": len(puzzle.clues),
            "initially_solved": initial_solved,
            "finally_solved": final_solved,
            "clues_solved": final_solved - initial_solved,
            "completion_rate": final_solved / len(puzzle.clues),
            "solving_time": end_time - start_time,
            "puzzle_size": f"{puzzle.width}x{puzzle.height}"
        }


def quick_solve(puzzle: CrosswordPuzzle, puzzle_name: str = "unknown") -> bool:
    """
    Convenience function for quick puzzle solving
    
    Args:
        puzzle: The CrosswordPuzzle instance to solve
        puzzle_name: Name identifier for the puzzle
        
    Returns:
        True if puzzle was successfully solved
    """
    solver = AgenticCrosswordSolver()
    return solver.solve(puzzle, puzzle_name=puzzle_name)
