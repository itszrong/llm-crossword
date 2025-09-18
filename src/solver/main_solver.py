#!/usr/bin/env python3
"""
Main Crossword Solver Integration

This module provides a simple interface to the agentic crossword solver
for integration with the existing crossword framework.
"""

import logging
from typing import Optional
from src.solver.agents import CoordinatorAgent
from src.solver.specialized_solvers import create_specialized_solver
from src.solver.review_system import TwoStageReviewSystem
from src.crossword.crossword import CrosswordPuzzle

logger = logging.getLogger(__name__)


class AgenticCrosswordSolver:
    """
    Main interface for the agentic crossword solver
    
    Usage:
        solver = AgenticCrosswordSolver(difficulty="medium")
        success = solver.solve(puzzle)
    """
    
    def __init__(self, difficulty: str = "medium", enable_review: bool = None):
        self.difficulty = difficulty.lower()
        
        # DESIGN DECISION: DIFFICULTY-BASED AUTO-ENABLEMENT
        # Auto-enable review only for hard and cryptic difficulties
        # RATIONALE: These puzzles benefit most from review, cost-optimization for easier puzzles
        if enable_review is None:
            self.enable_review = self.difficulty in ["hard", "cryptic"]
        else:
            self.enable_review = enable_review
        
        # Create coordinator and configure it for the difficulty level
        base_coordinator = CoordinatorAgent()
        self.coordinator = create_specialized_solver(base_coordinator, self.difficulty)
        
        # Initialize review system if enabled
        if self.enable_review:
            self.review_system = TwoStageReviewSystem()
        else:
            self.review_system = None
        
        logger.info(f"Initialized {self.difficulty.title()} CrosswordSolver (review: {self.enable_review})")
    
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
            logger.info(f"Starting {self.difficulty} difficulty agentic solver for {puzzle.width}x{puzzle.height} puzzle")
            logger.info(f"Puzzle has {len(puzzle.clues)} clues to solve")
            logger.info(f"Solver features: backtracking={self.coordinator.backtrack_enabled}, thinking_depth={self.coordinator.thinking_depth}")
        
        try:
            # Use the coordinator agent to solve the puzzle
            success = self.coordinator.solve_puzzle(puzzle, puzzle_name)
            
            # Apply review system if enabled, puzzle not fully solved, and solver made no progress
            if (self.enable_review and self.review_system and not success and 
                self._should_trigger_review()):
                if verbose:
                    logger.info("🎭 No progress detected - attempting review and correction process...")
                
                # Get the solver log
                solver_log = self.coordinator.solver_log
                if solver_log:
                    corrections_applied, review_report = self.review_system.review_and_correct(
                        puzzle, solver_log, max_corrections=3
                    )
                    
                    # Store review report for potential saving
                    self._last_review_report = review_report
                    
                    if corrections_applied:
                        # Check if puzzle is now solved
                        final_solved = sum(1 for clue in puzzle.clues if clue.answered)
                        success = final_solved == len(puzzle.clues)
                        
                        if verbose:
                            if success:
                                logger.info("🎉 Puzzle completed after review corrections!")
                            else:
                                logger.info(f"⚡ Partial improvement: {final_solved}/{len(puzzle.clues)} clues solved")
            
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
    
    def _should_trigger_review(self) -> bool:
        """
        Determine if the review system should be triggered based on solver progress
        
        KEY DECISION: PROGRESS-BASED TRIGGERING
        - Only activate expensive review when solver is genuinely stuck
        - Balances intervention timing vs resource efficiency
        
        TRIGGER CONDITIONS:
        1. At least 3 iterations completed (avoid premature triggering)
        2. 2+ of last 3 iterations made no progress (clear stall pattern)
        
        TRADEOFF ANALYSIS:
        ✅ PRO: Targeted intervention, cost-efficient, reduces noise
        ❌ CON: Reactive rather than proactive, may miss early issues
        
        Returns:
            True if review should be triggered (no progress made in recent iterations)
        """
        if not hasattr(self.coordinator, 'solver_log') or not self.coordinator.solver_log:
            return False
        
        solver_log = self.coordinator.solver_log
        
        # Check if there have been at least 3 iterations
        if len(solver_log.iterations) < 3:
            return False
        
        # Check the last 2-3 iterations for progress
        recent_iterations = solver_log.iterations[-3:]
        
        # Count how many recent iterations made no progress
        no_progress_count = sum(1 for iteration in recent_iterations 
                               if isinstance(iteration, dict) and not iteration.get("progress_made", False))
        
        # Trigger review if 2+ of the last 3 iterations made no progress
        should_trigger = no_progress_count >= 2
        
        if should_trigger:
            logger.info(f"🔍 Review trigger: {no_progress_count}/3 recent iterations made no progress")
        
        return should_trigger
    
    def solve_with_stats(self, puzzle: CrosswordPuzzle, puzzle_name: str = "unknown", 
                        log_path: str = None, review_log_path: str = None) -> dict:
        """
        Solve puzzle and return detailed statistics
        
        Args:
            puzzle: The CrosswordPuzzle instance to solve
            puzzle_name: Name identifier for the puzzle
            log_path: Optional path to save detailed JSON log
            review_log_path: Optional path to save review report
            
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
        
        # Save review log if review was used and path provided
        if review_log_path and self.enable_review and hasattr(self, '_last_review_report'):
            self.review_system.save_review_report(self._last_review_report, review_log_path)
        
        return {
            "success": success,
            "total_clues": len(puzzle.clues),
            "initially_solved": initial_solved,
            "finally_solved": final_solved,
            "clues_solved": final_solved - initial_solved,
            "completion_rate": final_solved / len(puzzle.clues),
            "solving_time": end_time - start_time,
            "puzzle_size": f"{puzzle.width}x{puzzle.height}",
            "review_enabled": self.enable_review
        }


def quick_solve(puzzle: CrosswordPuzzle, puzzle_name: str = "unknown", difficulty: str = "medium") -> bool:
    """
    Convenience function for quick puzzle solving
    
    Args:
        puzzle: The CrosswordPuzzle instance to solve
        puzzle_name: Name identifier for the puzzle
        difficulty: Difficulty level (easy, medium, hard, cryptic)
        
    Returns:
        True if puzzle was successfully solved
    """
    solver = AgenticCrosswordSolver(difficulty=difficulty)
    return solver.solve(puzzle, puzzle_name=puzzle_name)
