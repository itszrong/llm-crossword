#!/usr/bin/env python3
"""
Test script to demonstrate the two-stage review system
"""

import os
import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.crossword.crossword import CrosswordPuzzle
from src.crossword.utils import load_puzzle
from src.solver.main_solver import AgenticCrosswordSolver

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_review_system_on_cryptic():
    """Test the review system on the cryptic puzzle that failed in the terminal output"""
    
    # Load the cryptic puzzle
    cryptic_file = Path("../data/cryptic.json")
    if not cryptic_file.exists():
        logger.error(f"Cryptic puzzle file not found: {cryptic_file}")
        return
    
    puzzle = load_puzzle("../data/cryptic.json")
    
    logger.info("🧩 Testing Review System on Cryptic Puzzle")
    logger.info(f"Puzzle: {puzzle.width}x{puzzle.height} with {len(puzzle.clues)} clues")
    
    # Test with review system enabled (auto-enabled for cryptic)
    solver_with_review = AgenticCrosswordSolver(difficulty="cryptic")
    
    stats = solver_with_review.solve_with_stats(
        puzzle=puzzle,
        puzzle_name="cryptic_test_with_review",
        log_path="../logs/review_test_solver_log.json",
        review_log_path="../logs/review_test_review_report.json"
    )
    
    logger.info("📊 Results with Review System:")
    logger.info(f"  Success: {stats['success']}")
    logger.info(f"  Completion: {stats['finally_solved']}/{stats['total_clues']} clues ({stats['completion_rate']:.1%})")
    logger.info(f"  Time: {stats['solving_time']:.2f} seconds")
    logger.info(f"  Review enabled: {stats['review_enabled']}")
    
    # Display puzzle state
    print("\n🎯 Final puzzle grid:")
    print(puzzle)
    
    # Show some clue states
    print("\n🔍 Sample clue states:")
    for clue in puzzle.clues[:10]:  # Show first 10 clues
        status = "✅" if clue.answered else "❌"
        current_chars = puzzle.get_current_clue_chars(clue)
        current_word = "".join(char or "_" for char in current_chars)
        print(f"  {status} Clue {clue.number} {clue.direction.name}: '{clue.text}' = {current_word}")


def test_review_system_comparison():
    """Compare solving with and without review system"""
    
    # Load a hard puzzle for comparison
    hard_file = Path("../data/hard.json") 
    if not hard_file.exists():
        logger.warning("Hard puzzle file not found, using cryptic puzzle for comparison")
        hard_file = Path("../data/cryptic.json")
    
    # Test without review system (explicitly disabled)
    puzzle1 = load_puzzle(str(hard_file))
    solver_no_review = AgenticCrosswordSolver(difficulty="hard", enable_review=False)
    
    stats_no_review = solver_no_review.solve_with_stats(
        puzzle=puzzle1,
        puzzle_name="comparison_no_review"
    )
    
    # Test with review system (auto-enabled for hard)
    puzzle2 = load_puzzle(str(hard_file))
    solver_with_review = AgenticCrosswordSolver(difficulty="hard")
    
    stats_with_review = solver_with_review.solve_with_stats(
        puzzle=puzzle2, 
        puzzle_name="comparison_with_review",
        review_log_path="../logs/comparison_review_report.json"
    )
    
    logger.info("\n📊 Comparison Results:")
    logger.info("Without Review System:")
    logger.info(f"  Completion: {stats_no_review['finally_solved']}/{stats_no_review['total_clues']} ({stats_no_review['completion_rate']:.1%})")
    logger.info(f"  Time: {stats_no_review['solving_time']:.2f} seconds")
    
    logger.info("With Review System:")
    logger.info(f"  Completion: {stats_with_review['finally_solved']}/{stats_with_review['total_clues']} ({stats_with_review['completion_rate']:.1%})")
    logger.info(f"  Time: {stats_with_review['solving_time']:.2f} seconds")
    
    improvement = stats_with_review['finally_solved'] - stats_no_review['finally_solved']
    if improvement > 0:
        logger.info(f"🎉 Review system solved {improvement} additional clues!")
    elif improvement == 0:
        logger.info("⚖️ No difference in clues solved")
    else:
        logger.info(f"⚠️ Review system solved {abs(improvement)} fewer clues")


if __name__ == "__main__":
    logger.info("🚀 Starting Review System Test")
    
    try:
        # Test the review system on a failed puzzle
        test_review_system_on_cryptic()
        
        print("\n" + "="*80 + "\n")
        
        # Compare performance with and without review
        test_review_system_comparison()
        
        logger.info("✅ Review system testing completed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
