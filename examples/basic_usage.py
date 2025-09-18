#!/usr/bin/env python3
"""
Basic Usage Example - Agentic Crossword Solver

This example demonstrates the simplest way to use the agentic crossword solver
to solve puzzles of different difficulties.
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path to import solver
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.crossword.utils import load_puzzle
from src.solver.main_solver import AgenticCrosswordSolver

# Load environment variables
load_dotenv()

def solve_puzzle_example(difficulty: str):
    """
    Example of solving a puzzle with the agentic solver
    
    Args:
        difficulty: Puzzle difficulty ('easy', 'medium', 'hard', 'cryptic')
    """
    print(f"\n🧩 Solving {difficulty.title()} Puzzle")
    print("=" * 50)
    
    try:
        # Load the puzzle
        puzzle = load_puzzle(f"../data/{difficulty}.json")
        print(f"📋 Loaded {puzzle.width}x{puzzle.height} puzzle with {len(puzzle.clues)} clues")
        
        # Create solver (review system auto-enabled for hard/cryptic)
        solver = AgenticCrosswordSolver(difficulty=difficulty)
        
        print(f"🤖 Created {difficulty} solver (review enabled: {solver.enable_review})")
        
        # Solve the puzzle
        print("🔍 Solving puzzle...")
        success = solver.solve(puzzle, verbose=True, puzzle_name=f"{difficulty}_example")
        
        # Show results
        solved_clues = sum(1 for clue in puzzle.clues if clue.answered)
        completion_rate = solved_clues / len(puzzle.clues)
        
        print(f"\n📊 Results:")
        print(f"  ✅ Success: {success}")
        print(f"  📈 Completion: {solved_clues}/{len(puzzle.clues)} clues ({completion_rate:.1%})")
        
        # Show the final grid
        print(f"\n🎯 Final grid:")
        print(puzzle)
        
        # Validate solution
        if puzzle.validate_all():
            print("✅ Solution is completely correct!")
        else:
            print("⚠️  Solution has some gaps or errors")
            
        return success
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def solve_with_detailed_stats(difficulty: str):
    """
    Example of getting detailed statistics from the solver
    """
    print(f"\n📊 Detailed Analysis - {difficulty.title()} Puzzle")
    print("=" * 50)
    
    try:
        # Load puzzle
        puzzle = load_puzzle(f"../data/{difficulty}.json")
        
        # Create solver
        solver = AgenticCrosswordSolver(difficulty=difficulty)
        
        # Solve with detailed stats
        stats = solver.solve_with_stats(
            puzzle=puzzle,
            puzzle_name=f"{difficulty}_detailed",
            log_path=f"../logs/{difficulty}_basic_example.json"
        )
        
        # Display comprehensive results
        print(f"📈 Detailed Statistics:")
        print(f"  🎯 Success: {stats['success']}")
        print(f"  📊 Completion Rate: {stats['completion_rate']:.1%}")
        print(f"  ⏱️  Solving Time: {stats['solving_time']:.2f} seconds")
        print(f"  🧩 Puzzle Size: {stats['puzzle_size']}")
        print(f"  🎭 Review System: {stats['review_enabled']}")
        print(f"  📝 Clues Solved: {stats['finally_solved']}/{stats['total_clues']}")
        
        return stats
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """Main example function"""
    print("🚀 Agentic Crossword Solver - Basic Usage Examples")
    print("=" * 60)
    
    # Example 1: Simple solving
    print("\n📝 Example 1: Simple Solving")
    solve_puzzle_example("easy")
    
    # Example 2: Medium difficulty
    print("\n📝 Example 2: Medium Difficulty")
    solve_puzzle_example("medium")
    
    # Example 3: Detailed statistics
    print("\n📝 Example 3: Detailed Statistics")
    stats = solve_with_detailed_stats("medium")
    
    # Example 4: Hard puzzle with review system
    print("\n📝 Example 4: Hard Puzzle with Review System")
    solve_puzzle_example("hard")
    
    print(f"\n🎉 Examples completed!")
    print(f"💡 Check the logs/ directory for detailed solving logs")

if __name__ == "__main__":
    main()
