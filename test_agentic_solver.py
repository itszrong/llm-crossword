#!/usr/bin/env python3
"""
Test script for the Agentic Crossword Solver

This script tests the solver on puzzles of increasing difficulty
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.crossword.utils import load_puzzle
from src.solver.main_solver import AgenticCrosswordSolver

# Load environment variables
load_dotenv()

def test_puzzle(puzzle_file: str, puzzle_name: str):
    """Test solver on a specific puzzle"""
    print(f"\n{'='*50}")
    print(f"🧩 Testing {puzzle_name} Puzzle")
    print(f"{'='*50}")
    
    try:
        # Load the puzzle
        puzzle = load_puzzle(puzzle_file)
        print(f"📋 Loaded puzzle: {puzzle.width}x{puzzle.height} grid, {len(puzzle.clues)} clues")
        
        # Show clues
        print("\n📝 Clues:")
        for i, clue in enumerate(puzzle.clues):
            print(f"  {clue.number}. {clue.text} ({clue.length} letters, {clue.direction})")
        
        print(f"\n🎯 Initial grid:")
        print(puzzle)
        
        # Create solver and solve with statistics
        solver = AgenticCrosswordSolver()
        stats = solver.solve_with_stats(puzzle)
        
        # Display results
        print(f"\n📊 Results:")
        print(f"  ✅ Success: {stats['success']}")
        print(f"  📈 Completion: {stats['finally_solved']}/{stats['total_clues']} clues ({stats['completion_rate']:.1%})")
        print(f"  ⏱️  Time: {stats['solving_time']:.2f} seconds")
        
        print(f"\n🎯 Final grid:")
        print(puzzle)
        
        # Validate solution
        if puzzle.validate_all():
            print("✅ Solution is CORRECT!")
        else:
            print("❌ Solution has errors")
            
        return stats['success']
        
    except Exception as e:
        print(f"❌ Error testing {puzzle_name}: {e}")
        return False


def main():
    """Test solver on all puzzles"""
    print("🚀 Agentic Crossword Solver Test Suite")
    print("Using Multi-Agent Design Patterns")
    
    # Test puzzles in order of difficulty
    test_cases = [
        ("data/easy.json", "Easy"),
        ("data/medium.json", "Medium"),
        ("data/hard.json", "Hard"),
        ("data/cryptic.json", "Cryptic")
    ]
    
    results = {}
    
    for puzzle_file, puzzle_name in test_cases:
        if os.path.exists(puzzle_file):
            success = test_puzzle(puzzle_file, puzzle_name)
            results[puzzle_name] = success
        else:
            print(f"⚠️  Puzzle file {puzzle_file} not found")
            results[puzzle_name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("📋 SUMMARY")
    print(f"{'='*50}")
    
    for puzzle_name, success in results.items():
        status = "✅ SOLVED" if success else "❌ FAILED"
        print(f"  {puzzle_name}: {status}")
    
    solved_count = sum(results.values())
    total_count = len(results)
    print(f"\n🎯 Overall: {solved_count}/{total_count} puzzles solved")
    
    if solved_count == total_count:
        print("🎉 ALL PUZZLES SOLVED! Excellent work!")
    elif solved_count >= total_count // 2:
        print("👍 Good progress! More than half solved.")
    else:
        print("💪 Keep working! Room for improvement.")


if __name__ == "__main__":
    main()
