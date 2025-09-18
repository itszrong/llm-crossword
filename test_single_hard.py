#!/usr/bin/env python3
"""
Test script for a single hard puzzle with detailed visualization logging
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

def test_hard_puzzle():
    """Test the hard puzzle with detailed logging"""
    print("🧩 Testing Hard Puzzle with VisualizationAgent")
    print("=" * 60)
    
    try:
        # Load the puzzle
        puzzle = load_puzzle("data/hard.json")
        print(f"📋 Loaded puzzle: {puzzle.width}x{puzzle.height} grid, {len(puzzle.clues)} clues")
        
        # Show clues
        print("\n📝 Clues:")
        for clue in puzzle.clues:
            print(f"  {clue.number}. {clue.text} ({clue.length} letters, {clue.direction})")
        
        print(f"\n🎯 Initial grid:")
        print(puzzle)
        
        # Create specialized solver for hard difficulty
        solver = AgenticCrosswordSolver(difficulty="hard")
        
        print(f"\n🤖 Using HardCrosswordCoordinator with:")
        print(f"  ⚙️ Max iterations: {solver.coordinator.max_iterations}")
        print(f"  ↔️ Backtracking: {solver.coordinator.backtrack_enabled}")
        print(f"  🧠 Thinking depth: {solver.coordinator.thinking_depth}")
        print(f"  ⚡ Async solving: {getattr(solver.coordinator, 'use_async_solving', False)}")
        
        # Create log path
        log_path = "logs/hard_detailed_test.json"
        os.makedirs("logs", exist_ok=True)
        
        print("\n🔍 Starting solve with detailed visualization logging...")
        stats = solver.solve_with_stats(puzzle, puzzle_name="Hard_Test", log_path=log_path)
        
        # Display results
        print(f"\n📊 Results:")
        print(f"  ✅ Success: {stats['success']}")
        print(f"  📈 Completion: {stats['finally_solved']}/{stats['total_clues']} clues ({stats['completion_rate']:.1%})")
        print(f"  ⏱️  Time: {stats['solving_time']:.2f} seconds")
        print(f"  📝 Log saved to: {log_path}")
        
        print(f"\n🎯 Final grid:")
        print(puzzle)
        
        # Validate solution
        if puzzle.validate_all():
            print("✅ Solution is CORRECT!")
        else:
            print("❌ Solution has errors")
            
        print(f"\n📋 Visualization History:")
        print(f"  Total visualizations captured: {len(solver.coordinator.visualization_agent.visualization_history)}")
        
        # Show key clue states
        print(f"\n🔍 Final Clue States:")
        for clue in puzzle.clues:
            current_chars = puzzle.get_current_clue_chars(clue)
            current_word = "".join(char or "_" for char in current_chars)
            solved = "✅" if clue.answered and all(char is not None for char in current_chars) else "❌"
            print(f"  {solved} Clue {clue.number} {clue.direction}: '{clue.text}' = {current_word}")
            
        # Show some expected answers for debugging (hard puzzle samples)
        print(f"\n🎯 Expected Answers (Sample):")
        expected_answers = {
            "1_across": "OEDIPUSREX",    # Greek tragedy (7,3)
            "2_across": "PERANNUM",     # A year (3,5)
            "3_across": "OVAL",         # Elliptical shape (4)
            "4_across": "ACHE",         # Feeling of discomfort (4)
            "5_across": "ESSENCE",      # Kernel (7)
            "12_down": "OMEGA",         # Greek letter (5)
            "13_down": "DRACHMA",       # Greek money, formerly (7)
            "14_down": "PUNY"           # Small and weak (4)
        }
        for clue in puzzle.clues[:8]:  # Show first 8 clues
            clue_key = f"{clue.number}_{clue.direction}"
            if clue_key in expected_answers:
                print(f"  Clue {clue.number} {clue.direction}: '{clue.text}' → {expected_answers[clue_key]}")
            
        return stats['success']
        
    except Exception as e:
        print(f"❌ Error testing hard puzzle: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hard_puzzle()
    print(f"\n🎯 Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
