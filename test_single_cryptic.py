#!/usr/bin/env python3
"""
Test script for a single cryptic puzzle with detailed visualization logging
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

def test_cryptic_puzzle():
    """Test the cryptic puzzle with detailed logging"""
    print("🧩 Testing Cryptic Puzzle with VisualizationAgent")
    print("=" * 60)
    
    try:
        # Load the puzzle
        puzzle = load_puzzle("data/cryptic.json")
        print(f"📋 Loaded puzzle: {puzzle.width}x{puzzle.height} grid, {len(puzzle.clues)} clues")
        
        # Show clues
        print("\n📝 Cryptic Clues:")
        for clue in puzzle.clues:
            print(f"  {clue.number}. {clue.text} ({clue.length} letters, {clue.direction})")
        
        print(f"\n🎯 Initial grid:")
        print(puzzle)
        
        # Create specialized solver for cryptic difficulty
        solver = AgenticCrosswordSolver(difficulty="cryptic")
        
        print(f"\n🤖 Using CrypticCrosswordCoordinator with:")
        print(f"  ⚙️ Max iterations: {solver.coordinator.max_iterations}")
        print(f"  ↔️ Backtracking: {solver.coordinator.backtrack_enabled}")
        print(f"  🧠 Thinking depth: {solver.coordinator.thinking_depth}")
        print(f"  ⚡ Async solving: {getattr(solver.coordinator, 'use_async_solving', False)}")
        print(f"  🔀 Wordplay analysis: Enabled")
        print(f"  🧩 Anagram detection: Enabled")
        
        # Create log path
        log_path = "logs/cryptic_detailed_test.json"
        os.makedirs("logs", exist_ok=True)
        
        print("\n🔍 Starting solve with detailed cryptic wordplay analysis...")
        stats = solver.solve_with_stats(puzzle, puzzle_name="Cryptic_Test", log_path=log_path)
        
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
            
        # Show cryptic clue analysis
        print(f"\n🧩 Cryptic Clues Analysis:")
        cryptic_examples = {
            "1_across": {
                "clue": "Deliver dollar to complete start of betting spreads (7)",
                "answer": "BUTTERS",
                "wordplay": "B(etting) + UTTER$ (deliver dollar)"
            },
            "2_across": {
                "clue": "Campaigned for B. Dole surprisingly winning twice (7)",
                "answer": "LOBBIED", 
                "wordplay": "Anagram of 'B Dole' + BI (twice)"
            },
            "3_across": {
                "clue": "Discovered hot curry initially taken away (5)",
                "answer": "SPIED",
                "wordplay": "SP-ICE-D with H(ot) and C(urry) removed"
            },
            "15_down": {
                "clue": "Cry about time working in US city (6)",
                "answer": "BOSTON",
                "wordplay": "SOB (cry) around T (time) + ON (working)"
            }
        }
        
        for clue_key, info in cryptic_examples.items():
            print(f"  🔑 {clue_key}: '{info['clue']}'")
            print(f"     → {info['answer']} ({info['wordplay']})")
            
        return stats['success']
        
    except Exception as e:
        print(f"❌ Error testing cryptic puzzle: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cryptic_puzzle()
    print(f"\n🎯 Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
