#!/usr/bin/env python3
"""
Test script for a single hard puzzle with detailed visualization logging,
review system, and final completion fallback
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add parent directory to path to import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.crossword.utils import load_puzzle
from src.crossword.crossword import CrosswordPuzzle
from src.solver.main_solver import AgenticCrosswordSolver
from src.solver.review_system import TwoStageReviewSystem

# Load environment variables
load_dotenv()

def apply_additional_solver_passes(puzzle: CrosswordPuzzle, solver: AgenticCrosswordSolver, max_passes: int = 3) -> int:
    """
    Apply additional solver passes to try to complete more clues
    This uses only the solver's own capabilities, no ground truth
    """
    print(f"\n🔄 Applying Additional Solver Passes (max {max_passes})...")
    
    initial_solved = sum(1 for clue in puzzle.clues if clue.answered)
    total_improvements = 0
    
    for pass_num in range(1, max_passes + 1):
        print(f"\n  🔄 Pass {pass_num}:")
        
        # Try to solve remaining unsolved clues with fresh perspective
        unsolved_clues = [clue for clue in puzzle.clues if not clue.answered]
        
        if not unsolved_clues:
            print(f"    ✅ All clues already solved!")
            break
        
        print(f"    🎯 Attempting to solve {len(unsolved_clues)} remaining clues...")
        
        # Apply solver again on the current state
        pass_success = solver.solve(puzzle, verbose=False, puzzle_name=f"Hard_Test_Pass_{pass_num}")
        
        current_solved = sum(1 for clue in puzzle.clues if clue.answered)
        improvements_this_pass = current_solved - (initial_solved + total_improvements)
        total_improvements += improvements_this_pass
        
        print(f"    📊 Pass {pass_num} result: +{improvements_this_pass} clues solved")
        print(f"    📈 Current total: {current_solved}/{len(puzzle.clues)} clues ({current_solved/len(puzzle.clues)*100:.1f}%)")
        
        if improvements_this_pass == 0:
            print(f"    ⏹️  No progress in pass {pass_num}, stopping additional passes")
            break
    
    final_solved = sum(1 for clue in puzzle.clues if clue.answered)
    print(f"\n  📊 Additional passes result: +{total_improvements} clues solved")
    print(f"  📈 Final completion: {final_solved}/{len(puzzle.clues)} clues ({final_solved/len(puzzle.clues)*100:.1f}%)")
    
    return total_improvements

def test_hard_puzzle():
    """Test the hard puzzle with detailed logging"""
    print("🧩 Testing Hard Puzzle with VisualizationAgent")
    print("=" * 60)
    
    try:
        # Load the puzzle
        puzzle = load_puzzle("../data/hard.json")
        print(f"📋 Loaded puzzle: {puzzle.width}x{puzzle.height} grid, {len(puzzle.clues)} clues")
        
        # Show clues
        print("\n📝 Clues:")
        for clue in puzzle.clues:
            print(f"  {clue.number}. {clue.text} ({clue.length} letters, {clue.direction})")
        
        print(f"\n🎯 Initial grid:")
        print(puzzle)
        
        # Create specialized solver for hard difficulty with review system enabled
        solver = AgenticCrosswordSolver(difficulty="hard")  # Review system auto-enabled for hard difficulty
        
        print(f"\n🤖 Using HardCrosswordCoordinator with:")
        print(f"  ⚙️ Max iterations: {solver.coordinator.max_iterations}")
        print(f"  ↔️ Backtracking: {solver.coordinator.backtrack_enabled}")
        print(f"  🧠 Thinking depth: {solver.coordinator.thinking_depth}")
        print(f"  ⚡ Async solving: {getattr(solver.coordinator, 'use_async_solving', False)}")
        print(f"  🎭 Review system: {solver.enable_review} (auto-enabled for hard difficulty)")
        
        # Create log paths
        log_path = "../logs/hard_detailed_test.json"
        review_log_path = "../logs/hard_detailed_review_report.json"
        os.makedirs("../logs", exist_ok=True)
        
        print("\n🔍 Starting solve with detailed visualization logging and review system...")
        stats = solver.solve_with_stats(puzzle, puzzle_name="Hard_Test", log_path=log_path, review_log_path=review_log_path)
        
        # Display initial results
        print(f"\n📊 Initial Solver Results:")
        print(f"  ✅ Success: {stats['success']}")
        print(f"  📈 Completion: {stats['finally_solved']}/{stats['total_clues']} clues ({stats['completion_rate']:.1%})")
        print(f"  ⏱️  Time: {stats['solving_time']:.2f} seconds")
        print(f"  🎭 Review enabled: {stats['review_enabled']}")
        print(f"  📝 Log saved to: {log_path}")
        if stats['review_enabled'] and hasattr(solver, '_last_review_report'):
            print(f"  📋 Review report saved to: {review_log_path}")
        
        # Apply additional review if not fully solved and review wasn't triggered
        if not stats['success'] and stats['completion_rate'] < 1.0:
            print(f"\n🎭 Applying Manual Review System (completion: {stats['completion_rate']:.1%})...")
            
            try:
                # Get the solver log
                solver_log = solver.coordinator.solver_log if hasattr(solver.coordinator, 'solver_log') else None
                
                if solver_log and solver.review_system:
                    corrections_applied, review_report = solver.review_system.review_and_correct(
                        puzzle, solver_log, max_corrections=5
                    )
                    
                    if corrections_applied:
                        new_solved = sum(1 for clue in puzzle.clues if clue.answered)
                        print(f"  ✅ Review system applied corrections: {new_solved}/{len(puzzle.clues)} clues now solved")
                        stats['finally_solved'] = new_solved
                        stats['completion_rate'] = new_solved / len(puzzle.clues)
                    else:
                        print(f"  ℹ️  Review system found no applicable corrections")
                else:
                    print(f"  ❌ Review system unavailable (solver_log: {solver_log is not None})")
            except Exception as e:
                print(f"  ❌ Review system error: {e}")
        
        # Apply additional solver passes if still not fully solved
        if stats['completion_rate'] < 1.0:
            print(f"\n🔄 Applying Additional Solver Passes...")
            additional_solved = apply_additional_solver_passes(puzzle, solver, max_passes=3)
            
            if additional_solved > 0:
                new_solved = sum(1 for clue in puzzle.clues if clue.answered)
                stats['finally_solved'] = new_solved
                stats['completion_rate'] = new_solved / len(puzzle.clues)
                stats['success'] = stats['completion_rate'] >= 1.0
        
        print(f"\n🎯 Final grid:")
        print(puzzle)
        
        # Validate solution
        validation_success = puzzle.validate_all()
        if validation_success:
            print("✅ Solution is CORRECT!")
        else:
            print("❌ Solution has errors")
        
        # Display final comprehensive results
        print(f"\n📊 Final Comprehensive Results:")
        print(f"  ✅ Solver Success: {stats['success']}")
        print(f"  📈 Final Completion: {stats['finally_solved']}/{stats['total_clues']} clues ({stats['completion_rate']:.1%})")
        print(f"  ⏱️  Initial Solve Time: {stats['solving_time']:.2f} seconds")
        print(f"  🎭 Review System Used: {stats['review_enabled']}")
        print(f"  ✅ Solution Consistency: {validation_success}")
        print(f"  🧩 Solver Method: Agentic solving with review system and additional passes")
        
        print(f"\n📋 Visualization History:")
        print(f"  Total visualizations captured: {len(solver.coordinator.visualization_agent.visualization_history)}")
        
        # Show review system results if enabled and available
        if stats['review_enabled'] and hasattr(solver, '_last_review_report'):
            review_report = solver._last_review_report
            print(f"\n🎭 Review System Results:")
            print(f"  📊 Completion rate at review: {review_report.completion_rate:.1%}")
            print(f"  🔍 Issues identified: {len(review_report.insights)}")
            print(f"  🎯 Priority clues: {len(review_report.priority_clues)}")
            if review_report.insights:
                print(f"  📝 Key insights:")
                for insight in review_report.insights[:3]:  # Show top 3 insights
                    print(f"    - {insight.issue_type}: {insight.description}")
        
        # Show key clue states
        print(f"\n🔍 Final Clue States:")
        for clue in puzzle.clues:
            current_chars = puzzle.get_current_clue_chars(clue)
            current_word = "".join(char or "_" for char in current_chars)
            solved = "✅" if clue.answered and all(char is not None for char in current_chars) else "❌"
            print(f"  {solved} Clue {clue.number} {clue.direction}: '{clue.text}' = {current_word}")
            
        # Show summary of unsolved clues for analysis
        unsolved_clues = [clue for clue in puzzle.clues if not clue.answered]
        if unsolved_clues:
            print(f"\n❌ Remaining Unsolved Clues ({len(unsolved_clues)}):")
            for clue in unsolved_clues:
                current_chars = puzzle.get_current_clue_chars(clue)
                pattern = "".join(char or "_" for char in current_chars)
                print(f"  {clue.number} {clue.direction.name}: '{clue.text}' ({clue.length} letters) = {pattern}")
        else:
            print(f"\n🎉 All clues solved by the solver!")
            
        # Return comprehensive success (either solver succeeded or final completion achieved 100%)
        return stats['success'] and validation_success
        
    except Exception as e:
        print(f"❌ Error testing hard puzzle: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hard_puzzle()
    print(f"\n🎯 Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
