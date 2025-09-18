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

def iterative_solver_review_loop(puzzle: CrosswordPuzzle, solver: AgenticCrosswordSolver, max_cycles: int = 8) -> dict:
    """
    Apply iterative cycles of solver + review system until convergence or max cycles
    Each cycle: 1) Run solver, 2) Apply review system, 3) Check for improvement
    """
    print(f"\n🔄 Iterative Solver + Review Loop (max {max_cycles} cycles)")
    print("=" * 60)
    
    initial_solved = sum(1 for clue in puzzle.clues if clue.answered)
    best_solved = initial_solved
    total_improvements = 0
    cycles_completed = 0
    
    for cycle in range(1, max_cycles + 1):
        print(f"\n🔄 CYCLE {cycle}/{max_cycles}")
        print("-" * 40)
        
        cycle_start_solved = sum(1 for clue in puzzle.clues if clue.answered)
        
        # Phase 1: Run solver with fresh state
        print(f"  🎯 Phase 1: Running solver pass...")
        
        # Reset solver state for fresh attempt
        if hasattr(solver.coordinator, 'last_iteration_candidates'):
            solver.coordinator.last_iteration_candidates.clear()
        if hasattr(solver.coordinator, 'failed_clue_attempts'):
            solver.coordinator.failed_clue_attempts.clear()
        
        # Gradually increase aggressiveness with each cycle
        if cycle <= 2:
            # Early cycles: standard approach
            print(f"    🎯 Cycle {cycle}: Standard approach")
        elif cycle <= 4:
            # Mid cycles: more aggressive
            print(f"    🎯 Cycle {cycle}: More aggressive thresholds")
            solver.coordinator.confidence_threshold = max(0.15, 0.3 - cycle * 0.05)
            solver.coordinator.review_threshold = max(0.05, 0.15 - cycle * 0.02)
        else:
            # Late cycles: very permissive
            print(f"    🎯 Cycle {cycle}: Very permissive approach")
            solver.coordinator.confidence_threshold = 0.1
            solver.coordinator.review_threshold = 0.03
            solver.coordinator.max_iterations = min(20, 12 + cycle * 2)  # Even more iterations
        
        # Run solver
        solver_success = solver.solve(puzzle, verbose=False, puzzle_name=f"Hard_Cycle_{cycle}")
        after_solver_solved = sum(1 for clue in puzzle.clues if clue.answered)
        solver_improvement = after_solver_solved - cycle_start_solved
        
        print(f"    📊 Solver result: +{solver_improvement} clues ({after_solver_solved}/{len(puzzle.clues)})")
        
        # Phase 2: Apply review system if available and not fully solved
        review_improvement = 0
        if after_solver_solved < len(puzzle.clues) and solver.review_system:
            print(f"  🎭 Phase 2: Applying review system...")
            
            try:
                solver_log = solver.coordinator.solver_log if hasattr(solver.coordinator, 'solver_log') else None
                
                if solver_log:
                    corrections_applied, review_report = solver.review_system.review_and_correct(
                        puzzle, solver_log, max_corrections=8  # More corrections per cycle
                    )
                    
                    after_review_solved = sum(1 for clue in puzzle.clues if clue.answered)
                    review_improvement = after_review_solved - after_solver_solved
                    
                    if corrections_applied:
                        print(f"    ✅ Review applied {review_improvement} corrections")
                    else:
                        print(f"    ℹ️  Review found no applicable corrections")
                else:
                    print(f"    ❌ No solver log available for review")
                    
            except Exception as e:
                print(f"    ❌ Review system error: {e}")
        else:
            print(f"  🎭 Phase 2: Skipping review (puzzle complete or review unavailable)")
        
        # Calculate cycle results
        cycle_end_solved = sum(1 for clue in puzzle.clues if clue.answered)
        cycle_improvement = cycle_end_solved - cycle_start_solved
        total_improvements += cycle_improvement
        
        print(f"  📊 Cycle {cycle} total: +{cycle_improvement} clues")
        print(f"  📈 Current progress: {cycle_end_solved}/{len(puzzle.clues)} ({cycle_end_solved/len(puzzle.clues)*100:.1f}%)")
        
        # Update best result
        if cycle_end_solved > best_solved:
            best_solved = cycle_end_solved
        
        cycles_completed = cycle
        
        # Check for completion
        if cycle_end_solved >= len(puzzle.clues):
            print(f"  🎉 PUZZLE COMPLETED in cycle {cycle}!")
            break
        
        # Check for no improvement (convergence)
        if cycle_improvement == 0:
            print(f"  ⏹️  No improvement in cycle {cycle}")
            if cycle >= 3:  # Allow at least 3 cycles before giving up
                print(f"  🛑 Stopping early - no progress for multiple cycles")
                break
        
        # Reset thresholds for next cycle
        solver.coordinator.confidence_threshold = 0.3
        solver.coordinator.review_threshold = 0.15
        solver.coordinator.max_iterations = 12
    
    final_solved = sum(1 for clue in puzzle.clues if clue.answered)
    
    print(f"\n📊 Iterative Loop Summary:")
    print(f"  🔄 Cycles completed: {cycles_completed}/{max_cycles}")
    print(f"  📈 Total improvement: +{total_improvements} clues")
    print(f"  🎯 Final result: {final_solved}/{len(puzzle.clues)} clues ({final_solved/len(puzzle.clues)*100:.1f}%)")
    print(f"  ✅ Success: {final_solved >= len(puzzle.clues)}")
    
    return {
        'cycles_completed': cycles_completed,
        'initial_solved': initial_solved,
        'final_solved': final_solved,
        'total_improvement': total_improvements,
        'completion_rate': final_solved / len(puzzle.clues),
        'success': final_solved >= len(puzzle.clues)
    }


def apply_additional_solver_passes(puzzle: CrosswordPuzzle, solver: AgenticCrosswordSolver, max_passes: int = 5) -> int:
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
        
        # Reset solver state for fresh attempt
        if hasattr(solver.coordinator, 'last_iteration_candidates'):
            solver.coordinator.last_iteration_candidates.clear()
        if hasattr(solver.coordinator, 'failed_clue_attempts'):
            solver.coordinator.failed_clue_attempts.clear()
            
        # Modify strategy for different passes
        if pass_num == 1:
            # Pass 1: Standard approach
            print(f"    🎯 Pass {pass_num}: Standard solving approach")
        elif pass_num == 2:
            # Pass 2: More aggressive (lower confidence threshold)
            print(f"    🎯 Pass {pass_num}: More aggressive (lower thresholds)")
            original_confidence = solver.coordinator.confidence_threshold
            solver.coordinator.confidence_threshold = max(0.2, original_confidence - 0.1)
        elif pass_num == 3:
            # Pass 3: Focus on multi-word clues first
            print(f"    🎯 Pass {pass_num}: Multi-word clue priority")
            solver.coordinator.confidence_threshold = 0.2
        else:
            # Pass 4+: Very permissive
            print(f"    🎯 Pass {pass_num}: Very permissive approach")
            solver.coordinator.confidence_threshold = 0.1
            solver.coordinator.review_threshold = 0.05
            
        # Apply solver again on the current state
        pass_success = solver.solve(puzzle, verbose=False, puzzle_name=f"Hard_Test_Pass_{pass_num}")
        
        # Reset thresholds after each pass
        if pass_num >= 2:
            solver.coordinator.confidence_threshold = 0.3  # Reset to hard defaults
            solver.coordinator.review_threshold = 0.15
        
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
        
        # Apply iterative solver + review loop until convergence
        if stats['completion_rate'] < 1.0:
            print(f"\n🔄 Starting Iterative Solver + Review Loop...")
            final_stats = iterative_solver_review_loop(puzzle, solver, max_cycles=8)
            
            # Update stats with final results
            stats['finally_solved'] = final_stats['final_solved']
            stats['completion_rate'] = final_stats['completion_rate']
            stats['success'] = final_stats['success']
            stats['total_cycles'] = final_stats['cycles_completed']
        
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
        if 'total_cycles' in stats:
            print(f"  🔄 Iterative Cycles: {stats['total_cycles']}")
        print(f"  ✅ Solution Consistency: {validation_success}")
        print(f"  🧩 Solver Method: Iterative agentic solving with review system")
        
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
