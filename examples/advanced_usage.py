#!/usr/bin/env python3
"""
Advanced Usage Example - Agentic Crossword Solver

This example demonstrates advanced features including:
- Custom solver configuration
- Review system usage
- Multi-pass solving
- Performance analysis
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add parent directory to path to import solver
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.crossword.utils import load_puzzle
from src.solver.main_solver import AgenticCrosswordSolver
from src.solver.review_system import TwoStageReviewSystem

# Load environment variables
load_dotenv()

def advanced_solver_configuration():
    """
    Example of advanced solver configuration options
    """
    print("🔧 Advanced Solver Configuration")
    print("=" * 40)
    
    # Load a hard puzzle
    puzzle = load_puzzle("../data/hard.json")
    
    # Example 1: Explicit review system control
    print("\n1️⃣ Custom Review System Configuration:")
    
    # Solver with review explicitly enabled
    solver_with_review = AgenticCrosswordSolver(
        difficulty="medium",  # Medium difficulty but force review
        enable_review=True    # Explicitly enable review system
    )
    
    # Solver with review explicitly disabled
    solver_no_review = AgenticCrosswordSolver(
        difficulty="hard",    # Hard difficulty but disable review
        enable_review=False   # Explicitly disable review system
    )
    
    print(f"  Medium solver with review: {solver_with_review.enable_review}")
    print(f"  Hard solver without review: {solver_no_review.enable_review}")
    
    return solver_with_review, solver_no_review

def comparison_analysis():
    """
    Example of comparing solver performance with and without review system
    """
    print("\n📊 Performance Comparison Analysis")
    print("=" * 40)
    
    # Load the same puzzle twice for fair comparison
    puzzle1 = load_puzzle("../data/hard.json")
    puzzle2 = load_puzzle("../data/hard.json")
    
    # Test without review system
    print("\n🔍 Testing WITHOUT Review System:")
    solver_no_review = AgenticCrosswordSolver(difficulty="hard", enable_review=False)
    stats_no_review = solver_no_review.solve_with_stats(
        puzzle1, 
        puzzle_name="hard_no_review",
        log_path="../logs/comparison_no_review.json"
    )
    
    # Test with review system
    print("\n🎭 Testing WITH Review System:")
    solver_with_review = AgenticCrosswordSolver(difficulty="hard", enable_review=True)
    stats_with_review = solver_with_review.solve_with_stats(
        puzzle2,
        puzzle_name="hard_with_review", 
        log_path="../logs/comparison_with_review.json",
        review_log_path="../logs/comparison_review_report.json"
    )
    
    # Compare results
    print(f"\n📈 Comparison Results:")
    print(f"{'Metric':<20} {'No Review':<15} {'With Review':<15} {'Improvement'}")
    print("-" * 65)
    
    completion_diff = stats_with_review['completion_rate'] - stats_no_review['completion_rate']
    time_diff = stats_with_review['solving_time'] - stats_no_review['solving_time']
    clue_diff = stats_with_review['finally_solved'] - stats_no_review['finally_solved']
    
    print(f"{'Completion Rate':<20} {stats_no_review['completion_rate']:<14.1%} {stats_with_review['completion_rate']:<14.1%} {completion_diff:+.1%}")
    print(f"{'Clues Solved':<20} {stats_no_review['finally_solved']:<14} {stats_with_review['finally_solved']:<14} {clue_diff:+}")
    print(f"{'Solving Time':<20} {stats_no_review['solving_time']:<13.1f}s {stats_with_review['solving_time']:<13.1f}s {time_diff:+.1f}s")
    
    return stats_no_review, stats_with_review

def manual_review_system_usage():
    """
    Example of manually using the review system
    """
    print("\n🎭 Manual Review System Usage")
    print("=" * 40)
    
    # Load puzzle and solve partially
    puzzle = load_puzzle("../data/hard.json")
    solver = AgenticCrosswordSolver(difficulty="hard", enable_review=False)  # Disable auto-review
    
    print("🔍 Initial solving (review disabled)...")
    solver.solve(puzzle, verbose=False, puzzle_name="manual_review_test")
    
    initial_solved = sum(1 for clue in puzzle.clues if clue.answered)
    print(f"  Initial completion: {initial_solved}/{len(puzzle.clues)} clues")
    
    # Manually apply review system
    if hasattr(solver.coordinator, 'solver_log') and solver.coordinator.solver_log:
        print("\n🎭 Applying manual review system...")
        review_system = TwoStageReviewSystem()
        
        corrections_applied, review_report = review_system.review_and_correct(
            puzzle, 
            solver.coordinator.solver_log,
            max_corrections=5
        )
        
        final_solved = sum(1 for clue in puzzle.clues if clue.answered)
        
        print(f"📊 Review Results:")
        print(f"  Corrections applied: {corrections_applied}")
        print(f"  Final completion: {final_solved}/{len(puzzle.clues)} clues")
        print(f"  Improvement: +{final_solved - initial_solved} clues")
        print(f"  Issues identified: {len(review_report.insights)}")
        
        # Save detailed review report
        review_system.save_review_report(review_report, "../logs/manual_review_report.json")
        print(f"  Review report saved to: ../logs/manual_review_report.json")
        
        return review_report
    else:
        print("❌ No solver log available for review")
        return None

def analyze_solving_logs():
    """
    Example of analyzing detailed solving logs
    """
    print("\n📊 Solving Log Analysis")
    print("=" * 30)
    
    # Check if we have recent logs to analyze
    log_files = [
        "../logs/comparison_with_review.json",
        "../logs/comparison_no_review.json"
    ]
    
    for log_file in log_files:
        if os.path.exists(log_file):
            print(f"\n📝 Analyzing: {os.path.basename(log_file)}")
            
            try:
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
                
                # Extract key metrics
                if 'iterations' in log_data:
                    iterations = len(log_data['iterations'])
                    print(f"  Solving iterations: {iterations}")
                    
                    # Count progress per iteration
                    progress_count = 0
                    for iteration in log_data['iterations']:
                        if isinstance(iteration, dict) and iteration.get('progress_made', False):
                            progress_count += 1
                    
                    print(f"  Productive iterations: {progress_count}/{iterations}")
                    print(f"  Efficiency: {progress_count/iterations:.1%}")
                
                if 'final_stats' in log_data:
                    stats = log_data['final_stats']
                    print(f"  Final completion: {stats.get('completion_rate', 0):.1%}")
                    
            except Exception as e:
                print(f"  ❌ Error reading log: {e}")
        else:
            print(f"❌ Log file not found: {log_file}")

def main():
    """Main advanced usage demonstration"""
    print("🚀 Agentic Crossword Solver - Advanced Usage Examples")
    print("=" * 70)
    
    # Example 1: Advanced configuration
    print("\n📝 Example 1: Advanced Configuration")
    solver_with_review, solver_no_review = advanced_solver_configuration()
    
    # Example 2: Performance comparison
    print("\n📝 Example 2: Performance Comparison")
    stats_no_review, stats_with_review = comparison_analysis()
    
    # Example 3: Manual review system
    print("\n📝 Example 3: Manual Review System")
    review_report = manual_review_system_usage()
    
    # Example 4: Log analysis
    print("\n📝 Example 4: Log Analysis")
    analyze_solving_logs()
    
    print(f"\n🎉 Advanced examples completed!")
    print(f"\n💡 Key Takeaways:")
    print(f"  • Review system can be explicitly controlled")
    print(f"  • Performance can be measured and compared")
    print(f"  • Manual review provides detailed insights")
    print(f"  • Logs contain rich debugging information")

if __name__ == "__main__":
    main()
