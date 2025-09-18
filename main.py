#!/usr/bin/env python3
"""
Main Entry Point - Agentic Crossword Solver

This is the main entry point for the agentic crossword solver.
It provides a simple interface to test the solver with different puzzles.
"""

import os
import sys
from dotenv import load_dotenv

from src.crossword.utils import load_puzzle
from src.solver.main_solver import AgenticCrosswordSolver

# Load environment variables
load_dotenv()

def main():
    """Main function demonstrating the crossword solver"""
    print("🧩 Agentic Crossword Solver")
    print("=" * 40)
    
    # Test connection to OpenAI
    print("\n🔧 Testing OpenAI connection...")
    try:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_version=os.getenv("OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are a helpful assistant."}, 
                      {"role": "user", "content": "Hello! Just testing the connection."}],
            max_tokens=50
        )
        print("✅ OpenAI connection successful!")
        print(f"   Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ OpenAI connection failed: {e}")
        print("   Please check your .env file configuration")
        return
    
    # Demonstrate basic crossword functionality
    print("\n🧩 Testing Basic Crossword Functionality...")
    try:
        # Load a simple puzzle
        puzzle = load_puzzle("data/easy.json")
        print(f"📋 Loaded puzzle: {puzzle.width}x{puzzle.height} grid with {len(puzzle.clues)} clues")
        
        # Show initial state
        print("\n🎯 Initial puzzle grid:")
        print(puzzle)
        
        # Show first few clues
        print("\n📝 Sample clues:")
        for i, clue in enumerate(puzzle.clues[:3]):
            print(f"  {i+1}. {clue.text} ({clue.length} letters, {clue.direction.name})")
        
        # Demonstrate the agentic solver
        print("\n🤖 Testing Agentic Solver...")
        solver = AgenticCrosswordSolver(difficulty="easy")
        print(f"   Created solver with review system: {solver.enable_review}")
        
        # Solve the puzzle
        success = solver.solve(puzzle, verbose=True, puzzle_name="main_demo")
        
        # Show results
        solved_clues = sum(1 for clue in puzzle.clues if clue.answered)
        completion_rate = solved_clues / len(puzzle.clues)
        
        print(f"\n📊 Results:")
        print(f"   ✅ Success: {success}")
        print(f"   📈 Completion: {solved_clues}/{len(puzzle.clues)} clues ({completion_rate:.1%})")
        
        print(f"\n🎯 Final puzzle grid:")
        print(puzzle)
        
        # Validate solution
        if puzzle.validate_all():
            print("✅ Solution is completely correct!")
        else:
            print("⚠️  Solution has some gaps - this is normal for a demo")
            
    except Exception as e:
        print(f"❌ Error during puzzle solving: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎉 Demo completed!")
    print(f"\n💡 Next steps:")
    print(f"   • Run examples/basic_usage.py for more examples")
    print(f"   • Run examples/test_single_medium.py to test medium puzzles")
    print(f"   • Run examples/test_single_hard.py to see the review system in action")
    print(f"   • Check the logs/ directory for detailed solving logs")

if __name__ == "__main__":
    main()