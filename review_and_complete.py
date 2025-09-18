#!/usr/bin/env python3
"""
Review Agent Script for Completing Partially Solved Crosswords

This script takes a partially solved crossword and uses advanced review techniques
to complete the remaining clues by leveraging:
1. Pattern-based reasoning
2. Cross-reference validation  
3. Advanced LLM prompting with grid context
4. Specialized clue analysis
"""

import os
import json
import logging
from typing import List, Dict, Any
from src.crossword.crossword import CrosswordPuzzle
from src.solver.agents import CrosswordTools, ClueAgent, ReviewAgent, ClueCandidate
from src.crossword.types import Clue
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class ReviewCompleter:
    """Advanced review agent for completing partially solved crosswords"""
    
    def __init__(self):
        self.tools = CrosswordTools()
        self.clue_agent = ClueAgent(self.tools)
        self.review_agent = ReviewAgent(self.tools)
        
    def load_puzzle_state(self, puzzle_file: str) -> CrosswordPuzzle:
        """Load puzzle from JSON file"""
        with open(puzzle_file, 'r') as f:
            puzzle_data = json.load(f)
        
        # Convert clue data to Clue objects
        from src.crossword.types import Clue, Direction
        clues = []
        for clue_data in puzzle_data['clues']:
            direction = Direction.ACROSS if clue_data['direction'] == 'across' else Direction.DOWN
            clue = Clue(
                number=clue_data['number'],
                text=clue_data['text'],
                direction=direction,
                length=clue_data['length'],
                row=clue_data['row'],
                col=clue_data['col'],
                answer=clue_data.get('answer', ''),
                answered=clue_data.get('answered', False)
            )
            clues.append(clue)
        
        # Create puzzle
        puzzle = CrosswordPuzzle(
            width=puzzle_data['width'],
            height=puzzle_data['height'],
            clues=clues
        )
        
        return puzzle
    
    def analyze_unsolved_clues(self, puzzle: CrosswordPuzzle) -> List[Clue]:
        """Identify clues that are not fully solved"""
        unsolved = []
        for clue in puzzle.clues:
            if not clue.answered:
                # Check if it's partially filled
                current_chars = puzzle.get_current_clue_chars(clue)
                if any(char is not None for char in current_chars):
                    logger.warning(f"🔍 Partially filled: '{clue.text}' = {''.join(c if c else '_' for c in current_chars)}")
                else:
                    logger.warning(f"🔍 Empty: '{clue.text}'")
                unsolved.append(clue)
        return unsolved
    
    def get_pattern_context(self, puzzle: CrosswordPuzzle, clue: Clue) -> str:
        """Build rich context from current grid state"""
        current_chars = puzzle.get_current_clue_chars(clue)
        pattern = "".join(char if char else "_" for char in current_chars)
        
        # Get intersecting clues and their states
        intersecting_info = []
        for i, char in enumerate(current_chars):
            if char is not None:
                # Find intersecting clue at this position
                row, col = puzzle.get_clue_position(clue, i)
                intersecting_clue = puzzle.find_intersecting_clue(clue, i)
                if intersecting_clue:
                    int_chars = puzzle.get_current_clue_chars(intersecting_clue)
                    int_pattern = "".join(c if c else "_" for c in int_chars)
                    intersecting_info.append(f"  - Position {i+1}: '{char}' from '{intersecting_clue.text}' ({int_pattern})")
        
        context = f"""
Current Pattern: {pattern}
Length Required: {clue.length} letters
Direction: {clue.direction}

Intersecting Constraints:
{chr(10).join(intersecting_info) if intersecting_info else "  - No current intersections"}

Grid Context: Letters already placed provide strong constraints for this answer.
"""
        return context
    
    def solve_with_pattern_reasoning(self, puzzle: CrosswordPuzzle, clue: Clue) -> List[ClueCandidate]:
        """Use pattern-aware reasoning to solve clues"""
        pattern_context = self.get_pattern_context(puzzle, clue)
        current_chars = puzzle.get_current_clue_chars(clue)
        current_pattern = "".join(char if char else "_" for char in current_chars)
        
        # Build specialized prompt for pattern-based solving
        prompt = f"""
You are solving a crossword clue with STRONG CONSTRAINTS from intersecting words.

CLUE: "{clue.text}"
REQUIRED LENGTH: {clue.length} letters
CURRENT PATTERN: {current_pattern}

{pattern_context}

CRITICAL CONSTRAINTS:
- Any letter shown in the pattern is FIXED and cannot be changed
- Your answer must fit this EXACT pattern
- Every underscore (_) represents a letter you need to provide
- Consider what words fit this specific letter pattern

ANALYSIS APPROACH:
1. Look at the clue type and meaning
2. Consider what {clue.length}-letter words match the pattern {current_pattern}
3. Think about common crossword answers for this type of clue
4. Verify your answer matches the pattern exactly

For the clue "{clue.text}" with pattern {current_pattern}:
- What category of answer does this clue suggest?
- What common words fit this length and pattern?
- Does your answer make sense for the clue meaning?

Provide your best answer that fits the pattern exactly:
ANSWER: [word] | CONFIDENCE: [0.0-1.0] | REASONING: [detailed explanation of how it fits pattern and clue]
ALT1: [word] | CONFIDENCE: [0.0-1.0] | REASONING: [explanation]
ALT2: [word] | CONFIDENCE: [0.0-1.0] | REASONING: [explanation]
"""

        try:
            response = self.tools.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert crossword solver specializing in pattern-based reasoning."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            candidates = self.tools._parse_candidates(response.choices[0].message.content, clue)
            return candidates
            
        except Exception as e:
            logger.error(f"Error in pattern reasoning for '{clue.text}': {e}")
            return []
    
    def review_and_complete(self, puzzle: CrosswordPuzzle) -> Dict[str, Any]:
        """Main method to review and complete the puzzle"""
        logger.warning("🔍 Starting Review and Completion Process")
        
        # Analyze current state
        unsolved_clues = self.analyze_unsolved_clues(puzzle)
        logger.warning(f"📋 Found {len(unsolved_clues)} unsolved clues")
        
        if not unsolved_clues:
            logger.warning("✅ Puzzle is already complete!")
            return {"success": True, "completed_clues": 0}
        
        completed_clues = []
        
        # Process each unsolved clue
        for clue in unsolved_clues:
            logger.warning(f"\n🎯 Analyzing: '{clue.text}' ({clue.length} letters)")
            
            # Get candidates using pattern reasoning
            candidates = self.solve_with_pattern_reasoning(puzzle, clue)
            
            if not candidates:
                logger.warning(f"❌ No candidates generated for '{clue.text}'")
                continue
            
            # Review and select best candidate
            best_candidate = None
            best_score = 0.0
            
            for candidate in candidates:
                # Validate pattern fit
                current_chars = puzzle.get_current_clue_chars(clue)
                word = candidate.word.upper()
                
                # Check if word fits the current pattern
                fits_pattern = True
                if len(word) == len(current_chars):
                    for i, (current_char, word_char) in enumerate(zip(current_chars, word)):
                        if current_char is not None and current_char != word_char:
                            fits_pattern = False
                            break
                else:
                    fits_pattern = False
                
                if not fits_pattern:
                    logger.warning(f"❌ '{word}' doesn't fit pattern")
                    continue
                
                # Review the candidate
                review_score = self.review_agent.review_solution(clue, candidate, puzzle)
                total_score = candidate.confidence * review_score
                
                logger.warning(f"🎯 '{word}': confidence={candidate.confidence:.2f}, review={review_score:.2f}, total={total_score:.2f}")
                
                if total_score > best_score:
                    best_candidate = candidate
                    best_score = total_score
            
            # Apply best candidate if found
            if best_candidate and best_score > 0.4:  # Threshold for acceptance
                try:
                    puzzle.set_clue_answer(clue, best_candidate.word)
                    completed_clues.append({
                        "clue": clue.text,
                        "answer": best_candidate.word,
                        "confidence": best_candidate.confidence,
                        "score": best_score
                    })
                    logger.warning(f"✅ SOLVED: '{clue.text}' = {best_candidate.word} (score: {best_score:.2f})")
                except Exception as e:
                    logger.error(f"❌ Failed to apply answer '{best_candidate.word}' for '{clue.text}': {e}")
            else:
                logger.warning(f"❌ No acceptable solution found for '{clue.text}' (best score: {best_score:.2f})")
        
        return {
            "success": len(completed_clues) > 0,
            "completed_clues": len(completed_clues),
            "solutions": completed_clues
        }

def main():
    """Run the review and completion process"""
    print("🔍 Crossword Review and Completion System")
    print("=========================================")
    
    # Load the puzzle (you can modify this path)
    puzzle_file = "data/hard.json"
    
    # Initialize the reviewer
    reviewer = ReviewCompleter()
    
    try:
        # Load puzzle
        puzzle = reviewer.load_puzzle_state(puzzle_file)
        print(f"📋 Loaded puzzle: {puzzle.width}x{puzzle.height} grid, {len(puzzle.clues)} clues")
        
        # Show current completion status
        answered_clues = sum(1 for clue in puzzle.clues if clue.answered)
        completion_rate = (answered_clues / len(puzzle.clues)) * 100
        print(f"📊 Current completion: {answered_clues}/{len(puzzle.clues)} clues ({completion_rate:.1f}%)")
        
        # Run review and completion
        result = reviewer.review_and_complete(puzzle)
        
        # Show results
        print(f"\n📈 Review Results:")
        print(f"  ✅ Success: {result['success']}")
        print(f"  🎯 New solutions: {result['completed_clues']}")
        
        if result['success'] and 'solutions' in result:
            print(f"\n🔍 Solutions found:")
            for solution in result['solutions']:
                print(f"  • '{solution['clue']}' = {solution['answer']} (score: {solution['score']:.2f})")
        
        # Show final completion status
        final_answered = sum(1 for clue in puzzle.clues if clue.answered)
        final_completion = (final_answered / len(puzzle.clues)) * 100
        print(f"\n📊 Final completion: {final_answered}/{len(puzzle.clues)} clues ({final_completion:.1f}%)")
        
        if final_completion == 100.0:
            print("🎉 PUZZLE COMPLETE!")
        else:
            remaining_clues = [clue for clue in puzzle.clues if not clue.answered]
            print(f"\n❌ Remaining unsolved clues:")
            for clue in remaining_clues:
                current_chars = puzzle.get_current_clue_chars(clue)
                pattern = "".join(char if char else "_" for char in current_chars)
                print(f"  • '{clue.text}' = {pattern}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
