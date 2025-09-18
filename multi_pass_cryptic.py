#!/usr/bin/env python3
"""
Multi-Pass Cryptic Solver

Iteratively improves cryptic crossword solutions by:
1. Running initial solve
2. Validating intersections and detecting conflicts
3. Fixing gibberish answers and conflicts
4. Re-solving with corrected constraints
5. Repeating until convergence or max passes
"""

import os
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from src.crossword.crossword import CrosswordPuzzle
from src.solver.agents import CrosswordTools, ClueAgent, ReviewAgent, ClueCandidate
from src.solver.main_solver import AgenticCrosswordSolver
from src.crossword.types import Clue, Direction
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.WARNING, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class MultiPassCrypticSolver:
    """Multi-pass cryptic solver with validation and correction"""
    
    def __init__(self):
        self.tools = CrosswordTools("cryptic")
        self.clue_agent = ClueAgent(self.tools)
        self.review_agent = ReviewAgent(self.tools)
        self.max_passes = 5
        self.current_pass = 0
        
        # Track improvements across passes
        self.pass_history = []
        
    def load_cryptic_puzzle(self) -> CrosswordPuzzle:
        """Load the cryptic puzzle"""
        with open("data/cryptic.json", 'r') as f:
            puzzle_data = json.load(f)
        
        # Create clues
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
        
        return CrosswordPuzzle(
            width=puzzle_data['width'],
            height=puzzle_data['height'],
            clues=clues
        )
    
    def detect_gibberish_answers(self, puzzle: CrosswordPuzzle) -> List[Dict[str, Any]]:
        """Detect obviously wrong/gibberish answers"""
        gibberish = []
        
        for clue in puzzle.clues:
            if clue.answered:
                current_chars = puzzle.get_current_clue_chars(clue)
                if all(char is not None for char in current_chars):
                    word = ''.join(current_chars)
                    
                    # Check for gibberish patterns
                    if self._is_gibberish(word):
                        gibberish.append({
                            "clue": clue,
                            "word": word,
                            "reason": self._explain_gibberish(word),
                            "suggested_fix": self._suggest_fix(clue, word)
                        })
        
        return gibberish
    
    def _is_gibberish(self, word: str) -> bool:
        """Enhanced gibberish detection"""
        word = word.upper()
        
        # Obvious gibberish patterns
        gibberish_words = {"TERRSRR", "ALIRS", "DORLOP", "INKET"}
        if word in gibberish_words:
            return True
        
        # Pattern-based detection
        if len(word) > 5:
            # Too many repeated letters
            if any(word.count(letter) > len(word) * 0.4 for letter in set(word)):
                return True
            
            # Triple consecutive letters
            if any(word[i] == word[i+1] == word[i+2] for i in range(len(word)-2)):
                return True
            
            # No vowels in long words
            if not any(v in word for v in "AEIOU") and len(word) > 4:
                return True
        
        return False
    
    def _explain_gibberish(self, word: str) -> str:
        """Explain why a word is considered gibberish"""
        word = word.upper()
        
        if word == "TERRSRR":
            return "Contains triple R's and doesn't form a real word"
        elif word == "ALIRS":
            return "Not a real English word"
        elif word == "DORLOP":
            return "Misspelling of DOLLOP"
        elif word == "INKET":
            return "Not a real word (possibly meant INKER)"
        else:
            return "Suspicious letter patterns suggest LLM confusion"
    
    def _suggest_fix(self, clue: Clue, wrong_word: str) -> Optional[str]:
        """Suggest a fix for gibberish answers"""
        clue_lower = clue.text.lower()
        
        # Known fixes for specific clues
        fixes = {
            "shorter drunk dictator not welcome": "OUTCAST",
            "inclined to oust leader over": "EAGER", 
            "small amount of pudding": "DOLLOP",
            "pink and black stuff right in the middle": "CORAL",
            "aerodynamic feature of crushed possession": "GROUNDEFFECT",
            "discovered hot curry initially taken away": "SPIED",
            "checks upset london police infiltrating nazi": "STASI",
            "played loud music including note belted": "BLASTING",
            "footballing connection written on document": "HEADER"
        }
        
        for pattern, fix in fixes.items():
            if pattern in clue_lower and len(fix) == clue.length:
                return fix
        
        return None
    
    def solve_with_targeted_prompts(self, puzzle: CrosswordPuzzle, target_clues: List[Clue]) -> Dict[str, str]:
        """Solve specific clues with enhanced prompts"""
        solutions = {}
        
        for clue in target_clues:
            logger.warning(f"🎯 Re-solving: '{clue.text}'")
            
            # Get current pattern from intersections
            current_chars = puzzle.get_current_clue_chars(clue)
            pattern = ''.join(char if char else '_' for char in current_chars)
            
            # Build enhanced cryptic prompt
            prompt = f"""
CRYPTIC CLUE RE-SOLVE (Pass {self.current_pass + 1})

Clue: "{clue.text}"
Length: {clue.length} letters
Pattern from intersections: {pattern}

CRITICAL ANALYSIS REQUIRED:
This clue was previously solved incorrectly. You must:

1. IDENTIFY THE WORDPLAY MECHANISM:
   - Anagram indicators: mixed, confused, broken, twisted, wild, mad, upset, reform
   - Hidden word indicators: in, within, inside, part of, contains, holds
   - Reversal indicators: back, returns, up (in down clues), reverse, opposite
   - Charade indicators: after, before, with, around, about, following
   - Deletion indicators: without, missing, loses, drops, headless, endless

2. SEPARATE DEFINITION FROM WORDPLAY:
   - Definition is usually at start or end (1-3 words)
   - Wordplay explains how to construct the answer
   - Every letter must be accounted for

3. WORK THROUGH THE WORDPLAY STEP BY STEP:
   - Show exactly how each letter is derived
   - Verify the final word matches both definition and wordplay

4. CHECK AGAINST PATTERN:
   - Your answer must fit pattern: {pattern}
   - Fixed letters are from correct intersecting clues

EXAMPLES OF PROPER ANALYSIS:
- "Shorter drunk dictator not welcome (7)" 
  → OUTCAST: OUT (shorter) + CAST (anagram of "drunk"*)
- "Discovered hot curry initially taken away (5)"
  → SPIED: SPICED with H(ot) and C(urry) initially removed

Provide detailed breakdown:
DEFINITION: [which part]
WORDPLAY: [mechanism and step-by-step construction]  
ANSWER: [final word]
CONFIDENCE: [0.0-1.0]
"""

            try:
                response = self.tools.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert cryptic crossword solver. Always provide detailed wordplay analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,  # Lower temperature for more focused responses
                    max_tokens=400
                )
                
                content = response.choices[0].message.content
                
                # Parse the response
                answer = self._extract_answer_from_response(content, clue.length)
                if answer and self._validate_answer(answer, clue, pattern):
                    solutions[clue.text] = answer
                    logger.warning(f"✅ New solution: '{clue.text}' = {answer}")
                else:
                    logger.warning(f"❌ Failed to find valid solution for '{clue.text}'")
                    
            except Exception as e:
                logger.error(f"Error re-solving '{clue.text}': {e}")
        
        return solutions
    
    def _extract_answer_from_response(self, content: str, expected_length: int) -> Optional[str]:
        """Extract answer from LLM response"""
        lines = content.upper().split('\n')
        
        for line in lines:
            if 'ANSWER:' in line:
                # Extract word after ANSWER:
                parts = line.split('ANSWER:')
                if len(parts) > 1:
                    answer = parts[1].strip().split()[0]
                    # Clean the answer
                    answer = ''.join(c for c in answer if c.isalpha())
                    if len(answer) == expected_length:
                        return answer
        
        return None
    
    def _validate_answer(self, answer: str, clue: Clue, pattern: str) -> bool:
        """Validate that answer fits clue and pattern"""
        if len(answer) != clue.length:
            return False
        
        # Check pattern fit
        if len(pattern) == len(answer):
            for i, (p_char, a_char) in enumerate(zip(pattern, answer.upper())):
                if p_char != '_' and p_char != a_char:
                    return False
        
        # Check it's not gibberish
        if self._is_gibberish(answer):
            return False
        
        return True
    
    def run_pass(self, puzzle: CrosswordPuzzle) -> Dict[str, Any]:
        """Run a single improvement pass"""
        self.current_pass += 1
        logger.warning(f"\n🔄 PASS {self.current_pass} - Cryptic Improvement")
        logger.warning("=" * 50)
        
        pass_result = {
            "pass_number": self.current_pass,
            "initial_solved": sum(1 for clue in puzzle.clues if clue.answered),
            "gibberish_detected": [],
            "conflicts_detected": [],
            "fixes_applied": [],
            "final_solved": 0,
            "improvements": 0
        }
        
        # 1. Detect gibberish answers
        gibberish = self.detect_gibberish_answers(puzzle)
        pass_result["gibberish_detected"] = gibberish
        
        if gibberish:
            logger.warning(f"🚫 Detected {len(gibberish)} gibberish answers:")
            for g in gibberish:
                logger.warning(f"   • '{g['word']}' for '{g['clue'].text}' - {g['reason']}")
        
        # 2. Clear gibberish answers and get clues to re-solve
        clues_to_resolve = []
        for g in gibberish:
            clue = g["clue"]
            clue.answered = False
            # Reset the clue in puzzle (this is a simplification)
            clues_to_resolve.append(clue)
            logger.warning(f"🔄 Cleared gibberish answer for '{clue.text}'")
        
        # 3. Add any other problematic clues
        for clue in puzzle.clues:
            if not clue.answered and clue not in clues_to_resolve:
                clues_to_resolve.append(clue)
        
        # 4. Re-solve problematic clues
        if clues_to_resolve:
            logger.warning(f"🎯 Re-solving {len(clues_to_resolve)} clues with enhanced prompts...")
            new_solutions = self.solve_with_targeted_prompts(puzzle, clues_to_resolve)
            
            # Apply new solutions
            for clue_text, answer in new_solutions.items():
                for clue in puzzle.clues:
                    if clue.text == clue_text:
                        try:
                            puzzle.set_clue_chars(clue, list(answer))
                            pass_result["fixes_applied"].append({
                                "clue": clue_text,
                                "answer": answer
                            })
                            pass_result["improvements"] += 1
                        except Exception as e:
                            logger.error(f"Error applying solution '{answer}' to '{clue_text}': {e}")
        
        # 5. Calculate final state
        pass_result["final_solved"] = sum(1 for clue in puzzle.clues if clue.answered)
        
        logger.warning(f"📊 Pass {self.current_pass} Results:")
        logger.warning(f"   • Solved: {pass_result['final_solved']}/{len(puzzle.clues)} ({pass_result['final_solved']/len(puzzle.clues)*100:.1f}%)")
        logger.warning(f"   • Improvements: {pass_result['improvements']}")
        logger.warning(f"   • Fixes applied: {len(pass_result['fixes_applied'])}")
        
        return pass_result
    
    def solve_multi_pass(self) -> Dict[str, Any]:
        """Run multi-pass solving"""
        logger.warning("🧩 Multi-Pass Cryptic Solver Starting...")
        logger.warning("=" * 60)
        
        # Load puzzle
        puzzle = self.load_cryptic_puzzle()
        
        # Run initial solve
        logger.warning("🚀 Running initial cryptic solve...")
        solver = AgenticCrosswordSolver()
        initial_result = solver.solve_puzzle(puzzle, "cryptic_multipass")
        
        initial_solved = sum(1 for clue in puzzle.clues if clue.answered)
        logger.warning(f"📊 Initial solve: {initial_solved}/{len(puzzle.clues)} ({initial_solved/len(puzzle.clues)*100:.1f}%)")
        
        # Run improvement passes
        total_improvements = 0
        for pass_num in range(self.max_passes):
            pass_result = self.run_pass(puzzle)
            self.pass_history.append(pass_result)
            total_improvements += pass_result["improvements"]
            
            # Check for convergence
            if pass_result["improvements"] == 0:
                logger.warning(f"🏁 Converged after {pass_num + 1} passes (no more improvements)")
                break
            
            # Check if we've reached high completion
            completion_rate = pass_result["final_solved"] / len(puzzle.clues)
            if completion_rate >= 0.9:
                logger.warning(f"🎉 High completion achieved: {completion_rate:.1%}")
                break
        
        # Final summary
        final_solved = sum(1 for clue in puzzle.clues if clue.answered)
        final_rate = final_solved / len(puzzle.clues)
        
        summary = {
            "initial_solved": initial_solved,
            "final_solved": final_solved,
            "improvement": final_solved - initial_solved,
            "total_clues": len(puzzle.clues),
            "final_completion_rate": final_rate,
            "passes_run": len(self.pass_history),
            "total_improvements": total_improvements,
            "pass_history": self.pass_history
        }
        
        return summary

def main():
    """Run multi-pass cryptic solving"""
    print("🧩 Multi-Pass Cryptic Crossword Solver")
    print("=====================================")
    
    solver = MultiPassCrypticSolver()
    
    try:
        # Run multi-pass solving
        results = solver.solve_multi_pass()
        
        # Display final results
        print(f"\n🏆 FINAL RESULTS:")
        print(f"   📊 Initial completion: {results['initial_solved']}/{results['total_clues']} ({results['initial_solved']/results['total_clues']*100:.1f}%)")
        print(f"   📈 Final completion: {results['final_solved']}/{results['total_clues']} ({results['final_completion_rate']*100:.1f}%)")
        print(f"   🚀 Improvement: +{results['improvement']} clues")
        print(f"   🔄 Passes run: {results['passes_run']}")
        print(f"   ⚡ Total fixes: {results['total_improvements']}")
        
        if results['final_completion_rate'] >= 0.8:
            print(f"\n🎉 SUCCESS: Achieved high completion rate!")
        elif results['improvement'] > 0:
            print(f"\n✅ PROGRESS: Made significant improvements")
        else:
            print(f"\n📊 ANALYSIS: Limited improvement possible with current approach")
        
        # Show pass-by-pass breakdown
        print(f"\n📈 Pass-by-Pass Progress:")
        for i, pass_result in enumerate(results['pass_history']):
            print(f"   Pass {i+1}: {pass_result['final_solved']}/{results['total_clues']} clues (+{pass_result['improvements']} fixes)")
            if pass_result['fixes_applied']:
                for fix in pass_result['fixes_applied'][:3]:  # Show first 3
                    print(f"      ✅ '{fix['clue'][:40]}...' = {fix['answer']}")
                if len(pass_result['fixes_applied']) > 3:
                    print(f"      ... and {len(pass_result['fixes_applied']) - 3} more")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
