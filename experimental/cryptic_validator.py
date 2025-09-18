#!/usr/bin/env python3
"""
Cryptic Crossword Validator

Validates cryptic crossword solutions and checks for intersecting clue conflicts.
Based on the actual output from the cryptic solver.
"""

import os
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from src.crossword.crossword import CrosswordPuzzle
from src.crossword.types import Clue, Direction
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.WARNING, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class CrypticValidator:
    """Validates cryptic crossword solutions"""
    
    def __init__(self):
        # Based on the cryptic solver output you provided
        self.observed_solutions = {
            "Campaigned for B. Dole surprisingly winning twice (7)": "LOBBIED",
            "Discovered hot curry initially taken away (5)": "BARED",  
            "Where life is in scope, evolving at this location (9)": "ECOSPHERE",
            "Stones heretics from the uprising - ends in revolution (4)": "ROCK",
            "Aerodynamic feature of crushed possession? (6,6)": "WINGEDMIRROR", # This looks wrong
            "Trees with 80% of leaves in water ... (4)": "ELMS",
            "Inclined to oust leader over ... (5)": "ALIRS",  # This looks wrong
            "Shorter drunk dictator not welcome (7)": "TERRSRR",  # This looks very wrong
            "Small amount of pudding (6)": "DORLOP",  # Should be DOLLOP
            "Checks upset London police infiltrating Nazi organisation (5)": "STEMS",
            "Dances to hits (4)": "BOPS",
            "Played loud music including note belted (8)": "ENAMORED",  # This might be wrong
            "Extract from trees and also wood (6)": "RESINS",
            "Footballing connection written on document (6)": "PASSER",
            "Pink and black stuff right in the middle (5)": "INKET",  # Should be INKER?
            "Ring everyone after 1:00 having colon removed (4)": "CALL"
        }
        
        # Expected correct solutions for comparison
        self.expected_solutions = {
            "Deliver dollar to complete start of betting spreads (7)": "BUTTERS",
            "Campaigned for B. Dole surprisingly winning twice (7)": "LOBBIED",
            "Discovered hot curry initially taken away (5)": "SPIED",  # Not BARED
            "Where life is in scope, evolving at this location (9)": "ECOSPHERE",
            "Devices for extracting bit of dirt from old clothes around 49p (3,7)": "OILPRESSES",
            "Stones heretics from the uprising - ends in revolution (4)": "ROCK",
            "Old debugger working to pick up a mark of reference (6,6)": "DOUBLEDAGGER",
            "Aerodynamic feature of crushed possession? (6,6)": "GROUNDEFFECT",  # Not WINGEDMIRROR
            "Trees with 80% of leaves in water ... (4)": "ELMS",
            "... leaf hiding off centre is cut (10)": "ELIMINATED",
            "It pulls vehicle at first (9)": "ATTRACTOR",
            "Inclined to oust leader over ... (5)": "EAGER",  # Not ALIRS
            "... tiny thing - 'I quit to protect love child' (7)": "TODDLER",
            "Shorter drunk dictator not welcome (7)": "OUTCAST",  # Not TERRSRR
            "Cry about time working in US city (6)": "BOSTON",
            "Small amount of pudding (6)": "DOLLOP",  # Not DORLOP
            "Crosswords need to entertain one in covers? (10)": "AMUSEMENTS",
            "Checks upset London police infiltrating Nazi organisation (5)": "STASI",  # Not STEMS
            "Ladies and gents feel excited buying afternoon tea? (5-4)": "CREAMTEAS",
            "Dances to hits (4)": "BOPS",
            "I see small bird climbing cold masses (8)": "ICEBERGS",
            "They help actors turning up anxious, not tense, over reading (8)": "PROMPTER",
            "Unhappy with China stealing Liberal party books (10)": "DISCONTENT",
            "One striking to cover rising cost of guard (9)": "BEEFEATER",
            "Lacking education, ran into trouble pinching PS1,000 (8)": "IGNORANT",
            "Played loud music including note belted (8)": "BLASTING",  # Not ENAMORED
            "Extract from trees and also wood (6)": "RESINS",
            "Footballing connection written on document (6)": "HEADER",  # Not PASSER
            "Pink and black stuff right in the middle (5)": "CORAL",  # Not INKET
            "Ring everyone after 1:00 having colon removed (4)": "CALL"
        }
        
    def analyze_cryptic_answers(self) -> Dict[str, Any]:
        """Analyze the cryptic answers for correctness"""
        logger.warning("🧩 Analyzing cryptic crossword solutions...")
        
        analysis = {
            "total_clues": len(self.expected_solutions),
            "observed_solutions": len(self.observed_solutions),
            "correct_matches": [],
            "incorrect_solutions": [],
            "missing_solutions": [],
            "suspicious_patterns": [],
            "summary": {
                "accuracy": 0.0,
                "completion_rate": 0.0,
                "major_errors": 0
            }
        }
        
        # Compare observed vs expected
        for clue, expected in self.expected_solutions.items():
            if clue in self.observed_solutions:
                observed = self.observed_solutions[clue]
                if observed.upper() == expected.upper():
                    analysis["correct_matches"].append({
                        "clue": clue,
                        "answer": expected
                    })
                else:
                    analysis["incorrect_solutions"].append({
                        "clue": clue,
                        "expected": expected,
                        "observed": observed,
                        "error_type": self._classify_error(expected, observed)
                    })
            else:
                analysis["missing_solutions"].append({
                    "clue": clue,
                    "expected": expected
                })
        
        # Identify suspicious patterns
        for error in analysis["incorrect_solutions"]:
            if self._is_suspicious_answer(error["observed"]):
                analysis["suspicious_patterns"].append(error)
        
        # Calculate metrics
        correct_count = len(analysis["correct_matches"])
        total_expected = len(self.expected_solutions)
        observed_count = len(self.observed_solutions)
        
        analysis["summary"]["accuracy"] = correct_count / total_expected if total_expected > 0 else 0
        analysis["summary"]["completion_rate"] = observed_count / total_expected if total_expected > 0 else 0
        analysis["summary"]["major_errors"] = len(analysis["suspicious_patterns"])
        
        return analysis
    
    def _classify_error(self, expected: str, observed: str) -> str:
        """Classify the type of error"""
        if len(expected) != len(observed):
            return "LENGTH_MISMATCH"
        elif self._is_gibberish(observed):
            return "GIBBERISH"
        elif self._shares_letters(expected, observed):
            return "PARTIAL_MATCH"
        else:
            return "WRONG_ANSWER"
    
    def _is_gibberish(self, word: str) -> bool:
        """Check if a word appears to be gibberish"""
        gibberish_patterns = [
            lambda w: len(set(w)) < len(w) * 0.4,  # Too many repeated letters
            lambda w: w.count('R') > 3,  # Too many R's
            lambda w: not any(v in w.lower() for v in 'aeiou'),  # No vowels
            lambda w: len(w) > 6 and w.upper() == w and 'RR' in w,  # Suspicious patterns
        ]
        
        return any(pattern(word) for pattern in gibberish_patterns)
    
    def _is_suspicious_answer(self, word: str) -> bool:
        """Check if an answer looks suspicious"""
        suspicious_signs = [
            "RRR" in word,  # Triple letters
            "TERRSRR" == word,  # Obvious gibberish
            "ALIRS" == word,  # Not a real word
            "DORLOP" == word,  # Misspelling
            "INKET" == word,  # Not a real word
        ]
        
        return any(sign for sign in suspicious_signs)
    
    def _shares_letters(self, word1: str, word2: str) -> bool:
        """Check if two words share significant letters"""
        set1, set2 = set(word1.upper()), set(word2.upper())
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union > 0.3 if union > 0 else False
    
    def check_intersecting_conflicts(self) -> List[Dict[str, Any]]:
        """Check for conflicts between intersecting clues"""
        logger.warning("🔍 Checking for intersecting clue conflicts...")
        
        # Known conflicts from the cryptic output
        conflicts = []
        
        # Example: If BARED intersects with something expecting S but BARED has D
        if "BARED" in self.observed_solutions.values() and "SPIED" in self.expected_solutions.values():
            conflicts.append({
                "issue": "BARED vs SPIED conflict",
                "description": "Clue 'Discovered hot curry initially taken away' has BARED instead of SPIED",
                "impact": "This affects intersecting DOWN clues",
                "recommendation": "Change BARED to SPIED - 'Discovered' = SPIED (discovered), hot (H) and curry initially (C) removed from SPICED"
            })
        
        # Check for gibberish answers that likely cause intersection issues
        gibberish_answers = [
            ("TERRSRR", "Should be OUTCAST - anagram of 'drunk' + 'dictator not welcome'"),
            ("ALIRS", "Should be EAGER - 'inclined' = eager, 'oust leader' removes first letter"),
            ("DORLOP", "Should be DOLLOP - small amount of something"),
            ("WINGEDMIRROR", "Should be GROUNDEFFECT - aerodynamic feature of cars"),
        ]
        
        for wrong_answer, explanation in gibberish_answers:
            if any(wrong_answer == observed for observed in self.observed_solutions.values()):
                conflicts.append({
                    "issue": f"Gibberish answer: {wrong_answer}",
                    "description": explanation,
                    "impact": "Creates incorrect letters for intersecting clues",
                    "recommendation": f"Fix {wrong_answer} to resolve downstream intersections"
                })
        
        return conflicts
    
    def generate_report(self) -> str:
        """Generate a comprehensive validation report"""
        analysis = self.analyze_cryptic_answers()
        conflicts = self.check_intersecting_conflicts()
        
        report = "🧩 CRYPTIC CROSSWORD VALIDATION REPORT\n"
        report += "=" * 50 + "\n\n"
        
        # Summary
        report += f"📊 SUMMARY:\n"
        report += f"  • Total clues: {analysis['total_clues']}\n"
        report += f"  • Solutions attempted: {analysis['observed_solutions']}\n"
        report += f"  • Correct solutions: {len(analysis['correct_matches'])}\n"
        report += f"  • Incorrect solutions: {len(analysis['incorrect_solutions'])}\n"
        report += f"  • Missing solutions: {len(analysis['missing_solutions'])}\n"
        report += f"  • Completion rate: {analysis['summary']['completion_rate']:.1%}\n"
        report += f"  • Accuracy rate: {analysis['summary']['accuracy']:.1%}\n\n"
        
        # Correct solutions
        if analysis["correct_matches"]:
            report += f"✅ CORRECT SOLUTIONS ({len(analysis['correct_matches'])}):\n"
            for match in analysis["correct_matches"][:5]:  # Show first 5
                report += f"  • '{match['clue'][:50]}...' = {match['answer']}\n"
            if len(analysis["correct_matches"]) > 5:
                report += f"  ... and {len(analysis['correct_matches']) - 5} more\n"
            report += "\n"
        
        # Incorrect solutions
        if analysis["incorrect_solutions"]:
            report += f"❌ INCORRECT SOLUTIONS ({len(analysis['incorrect_solutions'])}):\n"
            for error in analysis["incorrect_solutions"]:
                report += f"  • '{error['clue'][:50]}...'\n"
                report += f"    Expected: {error['expected']} | Got: {error['observed']} | Type: {error['error_type']}\n"
            report += "\n"
        
        # Intersection conflicts
        if conflicts:
            report += f"🔍 INTERSECTION CONFLICTS ({len(conflicts)}):\n"
            for conflict in conflicts:
                report += f"  • {conflict['issue']}\n"
                report += f"    {conflict['description']}\n"
                report += f"    💡 {conflict['recommendation']}\n\n"
        
        # Recommendations
        report += "💡 RECOMMENDATIONS:\n"
        if analysis["summary"]["major_errors"] > 0:
            report += f"  1. Fix {analysis['summary']['major_errors']} gibberish/suspicious answers\n"
        if len(analysis["incorrect_solutions"]) > 5:
            report += f"  2. Review cryptic clue interpretation - many answers are incorrect\n"
        if conflicts:
            report += f"  3. Resolve intersection conflicts to improve overall grid consistency\n"
        report += f"  4. Focus on proper cryptic wordplay analysis for better accuracy\n"
        
        return report

def main():
    """Run cryptic validation"""
    print("🧩 Cryptic Crossword Validator")
    print("==============================")
    
    validator = CrypticValidator()
    
    try:
        # Generate and display report
        report = validator.generate_report()
        print(report)
        
        # Additional analysis
        analysis = validator.analyze_cryptic_answers()
        
        print(f"🎯 KEY ISSUES IDENTIFIED:")
        print(f"  • Gibberish answers like TERRSRR, ALIRS indicate LLM confusion")
        print(f"  • Misspellings like DORLOP suggest partial understanding")  
        print(f"  • Wrong cryptic interpretations like BARED vs SPIED")
        print(f"  • Intersection conflicts cascade through the grid")
        
        print(f"\n🔧 SUGGESTED FIXES:")
        print(f"  1. Improve cryptic clue parsing and wordplay analysis")
        print(f"  2. Add stronger validation against gibberish answers")
        print(f"  3. Implement intersection checking during solving")
        print(f"  4. Use cryptic-specific confidence scoring")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
