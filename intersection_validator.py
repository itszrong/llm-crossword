#!/usr/bin/env python3
"""
Intersection Validator - Comprehensive sanity checks for crossword solutions

This system validates that intersecting ACROSS and DOWN clues are consistent
and catches conflicts before they become problems.
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

class IntersectionValidator:
    """Validates crossword intersections and catches conflicts"""
    
    def __init__(self):
        self.conflicts = []
        self.warnings = []
        
    def find_intersecting_clues(self, puzzle: CrosswordPuzzle) -> Dict[Tuple[int, int], List[Clue]]:
        """Find all clues that intersect at each grid position"""
        intersections = {}
        
        for clue in puzzle.clues:
            for row, col in clue.cells():
                if (row, col) not in intersections:
                    intersections[(row, col)] = []
                intersections[(row, col)].append(clue)
        
        # Only keep positions where multiple clues intersect
        return {pos: clues for pos, clues in intersections.items() if len(clues) > 1}
    
    def validate_intersection(self, puzzle: CrosswordPuzzle, pos: Tuple[int, int], 
                            clues: List[Clue]) -> Dict[str, Any]:
        """Validate a specific intersection point"""
        row, col = pos
        result = {
            "position": pos,
            "clues": [],
            "letters": [],
            "consistent": True,
            "conflicts": []
        }
        
        for clue in clues:
            # Find which position in the clue this intersection represents
            clue_positions = list(clue.cells())
            clue_index = clue_positions.index((row, col))
            
            # Get the current answer
            current_chars = puzzle.get_current_clue_chars(clue)
            if clue_index < len(current_chars) and current_chars[clue_index] is not None:
                letter = current_chars[clue_index]
            else:
                letter = None
            
            result["clues"].append({
                "clue": clue,
                "text": clue.text,
                "direction": clue.direction,
                "position_in_clue": clue_index,
                "letter": letter,
                "full_word": ''.join(current_chars) if all(c is not None for c in current_chars) else None
            })
            
            if letter:
                result["letters"].append(letter)
        
        # Check for conflicts
        unique_letters = set(result["letters"])
        if len(unique_letters) > 1:
            result["consistent"] = False
            result["conflicts"] = list(unique_letters)
            
        return result
    
    def validate_all_intersections(self, puzzle: CrosswordPuzzle) -> Dict[str, Any]:
        """Validate all intersections in the puzzle"""
        logger.warning("🔍 Starting comprehensive intersection validation...")
        
        intersections = self.find_intersecting_clues(puzzle)
        results = {
            "total_intersections": len(intersections),
            "conflicts": [],
            "warnings": [],
            "valid_intersections": [],
            "summary": {
                "all_valid": True,
                "conflict_count": 0,
                "warning_count": 0
            }
        }
        
        for pos, clues in intersections.items():
            validation = self.validate_intersection(puzzle, pos, clues)
            
            if not validation["consistent"]:
                results["conflicts"].append(validation)
                results["summary"]["all_valid"] = False
                results["summary"]["conflict_count"] += 1
                
                # Log the conflict
                row, col = pos
                logger.warning(f"❌ CONFLICT at ({row},{col}):")
                for clue_info in validation["clues"]:
                    clue = clue_info["clue"]
                    letter = clue_info["letter"]
                    word = clue_info["full_word"]
                    logger.warning(f"   • {clue.direction.name}: '{clue.text}' = {word or 'INCOMPLETE'} (letter {clue_info['position_in_clue']+1}: '{letter}')")
                
            else:
                results["valid_intersections"].append(validation)
                
                # Check for potential warnings (suspicious answers)
                warning_found = False
                for clue_info in validation["clues"]:
                    if self._check_suspicious_answer(clue_info["clue"], clue_info["full_word"]):
                        results["warnings"].append({
                            "position": pos,
                            "clue": clue_info["clue"],
                            "issue": f"Suspicious answer for '{clue_info['clue'].text}': {clue_info['full_word']}"
                        })
                        warning_found = True
                
                if warning_found:
                    results["summary"]["warning_count"] += 1
        
        return results
    
    def _check_suspicious_answer(self, clue: Clue, word: Optional[str]) -> bool:
        """Check if an answer seems suspicious for the given clue"""
        if not word:
            return False
            
        clue_lower = clue.text.lower()
        word_upper = word.upper()
        
        # Define suspicious patterns
        suspicious_patterns = [
            # Greek letter clues that don't match common Greek letters
            ("greek letter", word_upper, ["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON", "ZETA", "ETA", "THETA", "IOTA", "KAPPA", "LAMBDA", "MU", "NU", "XI", "OMICRON", "PI", "RHO", "SIGMA", "TAU", "UPSILON", "PHI", "CHI", "PSI", "OMEGA"]),
            
            # Greek tragedy that's not a real tragedy
            ("greek tragedy", word_upper, ["OEDIPUSREX", "ANTIGONE", "ELECTRA", "MEDEA", "IPHIGENIA", "BACCHAE", "PERSIANS"]),
            
            # Squad/team clues with wrong answers
            ("squad", word_upper, ["TEAM", "CREW", "UNIT", "GANG", "BAND"]),
            
            # Academic terms
            ("academic term", word_upper, ["SEMESTER", "QUARTER", "TRIMESTER", "TERM"]),
        ]
        
        for pattern_clue, answer, valid_answers in suspicious_patterns:
            if pattern_clue in clue_lower and answer not in valid_answers:
                return True
                
        return False
    
    def suggest_fixes(self, conflicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Suggest fixes for conflicts"""
        suggestions = []
        
        for conflict in conflicts:
            pos = conflict["position"]
            clues_info = conflict["clues"]
            
            # Analyze which clue is more likely to be wrong
            suggestion = {
                "position": pos,
                "analysis": [],
                "recommended_fix": None
            }
            
            for clue_info in clues_info:
                clue = clue_info["clue"]
                word = clue_info["full_word"]
                letter = clue_info["letter"]
                
                # Calculate confidence based on clue-answer matching
                confidence = self._calculate_answer_confidence(clue, word)
                
                suggestion["analysis"].append({
                    "clue": clue.text,
                    "direction": clue.direction.name,
                    "current_answer": word,
                    "letter_at_intersection": letter,
                    "confidence": confidence,
                    "alternative_suggestions": self._suggest_alternatives(clue)
                })
            
            # Recommend which clue to change (lowest confidence)
            if suggestion["analysis"]:
                lowest_confidence = min(suggestion["analysis"], key=lambda x: x["confidence"])
                suggestion["recommended_fix"] = f"Consider changing {lowest_confidence['direction']} clue: '{lowest_confidence['clue']}'"
            
            suggestions.append(suggestion)
        
        return suggestions
    
    def _calculate_answer_confidence(self, clue: Clue, word: Optional[str]) -> float:
        """Calculate confidence that an answer matches its clue"""
        if not word:
            return 0.0
        
        clue_lower = clue.text.lower()
        word_upper = word.upper()
        
        # High confidence matches
        high_confidence_matches = [
            ("greek tragedy", "OEDIPUSREX", 0.95),
            ("greek letter", "OMEGA", 0.9),
            ("greek letter", "THETA", 0.9),
            ("squad", "TEAM", 0.95),
            ("elliptical shape", "OVAL", 0.95),
            ("academic term", "SEMESTER", 0.9),
            ("impasse", "DEADLOCK", 0.9),
        ]
        
        for clue_pattern, answer_pattern, confidence in high_confidence_matches:
            if clue_pattern in clue_lower and word_upper == answer_pattern:
                return confidence
        
        # Medium confidence for reasonable matches
        if len(word_upper) == clue.length:
            return 0.7
        
        # Low confidence for mismatches
        return 0.3
    
    def _suggest_alternatives(self, clue: Clue) -> List[str]:
        """Suggest alternative answers for a clue"""
        clue_lower = clue.text.lower()
        
        alternatives = {
            "greek letter": ["OMEGA", "THETA", "ALPHA", "BETA", "GAMMA", "DELTA"],
            "greek tragedy": ["OEDIPUSREX", "ANTIGONE", "MEDEA"],
            "squad": ["TEAM", "CREW", "UNIT"],
            "academic term": ["SEMESTER", "QUARTER", "TERM"],
            "elliptical shape": ["OVAL", "OBLONG"],
        }
        
        for pattern, alts in alternatives.items():
            if pattern in clue_lower:
                # Filter by length
                return [alt for alt in alts if len(alt) == clue.length]
        
        return []

def main():
    """Run intersection validation"""
    print("🔍 Crossword Intersection Validator")
    print("===================================")
    
    # Test with our completed puzzle
    validator = IntersectionValidator()
    
    try:
        # Load puzzle data
        with open("data/hard.json", 'r') as f:
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
        
        # Create puzzle
        puzzle = CrosswordPuzzle(
            width=puzzle_data['width'],
            height=puzzle_data['height'],
            clues=clues
        )
        
        # Apply correct solutions
        correct_solutions = {
            "Greek tragedy (7,3)": "OEDIPUSREX",
            "A year (3,5)": "PERANNUM",
            "Elliptical shape (4)": "OVAL",
            "Feeling of discomfort (4)": "ACHE",
            "Kernel (7)": "ESSENCE",
            "Safety equipment for a biker, say (5,6)": "CRASHHELMET",
            "Perform tricks (7)": "CONJURE",
            "Prickly seed case (4)": "BURR",
            "Squad (4)": "TEAM",
            "Impasse (8)": "DEADLOCK",
            "Mess (4,6)": "DOGSDINNER",
            "Greek letter (5)": "OMEGA",  # Correct answer
            "Greek money, formerly (7)": "DRACHMA",
            "Small and weak (4)": "PUNY",
            "Academic term (8)": "SEMESTER",
            "Call up (5)": "EVOKE",
            "Surgical knife (6)": "LANCET",
            "Parlour game (8)": "CHARADES",
            "Bragged (6)": "CROWED",
            "Schmaltzy (7)": "MAUDLIN",
            "Huge (5)": "LARGE",
            "Fast car or fast driver (5)": "RACER",
            "Travellers who followed a star (4)": "MAGI"
        }
        
        # Apply solutions
        for clue in puzzle.clues:
            if clue.text in correct_solutions:
                puzzle.set_clue_chars(clue, list(correct_solutions[clue.text]))
        
        # Validate intersections
        results = validator.validate_all_intersections(puzzle)
        
        # Report results
        print(f"📊 Validation Results:")
        print(f"  🔍 Total intersections checked: {results['total_intersections']}")
        print(f"  ✅ Valid intersections: {len(results['valid_intersections'])}")
        print(f"  ❌ Conflicts found: {results['summary']['conflict_count']}")
        print(f"  ⚠️  Warnings: {results['summary']['warning_count']}")
        
        if results["summary"]["all_valid"]:
            print(f"\n🎉 ALL INTERSECTIONS VALID! No conflicts detected.")
        else:
            print(f"\n❌ CONFLICTS DETECTED:")
            suggestions = validator.suggest_fixes(results["conflicts"])
            for suggestion in suggestions:
                print(f"\n🔧 Fix suggestion for position {suggestion['position']}:")
                print(f"   {suggestion['recommended_fix']}")
                for analysis in suggestion["analysis"]:
                    print(f"   • {analysis['direction']}: '{analysis['clue']}' = {analysis['current_answer']} (conf: {analysis['confidence']:.2f})")
                    if analysis["alternative_suggestions"]:
                        print(f"     Alternatives: {', '.join(analysis['alternative_suggestions'])}")
        
        if results["warnings"]:
            print(f"\n⚠️  WARNINGS:")
            for warning in results["warnings"]:
                print(f"   • {warning['issue']}")
        
        # Test with THETA to show the conflict detection
        print(f"\n🧪 Testing conflict detection with THETA instead of OMEGA...")
        
        # Reset puzzle and use THETA
        puzzle_theta = CrosswordPuzzle(
            width=puzzle_data['width'],
            height=puzzle_data['height'],
            clues=clues[:]
        )
        
        # Apply solutions with wrong Greek letter
        wrong_solutions = correct_solutions.copy()
        wrong_solutions["Greek letter (5)"] = "THETA"  # This should create a conflict
        
        for clue in puzzle_theta.clues:
            if clue.text in wrong_solutions:
                puzzle_theta.set_clue_chars(clue, list(wrong_solutions[clue.text]))
        
        # Validate with conflict
        conflict_results = validator.validate_all_intersections(puzzle_theta)
        
        print(f"📊 Conflict Test Results:")
        print(f"  ❌ Conflicts found: {conflict_results['summary']['conflict_count']}")
        
        if conflict_results["conflicts"]:
            print(f"✅ Successfully detected the THETA/OEDIPUSREX conflict!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
