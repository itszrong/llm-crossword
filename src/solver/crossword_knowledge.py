#!/usr/bin/env python3
"""
Crossword Knowledge Base for Common Categories

This module provides comprehensive knowledge for common crossword categories
to help prevent the solver from missing obvious answers like OMEGA for "Greek letter".
"""

from typing import Dict, List, Set

class CrosswordKnowledge:
    """Comprehensive knowledge base for crossword solving"""
    
    def __init__(self):
        self.knowledge_base = {
            # Greek letters by length
            "greek_letters": {
                3: ["ETA", "CHI", "PHI", "PSI", "RHO", "TAU"],
                4: ["BETA", "ZETA", "IOTA"],
                5: ["ALPHA", "DELTA", "GAMMA", "KAPPA", "OMEGA", "SIGMA", "THETA"],
                6: ["LAMBDA", "UPSILON"],
                7: ["EPSILON", "OMICRON"],
                8: ["UPSILON"]
            },
            
            # Roman numerals
            "roman_numerals": {
                1: ["I"],
                2: ["II", "IV", "VI", "IX", "XI"],
                3: ["III", "VII", "VIII", "XII", "XIV", "XVI"],
                4: ["XIII", "XVII", "XVIII", "XXIV"],
                5: ["XXXII"]
            },
            
            # Musical terms
            "musical_terms": {
                3: ["KEY", "BAR", "BOW"],
                4: ["NOTE", "CLEF", "TUNE", "SONG", "BEAT"],
                5: ["CHORD", "TEMPO", "SCALE", "SHARP", "FLAT"],
                6: ["MELODY", "RHYTHM"],
                7: ["HARMONY", "CADENCE"]
            },
            
            # Animals by length
            "animals": {
                3: ["CAT", "DOG", "BAT", "COW", "PIG", "RAT", "APE", "OWL", "ELK"],
                4: ["BEAR", "LION", "DEER", "GOAT", "DUCK", "FISH", "FROG", "CRAB", "SEAL"],
                5: ["HORSE", "MOUSE", "SHEEP", "SNAKE", "WHALE", "SHARK", "EAGLE", "TIGER"],
                6: ["RABBIT", "DONKEY", "SALMON", "TURKEY"],
                7: ["CHICKEN", "PENGUIN", "DOLPHIN", "GIRAFFE"],
                8: ["ELEPHANT", "KANGAROO"]
            },
            
            # Colors
            "colors": {
                3: ["RED", "TAN", "DUN"],
                4: ["BLUE", "GOLD", "GRAY", "PINK", "TEAL", "AQUA"],
                5: ["BLACK", "WHITE", "GREEN", "BROWN", "AMBER", "CORAL"],
                6: ["ORANGE", "PURPLE", "YELLOW", "SILVER", "MAROON"],
                7: ["CRIMSON", "EMERALD", "MAGENTA"],
                8: ["TURQUOISE"]
            },
            
            # Countries and capitals
            "countries": {
                4: ["PERU", "CHAD", "IRAQ", "IRAN", "CUBA"],
                5: ["CHINA", "EGYPT", "INDIA", "ITALY", "JAPAN", "SPAIN"],
                6: ["FRANCE", "GREECE", "POLAND", "RUSSIA", "TURKEY"],
                7: ["GERMANY", "HUNGARY", "IRELAND"],
                8: ["THAILAND"]
            },
            
            # Body parts
            "body_parts": {
                3: ["ARM", "LEG", "EYE", "EAR", "TOE", "JAW"],
                4: ["HEAD", "HAND", "FOOT", "NECK", "BACK", "KNEE", "CHIN"],
                5: ["HEART", "BRAIN", "CHEST", "ELBOW", "ANKLE", "THUMB"],
                6: ["FINGER", "SHOULDER"],
                7: ["STOMACH"]
            },
            
            # Literary works and authors
            "literature": {
                4: ["POEM", "TALE", "EPIC", "SAGA"],
                5: ["NOVEL", "DRAMA", "FABLE", "PROSE"],
                6: ["SONNET", "BALLAD"],
                7: ["TRAGEDY", "COMEDY"],
                8: ["THRILLER"]
            },
            
            # Tools and implements
            "tools": {
                3: ["AXE", "SAW", "AWL"],
                4: ["NAIL", "RAKE", "FILE"],
                5: ["SPADE", "DRILL", "LATHE"],
                6: ["HAMMER", "WRENCH", "PLIERS"],
                7: ["SCREWDRIVER"]
            }
        }
    
    def get_category_suggestions(self, clue_text: str, length: int) -> List[str]:
        """Get category-specific suggestions based on clue text and length"""
        clue_lower = clue_text.lower()
        suggestions = []
        
        # Greek letter detection
        if any(phrase in clue_lower for phrase in ["greek letter", "letter from greece", "fraternity letter"]):
            suggestions.extend(self.knowledge_base.get("greek_letters", {}).get(length, []))
        
        # Roman numeral detection  
        elif any(phrase in clue_lower for phrase in ["roman numeral", "roman number", "latin number"]):
            suggestions.extend(self.knowledge_base.get("roman_numerals", {}).get(length, []))
        
        # Musical terms
        elif any(phrase in clue_lower for phrase in ["musical", "music", "note", "chord", "song", "tune", "melody"]):
            suggestions.extend(self.knowledge_base.get("musical_terms", {}).get(length, []))
        
        # Animals
        elif any(phrase in clue_lower for phrase in ["animal", "beast", "creature", "pet", "mammal", "bird", "fish"]):
            suggestions.extend(self.knowledge_base.get("animals", {}).get(length, []))
        
        # Colors
        elif any(phrase in clue_lower for phrase in ["color", "colour", "hue", "shade", "tint"]):
            suggestions.extend(self.knowledge_base.get("colors", {}).get(length, []))
        
        # Countries
        elif any(phrase in clue_lower for phrase in ["country", "nation", "state", "republic"]):
            suggestions.extend(self.knowledge_base.get("countries", {}).get(length, []))
        
        # Body parts
        elif any(phrase in clue_lower for phrase in ["body part", "anatomy", "limb", "organ"]):
            suggestions.extend(self.knowledge_base.get("body_parts", {}).get(length, []))
        
        # Literature
        elif any(phrase in clue_lower for phrase in ["literary", "book", "author", "poet", "novel", "poem"]):
            suggestions.extend(self.knowledge_base.get("literature", {}).get(length, []))
        
        # Tools
        elif any(phrase in clue_lower for phrase in ["tool", "implement", "device", "instrument"]):
            suggestions.extend(self.knowledge_base.get("tools", {}).get(length, []))
        
        return suggestions
    
    def get_comprehensive_greek_letters(self, length: int) -> List[str]:
        """Get all Greek letters of specified length"""
        return self.knowledge_base.get("greek_letters", {}).get(length, [])
    
    def validate_category_answer(self, clue_text: str, answer: str) -> bool:
        """Validate if an answer makes sense for the category indicated in the clue"""
        clue_lower = clue_text.lower()
        answer_upper = answer.upper()
        
        # Check if the answer appears in any relevant category
        if "greek letter" in clue_lower:
            all_greek_letters = set()
            for letters in self.knowledge_base.get("greek_letters", {}).values():
                all_greek_letters.update(letters)
            return answer_upper in all_greek_letters
        
        # Add more validations as needed
        return True
