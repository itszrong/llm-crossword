#!/usr/bin/env python3
"""
Specialized Crossword Solvers for Different Difficulty Levels

This module provides specialized solver configurations that modify the base
CoordinatorAgent with different strategies and thresholds appropriate for 
different puzzle difficulties.
"""

import logging
from typing import Dict, Any
from src.solver.types import DifficultyLevel

logger = logging.getLogger(__name__)


class DifficultyConfigurator:
    """Factory class for configuring solvers based on difficulty level"""
    
    @staticmethod
    def get_solver_config(difficulty: str) -> Dict[str, Any]:
        """
        Get solver configuration parameters for a difficulty level
        
        Args:
            difficulty: The difficulty level (easy, medium, hard, cryptic)
            
        Returns:
            Dictionary with solver configuration parameters
        """
        difficulty = difficulty.lower()
        
        if difficulty == "easy":
            return {
                "max_iterations": 3,
                "backtrack_enabled": False,
                "thinking_depth": 1,
                "use_async_solving": False,
            "max_candidates_per_clue": 2,
            "confidence_threshold": 0.6,
            "review_threshold": 0.4,
            "prefer_high_confidence": True,
                "parallel_solving": False,
                "prompt_style": "simple"
            }
        
        elif difficulty == "medium":
            return {
                "max_iterations": 5,
                "backtrack_enabled": True,
                "thinking_depth": 2,
                "use_async_solving": True,
            "max_candidates_per_clue": 3,
            "confidence_threshold": 0.5,
            "review_threshold": 0.3,
            "prefer_high_confidence": True,
            "parallel_solving": True,
                "max_backtracks": 2,
                "prompt_style": "enhanced"
            }
        
        elif difficulty == "hard":
            return {
                "max_iterations": 8,
                "backtrack_enabled": True,
                "thinking_depth": 3,
                "use_async_solving": True,
            "max_candidates_per_clue": 5,
            "confidence_threshold": 0.4,
                "review_threshold": 0.2,  # More lenient to avoid filtering OEDIPUSREX
            "prefer_high_confidence": False,
            "parallel_solving": True,
            "max_backtracks": 5,
            "deep_reasoning": True,
                "constraint_propagation": True,
                "prompt_style": "advanced"
            }
        
        elif difficulty == "cryptic":
            return {
                "max_iterations": 12,
                "backtrack_enabled": True,
                "thinking_depth": 4,
                "use_async_solving": True,
            "max_candidates_per_clue": 8,
            "confidence_threshold": 0.3,
                "review_threshold": 0.1,  # Very lenient for cryptic clues
            "prefer_high_confidence": False,
            "parallel_solving": True,
            "max_backtracks": 10,
            "deep_reasoning": True,
            "constraint_propagation": True,
            "wordplay_analysis": True,
                "anagram_detection": True,
                "prompt_style": "cryptic"
            }
        
        else:
            logger.warning(f"Unknown difficulty '{difficulty}', using medium defaults")
            return DifficultyConfigurator.get_solver_config("medium")
    
    @staticmethod
    def get_prompt_additions(difficulty: str, clue_type: str = "definition") -> str:
        """
        Get additional prompt text based on difficulty level
        
        Args:
            difficulty: The difficulty level
            clue_type: The type of clue (definition, cryptic, etc.)
            
        Returns:
            Additional prompt text to append
        """
        difficulty = difficulty.lower()
        
        if difficulty == "easy":
            return """
For easy puzzles:
- Think of the most obvious, common answer
- Prefer simple, everyday words
- Avoid obscure references
"""
        
        elif difficulty == "medium":
            return """
For medium difficulty:
- Consider multiple meanings of words
- Think about synonyms and related terms
- Consider common crossword conventions
- Look for wordplay patterns
"""
        
        elif difficulty == "hard":
            return """
For hard difficulty:
- Consider wordplay and double meanings
- Think about less obvious connections
- Consider abbreviations and acronyms
- Look for misdirection in the clue
- Think step by step through possible interpretations
- Consider literary, historical, or specialized references
"""
        
        elif difficulty == "cryptic" or clue_type == "cryptic":
            return """
CRYPTIC CLUE ANALYSIS:
Cryptic clues have TWO parts that must BOTH work:
1. DEFINITION (straight definition, usually at start or end)
2. WORDPLAY (anagram, hidden word, charade, reversal, etc.)

STEP-BY-STEP APPROACH:
1. Identify which part is the definition (often 1-2 words)
2. Identify wordplay indicators and mechanism
3. Work through the wordplay letter by letter
4. Verify the final word matches BOTH definition and wordplay

ENHANCED CRYPTIC INDICATORS:
- Anagrams: mixed, confused, broken, twisted, wild, mad, upset, reform, etc.
- Hidden words: in, within, inside, part of, contains, holds, etc.
- Reversals: back, returns, up (in down clues), reverse, opposite, etc.
- Charades: after, before, with, around, about, following, etc.
- Homophones: sounds like, heard, spoken, audibly, etc.
- Deletions: without, missing, loses, drops, etc.

CRYPTIC SOLVING RULES:
- Every letter must be accounted for in the wordplay
- The definition should be a straightforward synonym
- Look for "connector words" like "of", "in", "for" that link parts
- Numbers in parentheses show word lengths (e.g., "7" or "3,4")

Provide multiple possibilities with DETAILED wordplay breakdown:
ANSWER: [word] | CONFIDENCE: [0.0-1.0] | DEFINITION: [which part] | WORDPLAY: [step-by-step mechanism]
ALT1: [word] | CONFIDENCE: [0.0-1.0] | DEFINITION: [which part] | WORDPLAY: [step-by-step mechanism]  
ALT2: [word] | CONFIDENCE: [0.0-1.0] | DEFINITION: [which part] | WORDPLAY: [step-by-step mechanism]
"""
        
        return ""


class SpecializedConstraintAgent:
    """Constraint agent with difficulty-specific behavior"""
    
    def __init__(self, base_agent, difficulty: str):
        self.base_agent = base_agent
        self.difficulty = difficulty.lower()
    
    def validate_solution(self, puzzle, clue, candidate, state=None):
        """Validate solution using the base agent's validation logic"""
        return self.base_agent.validate_solution(puzzle, clue, candidate, state)
    
    def resolve_conflicts(self, puzzle, state):
        """Resolve conflicts with difficulty-specific logic"""
        
        if self.difficulty == "cryptic":
            return self._cryptic_conflict_resolution(puzzle, state)
        elif self.difficulty == "hard":
            return self._hard_conflict_resolution(puzzle, state)
        else:
            # Use base implementation for easy/medium
            return self.base_agent.resolve_conflicts(puzzle, state)
    
    def _cryptic_conflict_resolution(self, puzzle, state):
        """Cryptic-specific conflict resolution prioritizing wordplay quality"""
        from src.solver.agents import get_clue_id
        
        if not state.candidates:
            return []
        
        priority_items = []
        
        for clue_id, candidates in state.candidates.items():
            if not candidates:
                continue
            
            # Find the clue object
            clue = None
            for c in puzzle.clues:
                if get_clue_id(c) == clue_id:
                    clue = c
                    break
            
            if not clue:
                continue
            
            for candidate in candidates:
                # Skip if validation fails
                is_valid, _ = self.base_agent.validate_solution(puzzle, clue, candidate, state)
                if not is_valid:
                    continue
                
                # Calculate priority with cryptic-specific boosts
                constraint_factor = self.base_agent._calculate_constraint_factor(puzzle, clue)
                
                # CRYPTIC-SPECIFIC BOOSTS
                wordplay_boost = 1.0
                reasoning = candidate.reasoning.lower()
                
                # Boost for quality wordplay explanations
                if any(mechanism in reasoning for mechanism in ["anagram", "hidden", "reversal"]):
                    wordplay_boost = 2.5
                elif "definition" in reasoning and "wordplay" in reasoning:
                    wordplay_boost = 2.0
                elif candidate.confidence >= 0.8:
                    wordplay_boost = 1.5
                
                # Extra boost for detailed explanations
                if any(detail in reasoning for detail in ["step by step", "letter by letter", "anagram of"]):
                    wordplay_boost *= 1.3
                
                priority = candidate.confidence * constraint_factor * wordplay_boost
                priority_items.append((priority, clue, candidate))
        
        # Sort by priority and apply greedily
        priority_items.sort(key=lambda x: x[0], reverse=True)
        
        solutions = []
        applied_clues = set()
        
        for priority, clue, candidate in priority_items:
            clue_id = get_clue_id(clue)
            if clue_id in applied_clues:
                continue
            
            solutions.append((clue, candidate))
            applied_clues.add(clue_id)
        
        return solutions
    
    def _hard_conflict_resolution(self, puzzle, state):
        """Hard-specific conflict resolution with more sophisticated selection"""
        # For now, use base implementation but could add sophisticated CSP solving
        return self.base_agent.resolve_conflicts(puzzle, state)


class SpecializedReviewAgent:
    """Review agent with difficulty-specific behavior"""
    
    def __init__(self, base_agent, difficulty: str):
        self.base_agent = base_agent
        self.difficulty = difficulty.lower()
    
    def review_solution(self, clue, candidate, puzzle):
        """Review solution with difficulty-specific criteria"""
        
        if self.difficulty == "cryptic":
            return self._cryptic_review(clue, candidate, puzzle)
        elif self.difficulty == "easy":
            return self._easy_review(clue, candidate, puzzle)
        else:
            # Use base implementation for medium/hard
            return self.base_agent.review_solution(clue, candidate, puzzle)
    
    def _easy_review(self, clue, candidate, puzzle):
        """Simplified review for easy puzzles"""
        # Basic checks only
        if len(candidate.word) != clue.length:
            return 0.0
        
        context_score = self.base_agent._check_context_consistency(clue, candidate, puzzle)
        return context_score * 0.8 + 0.2  # Give benefit of doubt
    
    def _cryptic_review(self, clue, candidate, puzzle):
        """Cryptic-specific review with wordplay validation"""
        base_score = self.base_agent.review_solution(clue, candidate, puzzle)
        
        # Check if wordplay explanation makes sense
        wordplay_score = self._validate_cryptic_wordplay(candidate)
        
        # Boost score for detailed wordplay explanations
        explanation_bonus = self._evaluate_explanation_quality(candidate)
        
        final_score = (base_score + wordplay_score + explanation_bonus) / 3
        return min(final_score, 1.0)
    
    def _validate_cryptic_wordplay(self, candidate):
        """Validate cryptic wordplay quality"""
        reasoning = candidate.reasoning.lower()
        
        # High confidence mechanisms
        if any(mechanism in reasoning for mechanism in ["anagram", "hidden", "reversal"]):
            return 0.9
        elif any(mechanism in reasoning for mechanism in ["charade", "homophone", "deletion"]):
            return 0.8
        elif "definition" in reasoning and "wordplay" in reasoning:
            return 0.7
        else:
            return 0.5
    
    def _evaluate_explanation_quality(self, candidate):
        """Evaluate explanation quality"""
        reasoning = candidate.reasoning.lower()
        
        quality_indicators = [
            "step by step", "letter by letter", "definition:", "wordplay:",
            "anagram of", "hidden in", "sounds like", "reversed", 
            "around", "contains", "plus", "minus"
        ]
        
        explanation_score = sum(1 for indicator in quality_indicators if indicator in reasoning)
        return min(explanation_score * 0.1, 0.3)  # Max bonus of 0.3


def create_specialized_solver(base_coordinator, difficulty: str):
    """
    Configure a base CoordinatorAgent for a specific difficulty level
    
    Args:
        base_coordinator: The base CoordinatorAgent instance
        difficulty: The difficulty level (easy, medium, hard, cryptic)
        
    Returns:
        The configured coordinator
    """
    from src.solver.agents import CrosswordTools, ClueAgent, ConstraintAgent, ReviewAgent
    
    config = DifficultyConfigurator.get_solver_config(difficulty)
    
    # Apply configuration to the coordinator
    for key, value in config.items():
        if hasattr(base_coordinator, key):
            setattr(base_coordinator, key, value)
    
    # Store difficulty for reference
    base_coordinator.difficulty = difficulty
    
    # Recreate tools and agents with difficulty awareness
    base_coordinator.tools = CrosswordTools(difficulty)
    base_coordinator.clue_agent = ClueAgent(base_coordinator.tools)
    base_coordinator.constraint_agent = ConstraintAgent(base_coordinator.tools)
    base_coordinator.review_agent = ReviewAgent(base_coordinator.tools)
    
    # Wrap agents with specialized behavior if needed
    if difficulty in ["cryptic", "hard"]:
        base_coordinator.constraint_agent = SpecializedConstraintAgent(
            base_coordinator.constraint_agent, difficulty
        )
    
    if difficulty in ["cryptic", "easy"]:
        base_coordinator.review_agent = SpecializedReviewAgent(
            base_coordinator.review_agent, difficulty
        )
    
    logger.info(f"Configured solver for {difficulty} difficulty")
    return base_coordinator