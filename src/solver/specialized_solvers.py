#!/usr/bin/env python3
"""
Specialized Crossword Solvers for Different Difficulty Levels

This module provides specialized solver implementations that inherit from the base
CoordinatorAgent but have different tools, strategies, and sophistication levels
appropriate for different puzzle difficulties.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum

from src.solver.types import DifficultyLevel
from src.crossword.crossword import CrosswordPuzzle
from src.crossword.types import Clue

logger = logging.getLogger(__name__)


@dataclass
class SolverMove:
    """Represents a move in the solving process for undo functionality"""
    clue: Clue
    candidate: ClueCandidate
    timestamp: str
    grid_state_before: Dict[str, Any]


class BaseDifficultyCoordinator:
    """Base class for difficulty-specific coordinators"""
    
    def __init__(self, difficulty: DifficultyLevel):
        super().__init__()
        self.difficulty = difficulty
        self.move_history: List[SolverMove] = []
        self.backtrack_enabled = False
        self.thinking_depth = 1
        
    @abstractmethod
    def get_specialized_tools(self) -> CrosswordTools:
        """Get tools specialized for this difficulty level"""
        pass
    
    @abstractmethod
    def get_solving_strategy(self) -> Dict[str, Any]:
        """Get solving strategy parameters for this difficulty"""
        pass
    
    def can_undo(self) -> bool:
        """Check if undo is available"""
        return self.backtrack_enabled and len(self.move_history) > 0
    
    def undo_last_move(self, puzzle: CrosswordPuzzle) -> bool:
        """Undo the last move"""
        if not self.can_undo():
            return False
        
        last_move = self.move_history.pop()
        
        # Restore grid state
        try:
            # Clear the clue
            puzzle.clear_clue_chars(last_move.clue)
            logger.info(f"Undid move: '{last_move.clue.text}' = {last_move.candidate.word}")
            return True
        except Exception as e:
            logger.error(f"Failed to undo move: {e}")
            return False


class EasyCrosswordCoordinator(BaseDifficultyCoordinator):
    """Solver optimized for easy puzzles - fast and straightforward"""
    
    def __init__(self):
        super().__init__(DifficultyLevel.EASY)
        self.max_iterations = 3  # Quick solving
        self.backtrack_enabled = False  # No undos needed
        self.thinking_depth = 1  # Simple reasoning
        
        # Use simplified tools
        self.tools = self.get_specialized_tools()
        self.clue_agent = SimpleClueAgent(self.tools)
        self.constraint_agent = FastConstraintAgent(self.tools)
        self.review_agent = BasicReviewAgent(self.tools)
        self.use_async_solving = False  # Keep it simple for easy puzzles
        # VisualizationAgent is inherited from parent
    
    def get_specialized_tools(self) -> CrosswordTools:
        """Simple tools for easy puzzles"""
        return SimpleCrosswordTools()
    
    def get_solving_strategy(self) -> Dict[str, Any]:
        return {
            "max_candidates_per_clue": 2,
            "confidence_threshold": 0.6,
            "review_threshold": 0.4,
            "prefer_high_confidence": True,
            "parallel_solving": False
        }


class MediumCrosswordCoordinator(BaseDifficultyCoordinator):
    """Solver for medium difficulty - balanced approach"""
    
    def __init__(self):
        super().__init__(DifficultyLevel.MEDIUM)
        self.max_iterations = 5
        self.backtrack_enabled = True  # Limited backtracking
        self.thinking_depth = 2
        
        self.tools = self.get_specialized_tools()
        self.clue_agent = StandardClueAgent(self.tools)
        self.constraint_agent = ConstraintAgent(self.tools)
        self.review_agent = ReviewAgent(self.tools)
        self.use_async_solving = True  # Enable async for medium+
    
    def get_specialized_tools(self) -> CrosswordTools:
        """Standard tools with some enhancements"""
        return EnhancedCrosswordTools()
    
    def get_solving_strategy(self) -> Dict[str, Any]:
        return {
            "max_candidates_per_clue": 3,
            "confidence_threshold": 0.5,
            "review_threshold": 0.3,
            "prefer_high_confidence": True,
            "parallel_solving": True,
            "max_backtracks": 2
        }


class HardCrosswordCoordinator(BaseDifficultyCoordinator):
    """Solver for hard puzzles - sophisticated reasoning"""
    
    def __init__(self):
        super().__init__(DifficultyLevel.HARD)
        self.max_iterations = 8
        self.backtrack_enabled = True
        self.thinking_depth = 3
        
        self.tools = self.get_specialized_tools()
        self.clue_agent = AdvancedClueAgent(self.tools)
        self.constraint_agent = AdvancedConstraintAgent(self.tools)
        self.review_agent = ThoroughReviewAgent(self.tools)
        self.reasoning_agent = ReasoningAgent(self.tools)
        self.use_async_solving = True  # Definitely use async for hard puzzles
    
    def get_specialized_tools(self) -> CrosswordTools:
        """Advanced tools with deep reasoning"""
        return AdvancedCrosswordTools()
    
    def get_solving_strategy(self) -> Dict[str, Any]:
        return {
            "max_candidates_per_clue": 5,
            "confidence_threshold": 0.4,
            "review_threshold": 0.2,
            "prefer_high_confidence": False,
            "parallel_solving": True,
            "max_backtracks": 5,
            "deep_reasoning": True,
            "constraint_propagation": True
        }


class CrypticCrosswordCoordinator(BaseDifficultyCoordinator):
    """Solver for cryptic puzzles - specialized for wordplay"""
    
    def __init__(self):
        super().__init__(DifficultyLevel.CRYPTIC)
        self.max_iterations = 12
        self.backtrack_enabled = True
        self.thinking_depth = 4
        
        self.tools = self.get_specialized_tools()
        self.clue_agent = CrypticClueAgent(self.tools)
        self.constraint_agent = AdvancedConstraintAgent(self.tools)
        self.review_agent = CrypticReviewAgent(self.tools)
        self.wordplay_agent = WordplayAgent(self.tools)
        self.reasoning_agent = ReasoningAgent(self.tools)
        self.use_async_solving = True  # Cryptic puzzles benefit most from async
    
    def get_specialized_tools(self) -> CrosswordTools:
        """Cryptic-specific tools"""
        return CrypticCrosswordTools()
    
    def get_solving_strategy(self) -> Dict[str, Any]:
        return {
            "max_candidates_per_clue": 8,
            "confidence_threshold": 0.3,
            "review_threshold": 0.1,
            "prefer_high_confidence": False,
            "parallel_solving": True,
            "max_backtracks": 10,
            "deep_reasoning": True,
            "constraint_propagation": True,
            "wordplay_analysis": True,
            "anagram_detection": True
        }


# Specialized Tool Classes

class SimpleCrosswordTools(CrosswordTools):
    """Simplified tools for easy puzzles"""
    
    def _build_clue_prompt(self, clue: Clue, context: str, clue_type: str) -> str:
        # Simplified prompt for easy puzzles
        return f"""
Solve this simple crossword clue:
Clue: "{clue.text}"
Length: {clue.length} letters

Think of the most obvious, common answer. Keep it simple.

Provide your best answer:
ANSWER: [word] | CONFIDENCE: HIGH/MEDIUM/LOW | REASONING: [brief explanation]
"""


class EnhancedCrosswordTools(CrosswordTools):
    """Enhanced tools for medium difficulty"""
    
    def _build_clue_prompt(self, clue: Clue, context: str, clue_type: str) -> str:
        base_prompt = super()._build_clue_prompt(clue, context, clue_type)
        return base_prompt + """

For medium difficulty:
- Consider multiple meanings of words
- Think about synonyms and related terms
- Consider common crossword conventions
"""


class AdvancedCrosswordTools(CrosswordTools):
    """Advanced tools for hard puzzles"""
    
    def _build_clue_prompt(self, clue: Clue, context: str, clue_type: str) -> str:
        base_prompt = super()._build_clue_prompt(clue, context, clue_type)
        return base_prompt + """

For hard difficulty:
- Consider wordplay and double meanings
- Think about less obvious connections
- Consider abbreviations and acronyms
- Look for misdirection in the clue
- Think step by step through possible interpretations
"""


class CrypticCrosswordTools(CrosswordTools):
    """Specialized tools for cryptic puzzles"""
    
    def _classify_clue_type(self, clue_text: str) -> str:
        # Always treat as cryptic
        return "cryptic"
    
    def _build_clue_prompt(self, clue: Clue, context: str, clue_type: str) -> str:
        return f"""
Solve this cryptic crossword clue:
Clue: "{clue.text}"
Length: {clue.length} letters

Cryptic clues have two parts:
1. Definition (usually at the beginning or end)
2. Wordplay (anagram, hidden word, charade, etc.)

Analyze this clue:
- Identify the definition part
- Identify the wordplay mechanism
- Work through the wordplay step by step
- Verify the answer satisfies both parts

Common cryptic indicators:
- Anagrams: mixed, confused, broken, twisted, etc.
- Hidden words: in, within, inside, part of, etc.
- Reversals: back, returns, up (in down clues), etc.
- Charades: after, before, with, etc.

Provide multiple possibilities:
ANSWER: [word] | CONFIDENCE: [level] | DEFINITION: [part] | WORDPLAY: [mechanism and explanation]
ALT1: [word] | CONFIDENCE: [level] | DEFINITION: [part] | WORDPLAY: [mechanism and explanation]
ALT2: [word] | CONFIDENCE: [level] | DEFINITION: [part] | WORDPLAY: [mechanism and explanation]
"""


# Specialized Agent Classes

class SimpleClueAgent(ClueAgent):
    """Simple clue agent for easy puzzles"""
    
    def __init__(self, tools: CrosswordTools):
        super().__init__(tools)
        self.max_retries = 1  # Quick attempts only


class StandardClueAgent(ClueAgent):
    """Standard clue agent for medium puzzles"""
    
    def __init__(self, tools: CrosswordTools):
        super().__init__(tools)
        self.max_retries = 2


class AdvancedClueAgent(ClueAgent):
    """Advanced clue agent for hard puzzles"""
    
    def __init__(self, tools: CrosswordTools):
        super().__init__(tools)
        self.max_retries = 3


class CrypticClueAgent(ClueAgent):
    """Cryptic clue agent with wordplay expertise"""
    
    def __init__(self, tools: CrosswordTools):
        super().__init__(tools)
        self.max_retries = 4


class FastConstraintAgent(ConstraintAgent):
    """Fast constraint checking for easy puzzles"""
    
    def resolve_conflicts(self, puzzle: CrosswordPuzzle, state: SolverState) -> List[Tuple[Clue, ClueCandidate]]:
        # Simple greedy approach - take highest confidence
        solution = []
        for clue_num, candidates in state.candidates.items():
            if candidates:
                clue = next(c for c in puzzle.clues if c.number == clue_num)
                best_candidate = candidates[0]  # Already sorted by confidence
                if self.validate_solution(puzzle, clue, best_candidate):
                    solution.append((clue, best_candidate))
        return solution


class AdvancedConstraintAgent(ConstraintAgent):
    """Advanced constraint agent with backtracking"""
    
    def resolve_conflicts(self, puzzle: CrosswordPuzzle, state: SolverState) -> List[Tuple[Clue, ClueCandidate]]:
        # Use constraint satisfaction with backtracking
        return self._solve_with_backtracking(puzzle, state)
    
    def _solve_with_backtracking(self, puzzle: CrosswordPuzzle, state: SolverState) -> List[Tuple[Clue, ClueCandidate]]:
        # Implement backtracking constraint satisfaction
        # This is a simplified version - would need full CSP implementation
        return super().resolve_conflicts(puzzle, state)


class BasicReviewAgent(ReviewAgent):
    """Basic review for easy puzzles"""
    
    def review_solution(self, clue: Clue, candidate: ClueCandidate, puzzle: CrosswordPuzzle) -> float:
        # Simplified review - mainly check constraints
        if len(candidate.word) != clue.length:
            return 0.0
        
        context_score = self._check_context_consistency(clue, candidate, puzzle)
        return context_score * 0.8 + 0.2  # Give benefit of doubt


class ThoroughReviewAgent(ReviewAgent):
    """Thorough review for hard puzzles"""
    
    def review_solution(self, clue: Clue, candidate: ClueCandidate, puzzle: CrosswordPuzzle) -> float:
        # Enhanced review with multiple checks
        score = super().review_solution(clue, candidate, puzzle)
        
        # Additional checks for hard puzzles
        if score > 0.5:
            # Double-check with secondary reasoning
            secondary_score = self._secondary_validation(clue, candidate)
            score = (score + secondary_score) / 2
        
        return score
    
    def _secondary_validation(self, clue: Clue, candidate: ClueCandidate) -> float:
        # Implement secondary validation logic
        return 0.8  # Placeholder


class CrypticReviewAgent(ReviewAgent):
    """Cryptic-specific review agent"""
    
    def review_solution(self, clue: Clue, candidate: ClueCandidate, puzzle: CrosswordPuzzle) -> float:
        # Cryptic-specific validation
        base_score = super().review_solution(clue, candidate, puzzle)
        
        # Check if wordplay explanation makes sense
        wordplay_score = self._validate_cryptic_wordplay(clue, candidate)
        
        return (base_score + wordplay_score) / 2
    
    def _validate_cryptic_wordplay(self, clue: Clue, candidate: ClueCandidate) -> float:
        # Validate cryptic wordplay - simplified implementation
        if "anagram" in candidate.reasoning.lower():
            return 0.9  # High confidence for explained anagrams
        elif "hidden" in candidate.reasoning.lower():
            return 0.8
        else:
            return 0.6  # Lower confidence for other mechanisms


# Additional Specialized Agents

class ReasoningAgent:
    """Agent for deep reasoning and step-by-step analysis"""
    
    def __init__(self, tools: CrosswordTools):
        self.tools = tools
    
    def deep_analyze(self, clue: Clue, candidates: List[ClueCandidate], puzzle: CrosswordPuzzle) -> List[ClueCandidate]:
        """Perform deep analysis on candidates"""
        # Implement deep reasoning logic
        return candidates


class WordplayAgent:
    """Agent specialized in cryptic wordplay analysis"""
    
    def __init__(self, tools: CrosswordTools):
        self.tools = tools
    
    def analyze_wordplay(self, clue: Clue) -> Dict[str, Any]:
        """Analyze cryptic wordplay mechanisms"""
        # Implement wordplay analysis
        return {
            "mechanism": "unknown",
            "definition_part": "",
            "wordplay_part": "",
            "confidence": 0.5
        }


# Factory function to create appropriate solver

def create_solver_for_difficulty(difficulty: str) -> BaseDifficultyCoordinator:
    """Factory function to create the appropriate solver for a difficulty level"""
    difficulty_map = {
        "easy": EasyCrosswordCoordinator,
        "medium": MediumCrosswordCoordinator,
        "hard": HardCrosswordCoordinator,
        "cryptic": CrypticCrosswordCoordinator
    }
    
    solver_class = difficulty_map.get(difficulty.lower())
    if not solver_class:
        raise ValueError(f"Unknown difficulty level: {difficulty}")
    
    return solver_class()
