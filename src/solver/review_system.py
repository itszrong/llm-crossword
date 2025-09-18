#!/usr/bin/env python3
"""
Two-Stage Review System for Crossword Solving

This module implements a review agent that analyzes failed/partial solutions
and a correction agent that attempts to fix remaining issues using insights
from the complete solving history.

AGENTIC DESIGN PATTERNS USED:
===========================================

1. MULTI-AGENT SYSTEM: ReviewAgent + CorrectionAgent + TwoStageReviewSystem
   - Multiple specialized agents working together
   - Each agent has distinct responsibilities and capabilities

2. REFLECTION PATTERN: ReviewAgent analyzes solving process and outcomes
   - Self-assessment of performance and failure modes
   - Meta-cognitive analysis of what went wrong

3. HIERARCHICAL AGENT STRUCTURE: TwoStageReviewSystem coordinates sub-agents
   - Orchestrator pattern managing workflow between agents
   - Clear separation of concerns and responsibilities

4. TOOL-USING AGENTS: Both agents use CrosswordTools for LLM interactions
   - Agents augmented with external tools and capabilities
   - Consistent interface for AI model interactions

5. MEMORY/CONTEXT PATTERN: Comprehensive logging and state tracking
   - Rich historical context from SolverLog
   - Persistent insights and decision tracking

KEY ARCHITECTURAL DECISIONS & TRADEOFFS:
=====================================

1. TWO-STAGE DESIGN (Review → Correction)
   DECISION: Separate analysis from action
   TRADEOFF: 
   ✅ PRO: Clear separation of concerns, better debugging, modular
   ❌ CON: More complexity, additional LLM calls, latency
   ALTERNATIVE: Single-stage agent doing both review and correction
   
2. TRIGGER-BASED ACTIVATION (Only when stuck)
   DECISION: Only activate review when solver makes no progress
   TRADEOFF:
   ✅ PRO: Reduces unnecessary LLM calls, cost-efficient, targeted intervention
   ❌ CON: May miss early opportunities for correction, reactive vs proactive
   ALTERNATIVE: Always-on review after each iteration
   
3. DIFFICULTY-SPECIFIC ENABLEMENT (Hard/Cryptic only)
   DECISION: Auto-enable only for difficult puzzles
   TRADEOFF:
   ✅ PRO: Resource efficient, targeted where most needed
   ❌ CON: May miss benefits on easier puzzles, less consistent behavior
   ALTERNATIVE: Enable for all difficulties
   
4. LIMITED CORRECTION ATTEMPTS (Max 3 corrections)
   DECISION: Cap the number of correction attempts
   TRADEOFF:
   ✅ PRO: Prevents infinite loops, bounds cost, faster termination
   ❌ CON: May miss solutions requiring more attempts
   ALTERNATIVE: Unlimited attempts with other stopping criteria
   
5. PRIORITIZED CLUE SELECTION (Score-based ranking)
   DECISION: Focus on highest-impact clues first
   TRADEOFF:
   ✅ PRO: Maximizes improvement per attempt, efficient resource usage
   ❌ CON: May miss cascading effects from lower-priority clues
   ALTERNATIVE: Random selection or breadth-first approach
   
6. SEMANTIC VERIFICATION THRESHOLD (Only <80% completion)
   DECISION: Skip expensive semantic checks when mostly solved
   TRADEOFF:
   ✅ PRO: Reduces LLM calls when puzzle is nearly complete
   ❌ CON: May miss semantic errors in late-stage solving
   ALTERNATIVE: Always verify or use completion-rate-based thresholds
"""

import logging
import json
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, asdict
from src.crossword.crossword import CrosswordPuzzle, Clue, Direction
from src.solver.agents import CrosswordTools, SolverLog

logger = logging.getLogger(__name__)


@dataclass
class ReviewInsight:
    """
    Individual insight from the review process
    
    DESIGN PATTERN: STRUCTURED KNOWLEDGE REPRESENTATION
    - Encapsulates findings from analysis in structured format
    - Enables consistent reasoning about different types of issues
    """
    clue_id: str
    issue_type: str  # 'intersection_conflict', 'pattern_mismatch', 'semantic_error', 'length_error'
    confidence: float
    description: str
    suggested_action: str


@dataclass
class ReviewReport:
    """
    Comprehensive review of the current puzzle state
    
    DESIGN PATTERN: AGGREGATED ANALYSIS RESULTS
    - Consolidates multiple insights into actionable intelligence
    - Provides prioritized recommendations for correction
    """
    puzzle_name: str
    total_clues: int
    solved_clues: int
    completion_rate: float
    insights: List[ReviewInsight]
    priority_clues: List[str]  # Clue IDs to focus on first
    overall_assessment: str


class ReviewAgent:
    """
    Analyzes the final state of a crossword puzzle and identifies issues
    
    DESIGN PATTERN: REFLECTION AGENT
    - Performs meta-cognitive analysis of solving process
    - Identifies failure modes and improvement opportunities
    - Uses historical context to understand what went wrong
    
    PATTERN: ANALYTICAL REASONING AGENT
    - Systematic analysis of puzzle state and solving history
    - Multi-faceted evaluation (semantic, structural, logical)
    - Evidence-based insight generation
    """
    
    def __init__(self):
        # PATTERN: TOOL-USING AGENT - Equipped with LLM capabilities
        self.tools = CrosswordTools()
    
    def analyze_puzzle_state(self, puzzle: CrosswordPuzzle, solver_log: SolverLog) -> ReviewReport:
        """
        Comprehensive analysis of the current puzzle state
        
        Args:
            puzzle: Current puzzle state
            solver_log: Complete solving history
            
        Returns:
            ReviewReport with detailed insights
        """
        logger.info("🔍 Starting comprehensive puzzle review...")
        
        insights = []
        solved_clues = sum(1 for clue in puzzle.clues if clue.answered)
        completion_rate = solved_clues / len(puzzle.clues)
        
        # Analyze unsolved clues
        unsolved_clues = [clue for clue in puzzle.clues if not clue.answered]
        
        for clue in unsolved_clues:
            clue_insights = self._analyze_unsolved_clue(clue, puzzle, solver_log)
            insights.extend(clue_insights)
        
        # OPTIMIZATION DECISION: CONDITIONAL SEMANTIC VERIFICATION
        # Only analyze solved clues for potential errors if completion rate < 80%
        # TRADEOFF: Reduces unnecessary LLM calls vs potentially missing late-stage errors
        if completion_rate < 0.8:
            solved_clues_list = [clue for clue in puzzle.clues if clue.answered]
            for clue in solved_clues_list:
                clue_insights = self._analyze_solved_clue(clue, puzzle, solver_log)
                insights.extend(clue_insights)
        
        # Identify intersection conflicts
        intersection_insights = self._analyze_intersections(puzzle)
        insights.extend(intersection_insights)
        
        # Prioritize clues for correction
        priority_clues = self._prioritize_clues(insights, puzzle)
        
        # Generate overall assessment
        overall_assessment = self._generate_assessment(puzzle, insights, completion_rate)
        
        return ReviewReport(
            puzzle_name=solver_log.puzzle_name,
            total_clues=len(puzzle.clues),
            solved_clues=solved_clues,
            completion_rate=completion_rate,
            insights=insights,
            priority_clues=priority_clues,
            overall_assessment=overall_assessment
        )
    
    def _analyze_unsolved_clue(self, clue: Clue, puzzle: CrosswordPuzzle, solver_log: SolverLog) -> List[ReviewInsight]:
        """Analyze why a specific clue remains unsolved"""
        insights = []
        clue_id = f"{clue.number}_{clue.direction.name.lower()}"
        
        # Check if there were attempts in the log
        attempts = self._get_clue_attempts(clue_id, solver_log)
        
        if not attempts:
            insights.append(ReviewInsight(
                clue_id=clue_id,
                issue_type="no_attempts",
                confidence=0.9,
                description=f"Clue '{clue.text}' was never attempted during solving",
                suggested_action="Generate new candidates and attempt solving"
            ))
        else:
            # Analyze why attempts failed
            current_chars = puzzle.get_current_clue_chars(clue)
            pattern = "".join(char or "_" for char in current_chars)
            
            # Check pattern constraints
            if '_' not in pattern:  # Fully constrained
                insights.append(ReviewInsight(
                    clue_id=clue_id,
                    issue_type="pattern_constraint",
                    confidence=0.8,
                    description=f"Clue has fixed pattern '{pattern}' from intersections but no valid word found",
                    suggested_action="Review intersection conflicts or generate pattern-specific candidates"
                ))
            
            # Check for intersection issues
            conflicting_clues = self._find_conflicting_intersections(clue, puzzle)
            if conflicting_clues:
                insights.append(ReviewInsight(
                    clue_id=clue_id,
                    issue_type="intersection_conflict",
                    confidence=0.9,
                    description=f"Intersections with {conflicting_clues} create unsolvable constraints",
                    suggested_action="Review and potentially modify intersecting answers"
                ))
        
        return insights
    
    def _analyze_solved_clue(self, clue: Clue, puzzle: CrosswordPuzzle, solver_log: SolverLog) -> List[ReviewInsight]:
        """Analyze a solved clue for potential errors using semantic analysis only"""
        insights = []
        clue_id = f"{clue.number}_{clue.direction.name.lower()}"
        
        current_chars = puzzle.get_current_clue_chars(clue)
        current_answer = ''.join(char or "_" for char in current_chars)
        
        # Use LLM to verify the answer makes semantic sense
        semantic_score = self._verify_semantic_correctness(clue, current_answer)
        
        if semantic_score < 0.6:  # Low confidence in correctness
            insights.append(ReviewInsight(
                clue_id=clue_id,
                issue_type="semantic_error",
                confidence=1.0 - semantic_score,
                description=f"Answer '{current_answer}' may not fit clue '{clue.text}' semantically (confidence: {semantic_score:.2f})",
                suggested_action="Generate alternative candidates that better match the clue meaning"
            ))
        
        return insights
    
    def _analyze_intersections(self, puzzle: CrosswordPuzzle) -> List[ReviewInsight]:
        """Analyze intersection conflicts across the puzzle"""
        insights = []
        
        for clue in puzzle.clues:
            if clue.answered:
                conflicting_clues = self._find_conflicting_intersections(clue, puzzle)
                if conflicting_clues:
                    clue_id = f"{clue.number}_{clue.direction.name.lower()}"
                    insights.append(ReviewInsight(
                        clue_id=clue_id,
                        issue_type="intersection_conflict",
                        confidence=0.95,
                        description=f"Answer conflicts with intersecting clues: {conflicting_clues}",
                        suggested_action="Review intersection logic and consider alternative answers"
                    ))
        
        return insights
    
    def _find_conflicting_intersections(self, clue: Clue, puzzle: CrosswordPuzzle) -> List[str]:
        """Find clues that have intersection conflicts with the given clue"""
        conflicts = []
        
        # This would need to implement intersection checking logic
        # For now, return empty list as placeholder
        # TODO: Implement proper intersection conflict detection
        
        return conflicts
    
    def _get_clue_attempts(self, clue_id: str, solver_log: SolverLog) -> List[Dict]:
        """Extract all attempts for a specific clue from the solver log"""
        attempts = []
        
        for iteration in solver_log.iterations:
            if isinstance(iteration, dict) and "candidates_generated" in iteration:
                for candidate_info in iteration["candidates_generated"]:
                    if candidate_info.get("clue_id") == clue_id:
                        attempts.append(candidate_info)
        
        return attempts
    
    def _verify_semantic_correctness(self, clue: Clue, answer: str) -> float:
        """Use LLM to verify if the answer semantically fits the clue"""
        try:
            prompt = f"""
Review this crossword clue and answer pair for correctness:

Clue: "{clue.text}"
Answer: "{answer}"
Length: {len(answer)} letters

Rate the semantic correctness on a scale of 0.0 to 1.0:
- 0.0-0.3: Clearly incorrect or nonsensical
- 0.4-0.6: Possible but weak connection
- 0.7-0.9: Good fit, makes sense
- 1.0: Perfect fit

Consider:
- Does the answer directly relate to the clue's meaning?
- Is this a reasonable crossword answer?
- For cryptic clues, does it follow cryptic conventions?

Respond with just a decimal number between 0.0 and 1.0.
"""
            
            response = self.tools.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert crossword solver and constructor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=50
            )
            
            content = response.choices[0].message.content.strip()
            try:
                return float(content)
            except ValueError:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error in semantic verification: {e}")
            return 0.5
    
    def _prioritize_clues(self, insights: List[ReviewInsight], puzzle: CrosswordPuzzle) -> List[str]:
        """Prioritize clues for correction based on insights"""
        # Group insights by clue_id and calculate priority scores
        clue_scores = {}
        
        for insight in insights:
            if insight.clue_id not in clue_scores:
                clue_scores[insight.clue_id] = 0
            
            # Weight by confidence and issue severity
            weight = 1.0
            if insight.issue_type == "intersection_conflict":
                weight = 2.0
            elif insight.issue_type == "semantic_error":
                weight = 1.5
            
            clue_scores[insight.clue_id] += insight.confidence * weight
        
        # Sort by score (highest first)
        prioritized = sorted(clue_scores.items(), key=lambda x: x[1], reverse=True)
        return [clue_id for clue_id, score in prioritized]
    
    def _generate_assessment(self, puzzle: CrosswordPuzzle, insights: List[ReviewInsight], completion_rate: float) -> str:
        """Generate an overall assessment of the puzzle state"""
        issue_counts = {}
        for insight in insights:
            issue_counts[insight.issue_type] = issue_counts.get(insight.issue_type, 0) + 1
        
        assessment = f"Puzzle completion: {completion_rate:.1%}\n"
        
        if completion_rate == 1.0:
            assessment += "🎉 Puzzle fully solved!"
            if insights:
                assessment += f" However, {len(insights)} potential issues detected for verification."
        elif completion_rate > 0.8:
            assessment += "🎯 Nearly complete! Focus on remaining clues and intersection conflicts."
        elif completion_rate > 0.5:
            assessment += "⚡ Good progress. Address intersection conflicts to unlock remaining clues."
        else:
            assessment += "🔧 Needs significant work. Focus on fundamental solving approach."
        
        if issue_counts:
            assessment += f"\nIssue breakdown: {dict(issue_counts)}"
        
        return assessment


class CorrectionAgent:
    """
    Uses review insights to attempt corrections on remaining clues
    
    DESIGN PATTERN: ACTION-ORIENTED AGENT
    - Takes insights from ReviewAgent and implements corrections
    - Focused on execution rather than analysis
    
    KEY DECISIONS:
    1. LIMITED ATTEMPTS: Max 3 corrections to prevent resource drain
    2. INSIGHT-DRIVEN: Uses structured insights to guide corrections
    3. CANDIDATE GENERATION: Creates targeted alternatives based on context
    
    TRADEOFF: Targeted corrections vs exhaustive search
    """
    
    def __init__(self):
        # PATTERN: TOOL-USING AGENT - Equipped with LLM capabilities for correction
        self.tools = CrosswordTools()
    
    def apply_corrections(self, puzzle: CrosswordPuzzle, review_report: ReviewReport, 
                         solver_log: SolverLog, max_corrections: int = 3) -> bool:
        """
        Apply corrections based on review insights
        
        Args:
            puzzle: Current puzzle state
            review_report: Review insights
            solver_log: Complete solving history
            max_corrections: Maximum number of correction attempts
            
        Returns:
            True if any corrections were successfully applied
        """
        logger.info("🔧 Starting correction process...")
        
        corrections_applied = 0
        
        # Focus on priority clues
        for clue_id in review_report.priority_clues[:max_corrections]:
            if corrections_applied >= max_corrections:
                break
                
            clue = self._find_clue_by_id(clue_id, puzzle)
            if not clue:
                continue
            
            # Get relevant insights for this clue
            clue_insights = [i for i in review_report.insights if i.clue_id == clue_id]
            
            if self._attempt_correction(clue, clue_insights, puzzle, solver_log):
                corrections_applied += 1
                logger.info(f"✅ Successfully corrected clue {clue_id}")
            else:
                logger.info(f"❌ Failed to correct clue {clue_id}")
        
        final_solved = sum(1 for clue in puzzle.clues if clue.answered)
        logger.info(f"🎯 Correction complete: {final_solved}/{len(puzzle.clues)} clues solved")
        
        return corrections_applied > 0
    
    def _find_clue_by_id(self, clue_id: str, puzzle: CrosswordPuzzle) -> Optional[Clue]:
        """Find a clue by its ID (number_direction format)"""
        try:
            number_str, direction_str = clue_id.split('_')
            number = int(number_str)
            direction = Direction.ACROSS if direction_str == 'across' else Direction.DOWN
            
            for clue in puzzle.clues:
                if clue.number == number and clue.direction == direction:
                    return clue
        except Exception:
            pass
        
        return None
    
    def _attempt_correction(self, clue: Clue, insights: List[ReviewInsight], 
                          puzzle: CrosswordPuzzle, solver_log: SolverLog) -> bool:
        """Attempt to correct a specific clue based on insights"""
        
        # Generate correction-focused candidates
        candidates = self._generate_correction_candidates(clue, insights, puzzle, solver_log)
        
        if not candidates:
            return False
        
        # Try each candidate
        for candidate in candidates[:3]:  # Try top 3 candidates
            if self._test_candidate(clue, candidate, puzzle):
                # Apply the candidate
                self._apply_candidate(clue, candidate, puzzle)
                return True
        
        return False
    
    def _generate_correction_candidates(self, clue: Clue, insights: List[ReviewInsight], 
                                      puzzle: CrosswordPuzzle, solver_log: SolverLog) -> List[str]:
        """Generate candidates specifically for correction"""
        
        current_chars = puzzle.get_current_clue_chars(clue)
        pattern = "".join(char or "_" for char in current_chars)
        
        # Use LLM with context from insights
        insight_context = "\n".join([f"- {i.description}" for i in insights])
        
        prompt = f"""
Given the following crossword clue and correction context, generate alternative answers:

Clue: "{clue.text}"
Length: {clue.length} letters
Current Pattern: {pattern}

Issues identified:
{insight_context}

Please generate 5 alternative answers that:
1. Fit the pattern constraints
2. Address the identified issues
3. Make semantic sense for the clue
4. Are valid crossword answers

Format each answer on a new line with just the word (uppercase).
"""
        
        try:
            response = self.tools.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert crossword constructor focused on corrections."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            content = response.choices[0].message.content
            candidates = [line.strip().upper() for line in content.split('\n') 
                         if line.strip() and line.strip().isalpha()]
            
            # Filter by length and pattern
            valid_candidates = []
            for candidate in candidates:
                if len(candidate) == clue.length and self._matches_pattern(candidate, pattern):
                    valid_candidates.append(candidate)
            
            return valid_candidates
            
        except Exception as e:
            logger.error(f"Error generating correction candidates: {e}")
            return []
    
    def _matches_pattern(self, word: str, pattern: str) -> bool:
        """Check if a word matches the given pattern"""
        if len(word) != len(pattern):
            return False
        
        for i, (char, pat) in enumerate(zip(word, pattern)):
            if pat != '_' and pat != char:
                return False
        
        return True
    
    def _test_candidate(self, clue: Clue, candidate: str, puzzle: CrosswordPuzzle) -> bool:
        """Test if a candidate would work without conflicts"""
        # This would implement proper intersection testing
        # For now, basic length check
        return len(candidate) == clue.length
    
    def _apply_candidate(self, clue: Clue, candidate: str, puzzle: CrosswordPuzzle):
        """Apply a candidate answer to the puzzle"""
        # Mark clue as answered
        clue.answered = True
        
        # Set the letters in the grid (this would need proper grid updating logic)
        # For now, just mark as answered
        logger.info(f"Applied candidate '{candidate}' to clue {clue.number} {clue.direction.name}")


class TwoStageReviewSystem:
    """Orchestrates the two-stage review and correction process"""
    
    def __init__(self):
        self.review_agent = ReviewAgent()
        self.correction_agent = CorrectionAgent()
    
    def review_and_correct(self, puzzle: CrosswordPuzzle, solver_log: SolverLog, 
                          max_corrections: int = 3) -> Tuple[bool, ReviewReport]:
        """
        Complete two-stage review and correction process
        
        Args:
            puzzle: Current puzzle state
            solver_log: Complete solving history
            max_corrections: Maximum correction attempts
            
        Returns:
            Tuple of (corrections_applied, review_report)
        """
        logger.info("🎭 Starting two-stage review system...")
        
        # Stage 1: Review
        review_report = self.review_agent.analyze_puzzle_state(puzzle, solver_log)
        
        logger.info("📋 Review Report:")
        logger.info(f"  Completion: {review_report.completion_rate:.1%}")
        logger.info(f"  Issues found: {len(review_report.insights)}")
        logger.info(f"  Priority clues: {len(review_report.priority_clues)}")
        logger.info(f"  Assessment: {review_report.overall_assessment}")
        
        # Stage 2: Correction (only if not fully solved)
        corrections_applied = False
        if review_report.completion_rate < 1.0:
            corrections_applied = self.correction_agent.apply_corrections(
                puzzle, review_report, solver_log, max_corrections
            )
        
        return corrections_applied, review_report
    
    def save_review_report(self, review_report: ReviewReport, output_path: str):
        """Save the review report to a JSON file"""
        import os
        
        # Convert to dictionary for JSON serialization
        report_dict = asdict(review_report)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Review report saved to {output_path}")


"""
COMPREHENSIVE DECISION FRAMEWORK SUMMARY:
=======================================

PERFORMANCE vs COST TRADEOFFS:
1. Trigger-based activation: Only when stuck (cost-efficient)
2. Limited correction attempts: Max 3 (bounded cost)
3. Conditional semantic verification: Skip when >80% complete
4. Difficulty-specific enablement: Hard/Cryptic only

ACCURACY vs EFFICIENCY TRADEOFFS:
1. Two-stage design: Separate analysis/action (better accuracy, more complex)
2. Prioritized clue selection: High-impact first (efficiency over completeness)
3. Pattern-guided corrections: Use constraints to narrow search space

ROBUSTNESS vs SIMPLICITY TRADEOFFS:
1. Multi-agent architecture: More robust but complex coordination
2. Structured insight representation: Systematic but overhead
3. Comprehensive logging: Full observability but storage cost

DESIGN PATTERN ADVANTAGES:
✅ Reflection Pattern: Self-improvement capability
✅ Multi-Agent System: Specialized expertise, parallel processing
✅ Hierarchical Structure: Clear responsibilities, coordinated action
✅ Tool-Using Agents: Leverages LLM capabilities effectively
✅ Memory/Context: Rich historical awareness

POTENTIAL INTERVIEW QUESTIONS & RESPONSES:

Q: "Why two agents instead of one?"
A: "Separation of analysis and action. ReviewAgent focuses on understanding 
   what went wrong, CorrectionAgent focuses on fixing it. This enables 
   specialized reasoning and better debugging."

Q: "How do you prevent infinite correction loops?"
A: "Multiple safeguards: max 3 correction attempts, progress tracking,
   trigger conditions (only when stuck), and bounded search space."

Q: "What about computational costs?"
A: "Designed for efficiency: trigger-based activation, conditional analysis,
   difficulty-specific enablement, limited attempts. Only activates when
   needed most (hard puzzles that are stuck)."

Q: "How do you handle scaling to larger puzzles?"
A: "Prioritized analysis focuses on highest-impact clues first. Structured
   insights enable efficient processing. Limited scope prevents exponential
   growth."

Q: "What are the failure modes?"
A: "Could miss early intervention opportunities (reactive), may not solve
   issues requiring >3 attempts, semantic verification threshold could miss
   late-stage errors. Trade-offs are documented and measurable."
"""
