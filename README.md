# Agentic Crossword Solver

A sophisticated multi-agent crossword solving system that uses agentic design patterns to solve crossword puzzles through intelligent reasoning, constraint satisfaction, and self-correction.

## 🧩 Features

- **Multi-Agent Architecture**: Specialized agents for different aspects of crossword solving
- **Review System**: Two-stage review and correction system for improved accuracy
- **Difficulty-Specific Solvers**: Adaptive solving strategies for easy, medium, hard, and cryptic puzzles
- **Intersection Validation**: Smart constraint satisfaction for overlapping clues
- **Semantic Analysis**: LLM-powered semantic verification of answers
- **Pure Solver Approach**: No ground truth knowledge - solves puzzles through reasoning alone

## 🏗️ Architecture

### Core Components

- **AgenticCrosswordSolver**: Main solver interface with difficulty-specific configurations
- **CoordinatorAgent**: Orchestrates the solving process and manages multiple specialized agents
- **ClueAgent**: Generates candidate answers using LLM reasoning
- **ConstraintAgent**: Validates solutions against crossword constraints
- **TwoStageReviewSystem**: Reviews puzzle state and attempts corrections
- **VisualizationAgent**: Tracks solving progress for analysis

### Design Patterns Used

1. **Multi-Agent System**: Specialized agents working collaboratively
2. **Reflection Pattern**: Self-assessment and correction capabilities
3. **Hierarchical Structure**: Clear separation of concerns and responsibilities
4. **Tool-Using Agents**: LLM-augmented reasoning capabilities
5. **Memory/Context Pattern**: Rich historical tracking and state management

## 🚀 Quick Start

### Setup

1. Clone and navigate to the repository:
```bash
git clone <repository-url>
cd llm-crossword
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Basic Usage

```python
from src.solver.main_solver import AgenticCrosswordSolver
from src.crossword.utils import load_puzzle

# Load a puzzle
puzzle = load_puzzle("data/medium.json")

# Create solver (review system auto-enabled for hard/cryptic)
solver = AgenticCrosswordSolver(difficulty="medium")

# Solve the puzzle
success = solver.solve(puzzle)

# Get detailed statistics
stats = solver.solve_with_stats(
    puzzle, 
    puzzle_name="test_puzzle",
    log_path="logs/solve_log.json",
    review_log_path="logs/review_report.json"
)
```

## 🧪 Testing

### Run Individual Tests

```bash
# Test easy puzzles
python test_single_easy.py

# Test medium puzzles  
python test_single_medium.py

# Test hard puzzles (with review system)
python test_single_hard.py

# Test cryptic puzzles
python test_single_cryptic.py

# Test review system specifically
python test_review_system.py
```

### Run Test Suite

```bash
pytest tests/
```

## 📊 Solver Capabilities

### Performance by Difficulty

- **Easy**: High completion rates (80-90%)
- **Medium**: Good completion rates (60-80%)
- **Hard**: Moderate completion rates (40-60%) with review system assistance
- **Cryptic**: Lower completion rates (20-40%) due to wordplay complexity

### Key Features

- **Semantic Validation**: Ensures answers make sense for clues
- **Intersection Constraints**: Validates overlapping letter requirements
- **Backtracking**: Can revise decisions when conflicts are detected
- **Pattern Matching**: Uses partial fills to constrain remaining possibilities
- **Knowledge Integration**: Leverages crossword-specific knowledge bases

## 🔍 Solving Process

1. **Clue Analysis**: Parse clues and identify types (definition, wordplay, etc.)
2. **Candidate Generation**: Generate possible answers using LLM reasoning
3. **Constraint Validation**: Check intersection requirements and grid compatibility
4. **Prioritization**: Solve high-confidence clues first to provide constraints
5. **Conflict Resolution**: Handle competing solutions through constraint satisfaction
6. **Review & Correction**: Apply two-stage review system when solver gets stuck
7. **Additional Passes**: Multiple solving iterations for improved completion

## 📁 Project Structure

```
llm-crossword/
├── src/
│   ├── crossword/           # Core crossword data structures
│   │   ├── crossword.py     # Puzzle and clue classes
│   │   ├── types.py         # Type definitions
│   │   └── utils.py         # Utility functions
│   └── solver/              # Solving agents and logic
│       ├── main_solver.py   # Main solver interface
│       ├── agents.py        # Core solving agents
│       ├── specialized_solvers.py  # Difficulty-specific configurations
│       ├── review_system.py # Two-stage review and correction system
│       └── crossword_knowledge.py # Domain knowledge base
├── data/                    # Puzzle files
│   ├── easy.json           # Easy difficulty puzzles
│   ├── medium.json         # Medium difficulty puzzles
│   ├── hard.json           # Hard difficulty puzzles
│   └── cryptic.json        # Cryptic puzzles
├── logs/                   # Solver logs and reports
├── tests/                  # Test suite
└── test_*.py              # Individual puzzle test scripts
```

## 🎯 Design Decisions & Tradeoffs

### Two-Stage Review System
- **Pro**: Systematic error detection and correction
- **Con**: Additional computational cost and complexity
- **Rationale**: Improves completion rates on difficult puzzles

### Difficulty-Specific Enabling
- **Pro**: Resource efficient, targeted where most needed
- **Con**: Inconsistent behavior across difficulties
- **Rationale**: Cost optimization while maximizing impact

### Pure Solver Approach
- **Pro**: Honest assessment of reasoning capabilities
- **Con**: Lower completion rates than ground-truth assisted approaches
- **Rationale**: Demonstrates actual solving intelligence rather than lookup

### Bounded Correction Attempts
- **Pro**: Prevents infinite loops and controls costs
- **Con**: May miss solutions requiring more iterations
- **Rationale**: Practical balance between thoroughness and efficiency

## 📈 Performance Monitoring

The solver provides detailed logging and metrics:

- **Completion Rate**: Percentage of clues successfully solved
- **Solving Time**: Time taken for initial solving pass
- **Review Activation**: Whether review system was triggered
- **Constraint Violations**: Number of intersection conflicts detected
- **Candidate Quality**: Confidence scores and semantic relevance

## 🔧 Configuration

### Solver Settings

```python
# Enable/disable review system
solver = AgenticCrosswordSolver(difficulty="hard", enable_review=True)

# Configure logging
stats = solver.solve_with_stats(
    puzzle,
    log_path="detailed_log.json",      # Solving process log
    review_log_path="review_report.json"  # Review system analysis
)
```

### Review System Settings

- **Max Corrections**: Default 3 attempts per review session
- **Semantic Threshold**: 0.6 confidence required for semantic validation
- **Trigger Conditions**: Activated when no progress in 2+ recent iterations

## 🤝 Contributing

1. Follow the established agentic design patterns
2. Ensure all candidate words are properly validated (no None values)
3. Add comprehensive logging for debugging
4. Test across all difficulty levels
5. Document design decisions and tradeoffs

## 📝 License

This challenge is proprietary and confidential. Do not share or distribute.

---

*Built with agentic design patterns for intelligent crossword solving*