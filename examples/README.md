# Examples Directory

This directory contains example scripts demonstrating how to use the agentic crossword solver.

## 📁 File Organization

### Core Examples
- `basic_usage.py` - Simple example of how to use the solver
- `advanced_usage.py` - Advanced features and configuration options

### Test Scripts (Single Puzzles)
- `test_single_easy.py` - Test easy difficulty puzzles
- `test_single_medium.py` - Test medium difficulty puzzles  
- `test_single_hard.py` - Test hard difficulty puzzles with review system
- `test_single_cryptic.py` - Test cryptic puzzles

### System Tests
- `test_agentic_solver.py` - Test the complete agentic solving system
- `test_review_system.py` - Test the two-stage review and correction system

### Utilities
- `final_completion.py` - Manual completion utility (for reference only)

## 🚀 Running Examples

### Basic Usage
```bash
cd examples
python basic_usage.py
```

### Test Specific Difficulty
```bash
python test_single_medium.py
```

### Test Review System
```bash
python test_review_system.py
```

## 📝 Notes

- All test scripts use the pure solver approach (no ground truth)
- Review system is automatically enabled for hard and cryptic difficulties
- Logs are saved to the `../logs/` directory
- Examples demonstrate real-world usage patterns
