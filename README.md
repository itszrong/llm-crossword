# LLM Crossword Solver

## Setup Instructions

1. Clone this repository:
```bash
git clone https://github.com/lcardno10/llm-crossword.git
cd llm-crossword
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your environment variables:
```bash
cp .env.example .env #Copy the example env
# Edit the .env and replace with the values provided during the interview
```

5. Verify setup:

Run the main script:
```bash
python main.py
```

Or run the notebook cells in scratchpad.ipynb:

6. Start coding!

## Project Structure

```
llm_crossword/
├── src/                 # Source code
├── tests/              # Test suite
└── data/               # Data files
```

## Testing

Run the test suite:
```bash
pytest
```

## License

This challenge is proprietary and confidential. Do not share or distribute.
