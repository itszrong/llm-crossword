"""
Utility functions for loading and manipulating crossword puzzles.

You should not need to worry about this module. We use it to load CrosswordPuzzle instances from .json and .puz files.
"""
import puz
import json 

from src.crossword.crossword import CrosswordPuzzle
from src.crossword.exceptions import InvalidClueError
from src.crossword.types import Clue, Direction

def load_puzzle(file_path: str) -> CrosswordPuzzle:
    """Load a puzzle from a .puz or .json file."""

    # Load puzzle from JSON file
    if file_path.endswith(".json"):
        with open(file_path, "r") as f:
            puzzle_data = json.load(f)

        # Initialize puzzle from JSON data
        puzzle = CrosswordPuzzle.model_validate(puzzle_data)

        # Update initial grid with puzzle file content

        return puzzle
    
    # Load puzzle from .puz file
    elif file_path.endswith(".puz"):

        puzzle_file = puz.read(file_path)
        
        # Initialize puzzle with dimensions
        puzzle = CrosswordPuzzle(
            width=puzzle_file.width,
            height=puzzle_file.height
        )

        numbering = puzzle_file.clue_numbering()

        # Add across clues with answers
        for across in numbering.across:
            row = across["cell"] // puzzle.width
            col = across["cell"] % puzzle.width
            answer = ''.join(puzzle_file.solution[across["cell"] + i] 
                            for i in range(across["len"]))
            try:
                puzzle.add_clue(Clue(
                    number=len(puzzle.clues) + 1,
                    text=across["clue"],
                    direction=Direction.ACROSS,
                    length=across["len"],
                    row=row,
                    col=col,
                    answer=answer
                ))
            except InvalidClueError as e:
                raise InvalidClueError(f"Invalid across clue: {e}")

        # Add down clues with answers
        for down in numbering.down:
            row = down["cell"] // puzzle.width
            col = down["cell"] % puzzle.width
            answer = ''.join(puzzle_file.solution[down["cell"] + i * puzzle.width] 
                            for i in range(down["len"]))
            try:
                puzzle.add_clue(Clue(
                    number=len(puzzle.clues) + 1,
                    text=down["clue"],
                    direction=Direction.DOWN,
                    length=down["len"],
                    row=row,
                    col=col,
                    answer=answer
                ))
            except InvalidClueError as e:
                raise InvalidClueError(f"Invalid down clue: {e}")

        return puzzle
    
    else:
        raise ValueError(f"Unsupported file format: {file_path.split('.')[-1]}")
