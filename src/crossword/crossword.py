from typing import List, Optional
from pydantic import BaseModel, Field

from .types import Direction, Cell, Clue, Grid
from .exceptions import InvalidGridError, InvalidClueError

class CrosswordPuzzle(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    clues: List[Clue] = Field(default_factory=list)
    grid_history: List[Grid] = Field(default_factory=list, alias="grid_history")
    clue_history: List[int] = Field(default_factory=list, alias="clue_history")

    def __init__(self, **data):
        super().__init__(**data)
        if not self.grid_history:
            # Initialize empty grid with cells
            cells = [[Cell(row=r, col=c, value=None) 
                     for c in range(self.width)]
                    for r in range(self.height)]
            
            initial_grid = Grid(
                width=self.width,
                height=self.height,
                cells=cells
            )
            self.grid_history.append(initial_grid)

    @property
    def current_grid(self) -> Grid:
        return self.grid_history[-1]

    def add_clue(self, clue: Clue) -> None:
        """Add a new clue to the puzzle"""
        
        # check for unqiue id
        if any(c.number == clue.number for c in self.clues):
            raise InvalidClueError(f"Clue with number {clue.number} already exists")

        # check for valid position                     
        if not self._validate_clue_chars_position(clue):
            raise InvalidClueError(f"Clue {clue.number} position is invalid")
                              
        self.clues.append(clue)

    def _validate_clue_chars_position(self, clue: Clue) -> bool:
        """Check if the clue position is valid"""
        if not (0 <= clue.row < self.height and 0 <= clue.col < self.width):
            return False
        if clue.direction == Direction.ACROSS:
            return clue.col + clue.length <= self.width
        return clue.row + clue.length <= self.height

    def get_current_clue_chars(self, clue: Clue) -> List[Optional[str]]:
        """Get current characters in the grid for a given clue"""     
        if clue not in self.clues:
            raise InvalidClueError("Clue not found in puzzle")
        
        return [self.current_grid.cells[row][col].value 
                for row, col in clue.cells()]

    def set_clue_chars(self, clue: Clue, chars: List[str]) -> None:
        """Fill in characters for a given clue"""
        if len(chars) != clue.length:
            raise InvalidClueError(f"Expected {clue.length} characters, got {len(chars)}")

        # Make all chars uppercase
        chars = [char.upper() for char in chars]

        # Create new grid based on current state
        new_grid = self.current_grid.model_copy(deep=True)
        
        for (row, col), char in zip(clue.cells(), chars):
            new_grid.cells[row][col].value = char

        self.grid_history.append(new_grid)
        
        clue.answered = True
        self.clue_history.append(self.clues.index(clue))

    def reveal_clue_answer(self, clue: Clue) -> None:
        """Reveal the answer for a specific clue"""
        if not clue.answer:
            raise InvalidClueError("No answer available for this clue")
            
        self.set_clue_chars(clue, list(clue.answer))

    def reveal_all(self) -> None:
        """Reveal all answers in the puzzle"""
        for clue in self.clues:
            if not clue.answered and clue.answer:
                self.reveal_clue_answer(clue)

    def get_clues_overlapping_with_cell(self, row: int, col: int) -> List[Clue]:
        """Get all clues that overlap with a specific cell"""
        return [
            clue for clue in self.clues if row in clue.cells() and col in clue.cells()
        ]

    def validate_clue_chars(self, clue: Clue) -> bool:
        """Check if the current entry for a clue is correct"""
        return self.get_current_clue_chars(clue) == list(clue.answer)
    
    def validate_all(self) -> bool:
        """Check if all entries in the puzzle are correct"""
        return all(self.validate_clue_chars(clue) for clue in self.clues if clue.answer)

    def undo(self) -> None:
        """Undo the last move"""
        if len(self.grid_history) <= 1:
            raise InvalidGridError("No moves to undo")
        self.grid_history.pop()
        clue_idx = self.clue_history.pop()
        self.clues[clue_idx].answered = False

    def reset(self) -> None:
        """Reset the puzzle to its initial state"""
        self.grid_history = [self.grid_history[0]]
        self.clue_history = []
        for clue in self.clues:
            clue.answered = False
            
    def __repr__(self):
        return f"<CrosswordPuzzle width={self.width} height={self.height} clues={len(self.clues)}>"
    
    def __str__(self):
        # Box drawing characters
        TOP_LEFT = "┌"
        TOP_RIGHT = "┐"
        BOTTOM_LEFT = "└"
        BOTTOM_RIGHT = "┘"
        HORIZONTAL = "─"
        VERTICAL = "│"
        UNUSED_CELL = "░"

        # Build the grid
        result = []

        # Top border
        result.append(TOP_LEFT + (HORIZONTAL * 3 * self.width) + TOP_RIGHT)

        # Grid content
        for row in range(self.height):
            formatted_cells = []
            for cell in self.current_grid.cells[row]:
                if cell.value is None and not any(
                    row == clue_row and cell.col == clue_col
                    for clue in self.clues
                    for clue_row, clue_col in clue.cells()
                ):
                    formatted_cells.append(f" {UNUSED_CELL} ")
                else:
                    formatted_cells.append(f" {cell.value or ' '} ")
            row_str = VERTICAL + "".join(formatted_cells) + VERTICAL
            result.append(row_str)

        # Bottom border
        result.append(BOTTOM_LEFT + (HORIZONTAL * 3 * self.width) + BOTTOM_RIGHT)

        return "\n".join(result)