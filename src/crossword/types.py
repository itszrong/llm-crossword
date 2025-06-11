from enum import Enum
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

class Direction(str, Enum):
    ACROSS = "across"
    DOWN = "down"

class Cell(BaseModel):
    row: int
    col: int
    value: Optional[str] = None

class Clue(BaseModel):
    number: int = Field(gt=0)
    text: str
    direction: Direction
    length: int = Field(gt=0)
    row: int = Field(ge=0)  
    col: int = Field(ge=0)
    answer: Optional[str] = None
    answered: bool = False

    def cells(self) -> List[Tuple[int, int]]:
        """Return list of (row, col) coordinates for this clue"""
        cells = []
        for i in range(self.length):
            if self.direction == Direction.ACROSS:
                cells.append((self.row, self.col + i))
            else:
                cells.append((self.row + i, self.col))
        return cells

class Grid(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0) 
    cells: List[List[Cell]]

    def initialize_empty(self) -> None:
        """Initialize an empty grid with blank cells"""
        self.cells = [[Cell(row=i, col=j) for j in range(self.width)] for i in range(self.height)]
