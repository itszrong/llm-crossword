import pytest
from src.crossword.crossword import CrosswordPuzzle
from src.crossword.types import Clue, Direction
from src.crossword.exceptions import InvalidClueError, InvalidGridError

@pytest.fixture
def clues():
    return [
        Clue(
            number=1,
            text="Feline friend",
            direction=Direction.ACROSS,
            length=3,
            row=0,
            col=0,
            answer="CAT"
        ),
        Clue(
            number=2,
            text="Dairy farm animal",
            direction=Direction.DOWN,
            length=3,
            row=0,
            col=0,
            answer="COW"
        ),
        Clue(
            number=3,
            text="A drop of sadness",
            direction=Direction.DOWN,
            length=4,
            row=0,
            col=2,
            answer="TEAR"
        )
    ]

@pytest.fixture
def puzzle(clues):
    """
    Creates a 5x5 puzzle with this layout:
    C A T - -
    O - E - -
    W - A - -
    - - R - -
    - - - - -
    """
    return CrosswordPuzzle(width=5, height=5, clues=clues)

class TestCrosswordPuzzle:
    def test_init(self, puzzle):
        assert puzzle.width == 5
        assert puzzle.height == 5
        assert len(puzzle.grid_history) == 1
        assert len(puzzle.clue_history) == 0
        
        # Check initial grid state
        current_grid = puzzle.current_grid
        assert current_grid.width == 5
        assert current_grid.height == 5
        assert all(cell.value in [None, "░"] 
                  for row in current_grid.cells 
                  for cell in row)

    def test_add_invalid_clue_position(self, puzzle):
        # Test clue extending beyond grid width
        invalid_across = Clue(
            number=4,
            text="Too long across",
            direction=Direction.ACROSS,
            length=6,  # Beyond grid width of 5
            row=0,
            col=0,
            answer="TOOLONG"
        )
        with pytest.raises(InvalidClueError):
            puzzle.add_clue(invalid_across)

        # Test clue extending beyond grid height
        invalid_down = Clue(
            number=5,
            text="Too long down",
            direction=Direction.DOWN,
            length=6,  # Beyond grid height of 5
            row=0,
            col=0,
            answer="TOOLONG"
        )
        with pytest.raises(InvalidClueError):
            puzzle.add_clue(invalid_down)

        # Test clue starting outside grid
        invalid_position = Clue(
            number=6,
            text="Outside grid",
            direction=Direction.ACROSS,
            length=3,
            row=6,  # Beyond grid height
            col=0,
            answer="BAD"
        )
        with pytest.raises(InvalidClueError):
            puzzle.add_clue(invalid_position)

    def test_set_clue_chars(self, puzzle):
        # Test setting first across clue
        clue = puzzle.clues[0]  # "CAT" across
        puzzle.set_clue_chars(clue, list("CAT"))
        
        chars = puzzle.get_current_clue_chars(clue)
        assert chars == list("CAT")
        assert puzzle.clues[0].answered is True
        assert len(puzzle.grid_history) == 2
        assert len(puzzle.clue_history) == 1

        # Test setting invalid length
        with pytest.raises(InvalidClueError):
            puzzle.set_clue_chars(clue, list("TOOLONG"))

    def test_get_current_clue_chars(self, puzzle):
        # Initially all cells should be None
        clue = puzzle.clues[0]  # "CAT" across
        assert puzzle.get_current_clue_chars(clue) == [None, None, None]

        # Set some characters and verify
        puzzle.set_clue_chars(clue, list("CAT"))
        assert puzzle.get_current_clue_chars(clue) == list("CAT")

        # Test getting chars for non-existent clue
        invalid_clue = Clue(
            number=99,
            text="Invalid",
            direction=Direction.ACROSS,
            length=3,
            row=0,
            col=0,
            answer="BAD"
        )
        with pytest.raises(InvalidClueError):
            puzzle.get_current_clue_chars(invalid_clue)

    def test_undo(self, puzzle):
        raise NotImplementedError

    def test_reset(self, puzzle):
        # Make some moves
        puzzle.set_clue_chars(puzzle.clues[0], list("CAT"))
        puzzle.set_clue_chars(puzzle.clues[1], list("COW"))

        # Verify initial state
        assert len(puzzle.grid_history) == 3
        assert len(puzzle.clue_history) == 2
        assert puzzle.clues[0].answered is True
        assert puzzle.clues[1].answered is True

        # Reset and verify
        puzzle.reset()
        assert len(puzzle.grid_history) == 1
        assert len(puzzle.clue_history) == 0
        assert puzzle.clues[0].answered is False
        assert puzzle.clues[1].answered is False
        assert all(char is None for char in puzzle.get_current_clue_chars(puzzle.clues[0]))
        assert all(char is None for char in puzzle.get_current_clue_chars(puzzle.clues[1]))

    def test_reveal_clue_answer(self, puzzle):
        clue = puzzle.clues[0]  # "CAT" across
        
        # Reveal clue
        puzzle.reveal_clue_answer(clue)
        assert puzzle.get_current_clue_chars(clue) == list("CAT")
        assert clue.answered is True

        # Test revealing clue with no answer
        no_answer_clue = Clue(
            number=99,
            text="No answer",
            direction=Direction.ACROSS,
            length=3,
            row=4,
            col=0,
            answer=None
        )
        with pytest.raises(InvalidClueError):
            puzzle.reveal_clue_answer(no_answer_clue)

    def test_reveal_all(self, puzzle):
        # Reveal all clues
        puzzle.reveal_all()

        # Verify all clues are answered with correct answers
        for clue in puzzle.clues:
            assert clue.answered is True
            assert puzzle.get_current_clue_chars(clue) == list(clue.answer)

        # Verify grid state
        current_grid = puzzle.current_grid
        assert current_grid.cells[0][0].value == "C"  # First cell of CAT/COW
        assert current_grid.cells[0][1].value == "A"  # Second cell of CAT
        assert current_grid.cells[0][2].value == "T"  # Third cell of CAT
        assert current_grid.cells[1][0].value == "O"  # Second cell of COW
        assert current_grid.cells[2][0].value == "W"  # Third cell of COW
        assert current_grid.cells[1][2].value == "E"  # Second cell of TEAR
        assert current_grid.cells[2][2].value == "A"  # Third cell of TEAR
