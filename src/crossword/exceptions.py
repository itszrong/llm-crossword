class CrosswordError(Exception):
    """Base exception for crossword errors"""
    pass

class InvalidGridError(CrosswordError):
    """Raised when grid operations are invalid"""
    pass

class InvalidClueError(CrosswordError):
    """Raised when clue operations are invalid"""
    pass