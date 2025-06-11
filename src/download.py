from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import puz
import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException
from unidecode import unidecode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Type aliases
GridPosition = Dict[str, int]
GridShape = Tuple[int, int]
GridMatrix = np.ndarray

@dataclass
class Position:
    """Represents a position in the crossword grid."""
    x: int
    y: int

@dataclass
class Clue:
    """Represents a crossword clue with its metadata."""
    number: int
    direction: str  # 'A' for across, 'D' for down
    text: str
    
    def __lt__(self, other: 'Clue') -> bool:
        """Enable sorting clues by number and direction."""
        if self.number == other.number:
            # Across clues come before Down clues
            return self.direction != 'D'
        return self.number < other.number

class CrosswordFetchError(Exception):
    """Custom exception for crossword fetching errors."""
    pass

class GuardianCrossword:
    """Handles fetching and processing Guardian crosswords."""
    
    BASE_URL = "https://www.theguardian.com/crosswords"
    
    def __init__(self, crossword_type: str, number: Optional[str] = None):
        """
        Initialize a new GuardianCrossword instance.
        
        Args:
            crossword_type: Type of crossword (cryptic, quick, etc.)
            number: Optional specific crossword number
        """
        self.crossword_type = crossword_type
        self.number = number or self._get_latest_number()
        self.json_data: Optional[Dict] = None
        self.puzzle = puz.Puzzle()

    def _get_latest_number(self) -> str:
        """
        Get the number of the latest crossword of specified type.
        
        Returns:
            Latest crossword number as string
        
        Raises:
            CrosswordFetchError: If unable to fetch or parse the crossword number
        """
        url = f'{self.BASE_URL}/series/{self.crossword_type}'
        pattern = fr'.*\/crosswords\/{self.crossword_type}\/[0-9]+'
        
        try:
            response = self._make_request(url)
            soup = BeautifulSoup(response.text, features="lxml")
            latest_links = soup.find_all(href=re.compile(pattern))
            
            if not latest_links:
                raise CrosswordFetchError(f"No crosswords found for type: {self.crossword_type}")
            
            latest_url = latest_links[0]['href']
            number_match = re.search(r'.*\/([0-9]+)', latest_url)
            
            if not number_match:
                raise CrosswordFetchError("Could not parse crossword number from URL")
            
            return number_match.group(1)
            
        except Exception as e:
            raise CrosswordFetchError(f"Failed to get latest number: {str(e)}")

    @staticmethod
    def _make_request(url: str, timeout: int = 10) -> requests.Response:
        """
        Make an HTTP request with error handling.
        
        Args:
            url: URL to request
            timeout: Request timeout in seconds
        
        Returns:
            Response object
        
        Raises:
            RequestException: If the request fails
        """
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except RequestException as e:
            raise CrosswordFetchError(f"HTTP request failed: {str(e)}")

    def fetch_crossword(self) -> None:
        """
        Fetch crossword data from The Guardian website.
        
        Raises:
            CrosswordFetchError: If fetching or parsing fails
        """
        url = f"{self.BASE_URL}/{self.crossword_type}/{self.number}"
        
        try:
            response = self._make_request(url)
            soup = BeautifulSoup(response.text, features="lxml")
            crossword_div = soup.find('div', {'class': 'js-crossword'})
            
            if not crossword_div:
                raise CrosswordFetchError("Crossword data not found on page")
            
            self.json_data = json.loads(crossword_div['data-crossword-data'])
            
        except json.JSONDecodeError as e:
            raise CrosswordFetchError(f"Failed to parse crossword JSON: {str(e)}")
        except Exception as e:
            raise CrosswordFetchError(f"Failed to fetch crossword: {str(e)}")

    def process_crossword(self) -> None:
        """
        Process crossword data and create .puz file.
        
        Raises:
            ValueError: If no crossword data is available
        """
        if not self.json_data:
            raise ValueError("No crossword data available. Call fetch_crossword() first.")
        
        self._set_puzzle_metadata()
        solution_matrix = self._initialize_grid()
        clues = self._process_entries(solution_matrix)
        self._set_puzzle_content(solution_matrix, clues)

    def _set_puzzle_metadata(self) -> None:
        """Set basic puzzle metadata."""
        self.puzzle.height = self.json_data['dimensions']['rows']
        self.puzzle.width = self.json_data['dimensions']['cols']
        self.puzzle.title = f"Guardian {self.json_data['name']}"
        self.puzzle.author = self.json_data.get('creator', {}).get('name', 'Unknown')

    def _initialize_grid(self) -> GridMatrix:
        """Initialize empty solution matrix."""
        return np.full(
            (self.puzzle.height, self.puzzle.width),
            '.',
            dtype=object
        )

    def _process_entries(self, solution_matrix: GridMatrix) -> List[Clue]:
        """Process crossword entries and fill the solution matrix."""
        clues = []
        for entry in self.json_data['entries']:
            clue = Clue(
                number=entry['number'],
                direction="D" if entry['direction'] == "down" else "A",
                text=unidecode(entry['clue'])
            )
            clues.append(clue)
            
            self._fill_grid(
                solution_matrix,
                entry['position'],
                clue.direction,
                entry['length'],
                entry['solution']
            )
        
        return clues

    def _set_puzzle_content(self, solution_matrix: GridMatrix, clues: List[Clue]) -> None:
        """Set puzzle solution, fill, and clues."""
        solution_string = ''.join(solution_matrix.flatten())
        fill_string = re.sub(r'[A-Za-z]', '-', solution_string)
        
        self.puzzle.solution = solution_string
        self.puzzle.fill = fill_string
        self.puzzle.clues = [c.text for c in sorted(clues)]

    def _fill_grid(
        self,
        solution_matrix: GridMatrix,
        position: GridPosition,
        clue_direction: str,
        clue_length: int,
        clue_solution: str
    ) -> None:
        """
        Fill a crossword grid with a solution.
        
        Args:
            solution_matrix: Grid matrix to fill
            position: Starting position coordinates
            clue_direction: 'A' for across or 'D' for down
            clue_length: Length of the solution
            clue_solution: The solution string
        
        Raises:
            ValueError: If position coordinates are invalid
        """
        if not self._validate_position(position, solution_matrix.shape, clue_length, clue_direction):
            raise ValueError(f"Invalid position or solution length: {position}")
        
        pos = Position(x=position['x'], y=position['y'])
        
        if clue_direction == "A":
            solution_matrix[pos.y, pos.x:pos.x + clue_length] = list(clue_solution)
        else:
            solution_matrix[pos.y:pos.y + clue_length, pos.x] = list(clue_solution)

    @staticmethod
    def _validate_position(
        position: GridPosition,
        matrix_shape: GridShape,
        clue_length: int,
        direction: str
    ) -> bool:
        """
        Validate position coordinates against matrix boundaries.
        
        Args:
            position: Starting position coordinates
            matrix_shape: Shape of the solution matrix
            clue_length: Length of the solution
            direction: 'A' for across or 'D' for down
        
        Returns:
            True if position is valid, False otherwise
        """
        rows, cols = matrix_shape
        x, y = position['x'], position['y']
        
        if x < 0 or y < 0 or x >= cols or y >= rows:
            return False
        
        if direction == "A" and x + clue_length > cols:
            return False
        if direction == "D" and y + clue_length > rows:
            return False
        
        return True

    def save_puzzle(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Save the puzzle to a .puz file.
        
        Args:
            output_dir: Optional directory to save the puzzle in
        """
        if not self.json_data:
            raise ValueError("No puzzle data to save")
        
        filename = f"Guardian {self.json_data['name']}.puz"
        if output_dir:
            output_path = Path(output_dir) / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path(filename)
        
        self.puzzle.save(str(output_path))
        logger.info(f"Successfully saved puzzle to {output_path}")

def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Guardian Crossword Puzzle Generator")
    parser.add_argument(
        '-t', '--type',
        default='cryptic',
        help='Crossword type (default: cryptic)'
    )
    parser.add_argument(
        '-n', '--number',
        help='Crossword number (default: latest)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        help='Output directory for the puzzle file'
    )
    
    args = parser.parse_args()
    
    try:
        crossword = GuardianCrossword(args.type, args.number)
        crossword.fetch_crossword()
        crossword.process_crossword()
        crossword.save_puzzle(args.output_dir)
        
    except CrosswordFetchError as e:
        logger.error(f"Failed to fetch crossword: {str(e)}")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()