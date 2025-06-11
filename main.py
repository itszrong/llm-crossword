import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from src.crossword.utils import load_puzzle

# Load environment variables from .env file
load_dotenv()

# Load the puzzle
puzzle = load_puzzle("data/easy.json")

print('--- Clues ---')
clue = puzzle.clues[0]
print(clue)

print('--- Set a guess ---')
puzzle.set_clue_chars(puzzle.clues[0], ["a", "b", "c"])
print(puzzle)

print('--- Undo ---')
puzzle.undo()
print(puzzle)

print('--- Set a guess for clue 1 ---')
puzzle.set_clue_chars(puzzle.clues[0], ["c", "a", "t"])

print('--- Set a guess for clue 2 ---')
puzzle.set_clue_chars(puzzle.clues[1], ["c", "o", "w"])

print('--- Completed all? ---')
print(puzzle.validate_all())
print(puzzle)

print('--- Set a guess for clue 3 ---')
puzzle.set_clue_chars(puzzle.clues[2], ["t", "e", "a","r"])

print('--- Completed all? ---')
print(puzzle.validate_all())
print(puzzle)

print('--- Reset ---')
puzzle.reset()
print(puzzle)

# grid history

print('--- OpenAI Hello World ---')
def openai_hello_world():
    client = AzureOpenAI(
        api_version=os.getenv("OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a helpful assistant."}, 
                  {"role": "user", "content": "Hello!"}]
    )
    return response.choices[0].message.content

print(openai_hello_world())
