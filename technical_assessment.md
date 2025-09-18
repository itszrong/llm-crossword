Innovation Fellows Technical Assessment
Problem Statement
The Prime Minister has an urgent request! Due to the high workload of the job, they need
help solving their essential morning crossword. In this exercise, you'll build a LLM powered
crossword puzzle solver that demonstrates your software engineering and AI skills.
Assessment Structure
This assessment will be split into 2 sections:
- Part 1: 2 hours private coding time
- Part 2: 30 minutes paired programming session
After working on the individual coding task, you will join the assessment call to pair program
with an interviewer. Here you will discuss your solution and continue developing the
crossword puzzle solver.
When the interview is over please send a zip file of your code to the interviewer.
Guidelines
● Try to build an efficient and fast solution
● Focus on clean, maintainable code
● Be prepared to talk about key decisions and tradeoffs
● There is no single correct solution that we are looking for
● You are not expected to complete the challenge, but you should be able to explain
how you could get there
● AI assisted coding is allowed
● You can use any tools or libraries that you see fit
Tasks
Build an LLM powered crossword puzzle solver for the json crossword files, starting with
‘data/easy.json’, then progressing to medium, hard and finally cryptic.
The solver should:
1. Read the clues and provide possible answers
2. Resolve any conflicting answers
3. Return a completed crossword as a result
We provide a `main.py` script to help you get started. However, you can structure your code
whichever way you think is best. We also provide a `scratchpad.ipynb` notebook to help you
experiment with trying different solutions.
The crossword answers are included with the puzzles to help you validate your solutions, but
your solution should complete the crossword without seeing any answers.