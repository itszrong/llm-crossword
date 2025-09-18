# Agentic Crossword Solver Flow Diagram

```mermaid
graph TB
    %% Entry Points
    User[🧑‍💻 User] --> TestRunner[test_agentic_solver.py]
    User --> DirectSolver[AgenticCrosswordSolver]
    
    %% Main Solver Interface
    TestRunner --> AgenticSolver["AgenticCrosswordSolver<br/>📋 Main Interface"]
    DirectSolver --> AgenticSolver
    
    AgenticSolver --> |"1. Initialize"| DifficultyConfig["DifficultyConfigurator<br/>⚙️ Configure based on difficulty"]
    DifficultyConfig --> |"Easy/Medium/Hard/Cryptic configs"| CoordinatorSetup[Create Specialized Coordinator]
    
    %% Coordinator Agent Setup
    CoordinatorSetup --> Coordinator["CoordinatorAgent<br/>🎯 Main Orchestrator"]
    
    Coordinator --> |"Initialize Tools & Agents"| Tools["CrosswordTools<br/>🛠️ LLM Interface"]
    Coordinator --> ClueAgent["ClueAgent<br/>🧩 Clue Solver"]
    Coordinator --> ConstraintAgent["ConstraintAgent<br/>🔗 Validation & Conflicts"]
    Coordinator --> ReviewAgent["ReviewAgent<br/>✅ Quality Control"]
    Coordinator --> VisualizationAgent["VisualizationAgent<br/>📊 State Tracking"]
    
    %% Main Solving Loop
    Coordinator --> |"solve_puzzle()"| SolvingLoop{{"🔄 Solving Iterations<br/>(max 5-12 based on difficulty)"}}
    
    %% Phase 1: Candidate Generation
    SolvingLoop --> |"Phase 1"| GenerateCandidates["_generate_candidates<br/>🎯 Generate Solutions"]
    GenerateCandidates --> |"For each unsolved clue"| ClueAgent
    
    ClueAgent --> |"solve_clue() or solve_clue_async()"| Tools
    Tools --> |"LLM Call with context"| LLMPrompt["🤖 Specialized Prompts<br/>• Context-aware<br/>• Attempt history<br/>• Pattern constraints<br/>• Difficulty-specific"]
    
    LLMPrompt --> |"Parse response"| Candidates["ClueCandidate List<br/>• word<br/>• confidence<br/>• reasoning<br/>• clue_type"]
    
    %% Async Processing Branch
    GenerateCandidates --> |"If multiple clues"| AsyncCheck{Async Enabled?}
    AsyncCheck --> |"Yes (Medium+)"| AsyncGenerate["_generate_candidates_async<br/>⚡ Concurrent Processing"]
    AsyncCheck --> |"No (Easy)"| SequentialGenerate[Sequential Processing]
    
    AsyncGenerate --> |"Semaphore-limited"| ClueAgent
    SequentialGenerate --> ClueAgent
    
    %% Phase 2: Review Candidates  
    Candidates --> |"Phase 2"| ReviewCandidates["_review_candidates<br/>🔍 Quality Filter"]
    ReviewCandidates --> ReviewAgent
    ReviewAgent --> |"review_solution()"| QualityScore["Quality Scoring<br/>• Semantic check<br/>• Context consistency<br/>• Intersection compatibility"]
    
    %% Phase 3: Conflict Resolution
    QualityScore --> |"Phase 3"| ResolveConflicts["resolve_conflicts<br/>⚖️ Select Best Solutions"]
    ResolveConflicts --> ConstraintAgent
    
    ConstraintAgent --> |"Validate intersections"| ValidationChecks["Validation Checks<br/>• Length constraints<br/>• Pattern matching<br/>• Intersection conflicts"]
    
    ValidationChecks --> |"Priority scoring"| PriorityCalc["Priority Calculation<br/>confidence × constraint_factor × boosts<br/>• Multi-word boost<br/>• High-confidence boost<br/>• Technical clue boost"]
    
    PriorityCalc --> |"Greedy selection"| Solutions["Selected Solutions<br/>List of (Clue, Candidate) pairs"]
    
    %% Phase 4: Apply Solutions
    Solutions --> |"Phase 4"| ApplySolutions["_apply_solutions<br/>📝 Update Grid"]
    ApplySolutions --> |"set_clue_chars()"| GridUpdate["Grid Update<br/>• Update cell values<br/>• Mark clues as answered<br/>• Track history"]
    
    %% Visualization & Logging
    GridUpdate --> VisualizationAgent
    VisualizationAgent --> |"capture_grid_state()"| StateCapture["State Capture<br/>• Grid visualization<br/>• Clue states<br/>• Statistics<br/>• Change detection"]
    
    StateCapture --> |"Log iteration"| IterationLog["Iteration Logging<br/>• Candidates generated<br/>• Solutions applied<br/>• Progress made<br/>• Visual diffs"]
    
    %% Loop Control
    IterationLog --> ProgressCheck{Progress Made?}
    ProgressCheck --> |"Yes"| CompletionCheck{Puzzle Complete?}
    ProgressCheck --> |"No"| RetryLogic["Retry Logic<br/>retry_count++"]
    
    CompletionCheck --> |"Yes ✅"| Success["🎉 Success!<br/>Save logs & return"]
    CompletionCheck --> |"No"| SolvingLoop
    
    RetryLogic --> |"< 3 retries"| SolvingLoop
    RetryLogic --> |">= 3 retries"| ReviewTrigger{"Review System<br/>Enabled?"}
    
    %% Review System Branch
    ReviewTrigger --> |"Yes (Hard/Cryptic)"| TwoStageReview["TwoStageReviewSystem<br/>🎭 Advanced Recovery"]
    ReviewTrigger --> |"No"| Failure["❌ Incomplete<br/>Save logs & return"]
    
    %% Two-Stage Review Process
    TwoStageReview --> |"Stage 1"| ReviewStage["ReviewAgent Analysis<br/>🔍 Identify Issues"]
    ReviewStage --> |"analyze_puzzle_state()"| IssueAnalysis["Issue Analysis<br/>• Unsolved clues<br/>• Intersection conflicts<br/>• Semantic errors<br/>• Pattern constraints"]
    
    IssueAnalysis --> |"Generate insights"| ReviewInsights["ReviewInsights<br/>• Issue type<br/>• Confidence<br/>• Description<br/>• Suggested action"]
    
    ReviewInsights --> |"Stage 2"| CorrectionStage["CorrectionAgent<br/>🔧 Apply Fixes"]
    CorrectionStage --> |"apply_corrections()"| CorrectionProcess["Correction Process<br/>• Priority clue selection<br/>• Generate alternatives<br/>• Test candidates<br/>• Apply solutions"]
    
    CorrectionProcess --> |"Max 3 corrections"| CorrectionResult{"Corrections<br/>Applied?"}
    
    CorrectionResult --> |"Yes"| FinalCheck{"Final Puzzle<br/>Complete?"}
    CorrectionResult --> |"No"| Failure
    
    FinalCheck --> |"Yes"| Success
    FinalCheck --> |"No"| PartialSuccess["⚡ Partial Success<br/>Some improvements made"]
    
    %% Final Outputs
    Success --> FinalLog["📊 Final Logging<br/>• Performance metrics<br/>• Detailed JSON logs<br/>• Grid visualization<br/>• Solution validation"]
    Failure --> FinalLog
    PartialSuccess --> FinalLog
    
    %% Tool Details Subgraph
    subgraph ToolDetails["🛠️ CrosswordTools Details"]
        direction TB
        ToolMethods["Tool Methods<br/>• solve_clue()<br/>• validate_intersection()<br/>• get_grid_context()<br/>• get_current_pattern()"]
        
        PromptEngineering["Prompt Engineering<br/>• _classify_clue_type()<br/>• _build_clue_prompt()<br/>• _parse_clue_response()"]
        
        ContextAware["Context Awareness<br/>• Attempt history<br/>• Pattern constraints<br/>• Intersection info<br/>• Difficulty settings"]
        
        LLMInterface["LLM Interface<br/>• Azure OpenAI<br/>• GPT-4o<br/>• Structured prompts<br/>• Response parsing"]
        
        ToolMethods --> PromptEngineering
        PromptEngineering --> ContextAware
        ContextAware --> LLMInterface
    end
    
    %% Specialized Solvers Subgraph
    subgraph SpecializedSolvers["⚙️ Difficulty Specialization"]
        direction TB
        EasySolver["Easy Solver<br/>• 3 iterations<br/>• No backtracking<br/>• Sync processing<br/>• Simple prompts"]
        
        MediumSolver["Medium Solver<br/>• 5 iterations<br/>• Backtracking enabled<br/>• Async processing<br/>• Enhanced prompts"]
        
        HardSolver["Hard Solver<br/>• 8 iterations<br/>• Deep reasoning<br/>• Constraint propagation<br/>• Advanced prompts"]
        
        CrypticSolver["Cryptic Solver<br/>• 12 iterations<br/>• Wordplay analysis<br/>• Anagram detection<br/>• Cryptic prompts"]
    end
    
    %% Data Flow Annotations
    classDef userClass fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef agentClass fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef toolClass fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef processClass fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef decisionClass fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef outputClass fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    
    class User,TestRunner userClass
    class Coordinator,ClueAgent,ConstraintAgent,ReviewAgent,VisualizationAgent agentClass
    class Tools,LLMPrompt toolClass
    class GenerateCandidates,ReviewCandidates,ResolveConflicts,ApplySolutions processClass
    class ProgressCheck,CompletionCheck,AsyncCheck,ReviewTrigger decisionClass
    class Success,Failure,PartialSuccess,FinalLog outputClass
```
