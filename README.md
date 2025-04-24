# Recursive Agent Orchestration
A sophisticated recursive architecture that spawns and manages LLM-driven agents via a master-delegator model, with task-specific prompt tuning, execution constraints, and multi-agent recursion without human intervention.

> This uses a concept where a primary agent creates specialized sub-agents on demand. Each agent is dynamically generated with task-specific prompting and execution parameters, enabling efficient problem-solving without human intervention while not having a bias and having opinions from various agents instead of deciding on its own. Uses Ensemble technique.

# Architecture
## Here's how it works
![architecture diagram](/assets/architecture.png)

The system follows a recursive workflow:
- `Master Agent` analyzes the input task and generates a comprehensive task breakdown
- For each subtask, the `Master Agent` creates a specialized `Child Agent` with custom fine-tuned prompts
- `Child Agents` work on their assigned tasks and can create their own `Sub-Child Agents` when further specialization is needed
- Results flow back up the hierarchy for integration and final output

The key innovation is that agent creation and prompt engineering happen automatically at runtime, with no predefined agent structures or human-designed prompts.

## Caveats
As you see, this runs on recrusive method. building agents upon agents as it needs. This can result in building a complex agent tree as shown below
![agent tree](/assets/tree.png)

The recursive nature of this system creates potential challenges:
- **Agent Proliferation:** Complex tasks may spawn extensive agent trees
- **Resource Management:** Each additional agent consumes computational resources
- **Runtime Concerns:** Deep agent hierarchies can significantly impact completion time

While the system offers powerful flexibility by giving control to the `Master Agent`, careful monitoring is recommended for resource-intensive applications.

## Demo
https://github.com/user-attachments/assets/ec99d37f-c6c9-4fbd-a07e-1997d88aaa99



# Citations and Resources
1. Emergence AI's 2025 Orchestrator
Automatically creates agents and assembles multi-agent systems with minimal human intervention... continuously refining tools through recursive self-improvement

2. ReDel's Recursive Systems (2024)
Introduces systems where a root agent "decomposes tasks into subtasks then delegates to sub-agents" rather than using human-defined agent graphs.

3. Beyond Better's Orchestrator (2025)
Implements "sub-agents created with specialized capabilities for token efficiency and parallel processing" through dynamic task analysis.


# License
This project is licensed under the GNU General Public License (GPL).
