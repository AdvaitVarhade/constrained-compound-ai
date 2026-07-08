# Constrained Compound AI System

A highly constrained, autonomous agentic loop designed to scaffold and manage software projects strictly within a 4GB VRAM budget. Built entirely in TypeScript, this system leverages the Qwen 2.5 Coder 3B model via `llama.cpp` and employs a deterministic "Airlock" architecture to prevent LLM hallucinations from breaking the autonomous loop.

##  Key Features

* **Incremental Process Model**: Adheres to strict, single-step execution boundaries. The system plans one step, scaffolds the code, dynamically tests it, and then commits the changes via a Git-backed sandbox.
* **Orchestrator-Evaluator Pattern**: The engine autonomously evaluates its own progress against the overarching objective using a dual-persona pattern. The **Orchestrator** plans the execution, while the **Evaluator** uses the base model to verify completion.
* **LoRA Hot-Swapping**: Operates entirely within 4GB VRAM by multiplexing multiple LoRA adapters (Orchestrator, Coder, Debugger) on top of a single base model. Adapters are hot-swapped in milliseconds between inference phases.
* **Self-Healing Sandbox Dependencies**: Dynamically parses compilation/verification errors for missing npm modules or typescript type declarations, automatically executes safe `npm install` inside the sandbox, and immediately re-triggers verification.
* **Multi-File Scoped Transactions**: Supports orchestration and editing of multiple target files sequentially within a single transactional Git checkpoint before compiling and committing.
* **Command-Line Interface (CLI)**: Features a CLI runner (`cli.ts`) allowing users to execute the Agent OS against any target directory with a custom objective and iteration boundaries.
* **KV Cache Slot Protection**: Implements randomized section-level cache-busting sequences to force the `llama-server` to evaluate from scratch on persona/LoRA swaps, completely preventing KV cache slot leakage and model corruption.
* **GBNF Explanation Key-Locking**: Locks the GBNF grammar `"explanation"` property to a fixed value (`"success"`) for code scaffolding to prevent smaller Coder models from falling into infinite explanation repetition loops.
* **Deterministic Sandboxing (`ToolchainBridge.ts` & `WorkspaceMemory.ts`)**:
  * **Git-Backed Rollbacks**: If the Coder generates a malformed diff or introduces a syntax error, the transaction is instantly rolled back to the pre-edit Git commit.
  * **Output Sanitization**: Strips markdown fences from hallucinated LLM responses before passing them into the deterministic JSON parsers.
  * **Strict GBNF Grammars**: Enforces valid JSON emission natively at the inference level (`llama.cpp` GBNF) to ensure the State Machine never crashes due to malformed output.
  * **Forbidden Command Blocking**: Sandboxes terminal verification commands to prevent the LLM from trying to manually inject code via shell (e.g., blocking `echo ... > file`).


##  Architecture Overview

### 1. Agent Engine (`AgentEngine.ts`)
The central nervous system of the agent. It implements the bounded event loop that drives the `IDLE -> ORCHESTRATOR -> CODING -> VERIFYING -> DEBUG -> COMPLETED` state machine. It parses GBNF outputs and dynamically swaps the active persona.

### 2. Inference Gateway (`InferenceGateway.ts`)
A lightweight wrapper around the local `llama.cpp` server API. It handles LoRA multiplexing, streams prompts to the local LLM, and sanitizes the returned JSON strings.

### 3. Toolchain Bridge (`ToolchainBridge.ts`)
The execution sandbox. It parses unified diffs and executes verification commands (`tsc --noEmit`, `node`, etc.). If a command times out or returns a non-zero exit code, it relays the `stderr` logs back to the LLM for a debug iteration.

### 4. Workspace Memory (`WorkspaceMemory.ts`)
Manages the agent's short-term and long-term memory. It maintains the project's overarching objective (`project.md`), tracks known bugs across rollbacks (`known_bugs.md`), and intelligently builds the context window using only relevant tracked files.

##  Setup & Execution

### Prerequisites
* Node.js (v18+)
* Git
* A CUDA 12.4 compatible GPU with at least 4GB VRAM
* `qwen2.5-coder-3b-q4_k_m.gguf` base model placed in the `./models/` directory

### Running the System
1. **Install Dependencies**
   ```bash
   npm install
   ```
2. **Start the Inference Server**
   Start the local `llama.cpp` server with the LoRA adapters:
   ```bash
   .\start-inference-server.bat
   ```
3. **Execute the Engine**
   ```bash
   npx tsx AgentEngine.ts
   ```

## 🛡️ License
MIT License.
