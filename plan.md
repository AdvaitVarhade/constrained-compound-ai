# **Constrained Compound AI System \- Execution Plan**

## **1\. Context: What and How**

**The Objective:** Build an autonomous software engineering system ("AI OS") that executes complex coding tasks reliably on severely constrained consumer hardware (specifically, a GPU with a strict 4GB VRAM limit, like an RTX 3050).

**The Problem:** Monolithic 7B+ parameter LLMs require \>6GB of VRAM just for weights, context, and KV cache. Loading separate models for different agentic roles (Orchestrator, Coder, Debugger) sequentially causes catastrophic disk I/O latency.

**The Solution (The "How"):**

1. **Unified Base Model:** Run a single 4-bit quantized base model (e.g., Qwen-2.5-7B GGUF) using llama.cpp. We map roughly 16 layers to the 4GB GPU VRAM and offload the rest to system RAM.  
2. **LoRA Multiplexing:** Instead of loading different models, we load multiple tiny LoRA adapters into memory and dynamically hot-swap their scales (from 0.0 to 1.0) via the /lora-adapters API endpoint to shift personas in milliseconds.  
3. **GBNF Constraints:** We intercept logits at the C++ level to enforce strict JSON schemas, guaranteeing that tiny models output perfectly formatted tool-calls without hallucination.  
4. **Persistent Markdown Memory:** Instead of a massive 32k context window (which blows up the KV cache), we use a hyper-localized RAG approach. State is maintained in tiny local files (project.md, architecture.md, known\_bugs.md) which are injected into a strict 4096-token prompt window.  
5. **Git-Backed Sandbox:** Code modifications are protected by automated Git commits and rollbacks. If a compilation/test subprocess fails, the system logs the stderr trace and reverts the file hash.

## **2\. Core Rules & Instructions**

To ensure system stability, every component must adhere to the following unyielding constraints:

* **\[RULE 1\] Strict KV Cache Limit:** The context window is hard-capped at 4096 tokens. The TypeScript orchestrator must aggressively truncate known\_bugs.md and file inputs if they exceed this limit before sending the prompt to the inference gateway.  
* **\[RULE 2\] Deterministic Output Only:** All API calls to llama-server that require routing, logic, or file modification MUST include a .gbnf grammar file payload. Unconstrained string generation is forbidden except for the Documentation specialist.  
* **\[RULE 3\] Immutable Rollbacks:** The ToolchainBridge cannot write to a source file unless a git commit \-m "pre-edit checkpoint" has successfully resolved. If subprocess verification (compilation/testing) fails, a git reset \--hard HEAD\~1 must be triggered automatically.  
* **\[RULE 4\] OOM is Fatal, Latency is Acceptable:** If configuring llama.cpp, prioritize preventing Out-Of-Memory (OOM) crashes over token generation speed. Offloading to CPU (system RAM) is the accepted compromise.  
* **\[RULE 5\] Delta Generation Only:** The Coder LoRA must ONLY generate Unified Diffs or block-replacements. It must never attempt to rewrite an entire \>100 line file, as this destroys the generation sequence budget.

## **3\. Incremental Execution Phases**

### **Phase 1: Inference Physics & Provisioning**

* \[ \] Download Qwen-2.5-7B (Q4\_K\_M GGUF).  
* \[ \] Compile or download llama-server (llama.cpp).  
* \[ \] Determine exact \--n-gpu-layers to keep VRAM usage strictly under 3.5GB (leaving 0.5GB for OS overhead).  
* \[ \] Verify llama-server starts successfully with CPU/GPU hybrid offloading.  
* \[ \] Test the \--lora-init-without-apply flag with dummy LoRA adapters to verify memory mapping.

### **Phase 2: TypeScript Orchestration & Inference Gateway**

* \[ \] Initialize Node.js/TypeScript project.  
* \[ \] Build the InferenceGateway.ts class to wrap the llama-server REST API.  
* \[ \] Implement the /completion endpoint wrapper supporting grammar (GBNF) payloads.  
* \[ \] Implement the /lora-adapters hot-swapping logic (scale 0.0 to 1.0).  
* \[ \] **Verification:** Successfully generate a valid JSON object using a 7B model enforced by a GBNF schema, then hot-swap a LoRA and generate a different response.

### **Phase 3: Persistent Memory & Context Management**

* \[ \] Create the WorkspaceMemory.ts module.  
* \[ \] Implement robust fs (File System) read/write streams for architecture.md, project.md, and known\_bugs.md.  
* \[ \] Implement a lightweight token-counter function to ensure concatenated markdown files never exceed the 4000 token budget.  
* \[ \] **Verification:** System successfully reads the markdown state, detects a token overflow, and safely truncates the oldest bug logs before returning the prompt string.

### **Phase 4: Toolchain Bridge & Git Sandboxing**

* \[ \] Create the ToolchainBridge.ts module utilizing Node's child\_process.spawn and exec.  
* \[ \] Implement createGitCheckpoint() to snapshot the local repository.  
* \[ \] Implement runSubprocess() to execute compilers (e.g., tsc) or linters, capturing stdout, stderr, and exitCode.  
* \[ \] Implement executeRollback() (git reset \--hard HEAD\~1 \+ git clean \-fd).  
* \[ \] **Verification:** Programmatically inject a syntax error into a dummy file, run the verification subprocess, catch the non-zero exit code, and successfully rollback the file.

### **Phase 5: The Execution Loop (State Machine)**

* \[ \] Build the central orchestration loop connecting all components.  
* \[ \] Define the Orchestrator agent logic (reads state, outputs next action via JSON).  
* \[ \] Define the Coder agent logic (takes target file, outputs Unified Diff via JSON).  
* \[ \] Implement the Diff parsing algorithm to apply Coder outputs to the local files.  
* \[ \] **Verification:** Provide the system a high-level objective in project.md. System loops autonomously: Plans \-\> Codes \-\> Fails Test \-\> Debugs \-\> Codes \-\> Passes Test.

### **Phase 6: LoRA Fine-Tuning Pipeline (Data Engineering)**

* \[ \] Setup QLoRA training scripts (HuggingFace PEFT) for a cloud instance or 16GB GPU.  
* \[ \] Generate synthetic dataset for the Orchestrator (JSON routing rules).  
* \[ \] Generate dataset for the Coder (Unified Diffs mapped to commit messages).  
* \[ \] Train adapters, convert to GGUF format, and integrate back into Phase 1\.

## **4\. Bugs Faced & Resolutions**

*(This section acts as a living document to record low-level system failures and how we engineered around them).*

### **Bug 1: LoRA hot-swap via API fails silently.**

* **Symptom:** POST request to /lora-adapters returns 200 OK, but the model behavior does not change.  
* **Root Cause:** The llama-server was initialized without the LoRA files passed at startup. It cannot dynamically load *new* files from disk via the API, it can only scale adapters already mapped into memory.  
* **Resolution:** Modified the server startup command to include \--lora orchestrator.gguf \--lora-init-without-apply \--lora coder.gguf \--lora-init-without-apply. The API now successfully toggles their scales.

### **Bug 2: Unified Diff application fails due to missing whitespace.**

* **Symptom:** The Coder outputs a syntactically correct diff, but the patch command fails because the LLM truncated leading spaces on unmodified context lines.  
* **Root Cause:** LLMs struggle with exact whitespace reproduction in strict formatting scenarios.  
* **Resolution:** Abandoned standard Unix patch. Wrote a custom fuzzy-matching diff applier in TypeScript (WorkspaceMemory.applyDiff()) that locates the target block using Levenshtein distance on the non-whitespace characters, then replaces the block.

### **Bug 3: KV Cache Corruption after LoRA Swap**

* **Symptom:** After swapping from Orchestrator to Coder, the first few generated tokens are gibberish.  
* **Root Cause:** The existing KV cache holds attention vectors calculated using the Orchestrator's weights. When the Coder weights activate, those vectors are mathematically invalid.  
* **Resolution:** Forced a slot-clear or sent a completely fresh prompt string on the first request after a LoRA swap. llama.cpp handles KV invalidation if the prompt hashes differ significantly.

*(Add new bugs here as the implementation progresses...)*