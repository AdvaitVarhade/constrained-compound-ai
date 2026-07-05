import json
import os
from typing import List, Dict, Any

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Please install transformers: pip install transformers")
    # For now, we allow fallback approximation if transformers isn't available locally
    # exit(1)

# Using Qwen-2.5 tokenizer for token counting constraint verification
# We use the 3B Coder model which is the absolute maximum that can fit in 4GB VRAM with QLoRA
TOKENIZER_MODEL = "Qwen/Qwen2.5-Coder-3B"
# Strict 4096 KV cache limit, reserving 96 tokens for overhead
MAX_TOKENS = 4000 

def load_tokenizer():
    """Attempts to load the HuggingFace tokenizer. Falls back to None if unavailable."""
    print(f"Attempting to load tokenizer {TOKENIZER_MODEL}...")
    try:
        # trust_remote_code is often required for modern models
        return AutoTokenizer.from_pretrained(TOKENIZER_MODEL, trust_remote_code=True)
    except Exception as e:
        print(f"Warning: Could not load {TOKENIZER_MODEL}. Falling back to character-based length approximation. Error: {e}")
        return None

def verify_and_save(dataset: List[Dict[str, Any]], filename: str, tokenizer) -> None:
    """
    Verifies that no input-output pair exceeds the max token limit and saves to .jsonl
    This enforces [RULE 1] Strict KV Cache Limit before training even begins.
    """
    valid_examples = []
    dropped = 0
    
    for example in dataset:
        text_to_tokenize = ""
        # We assume messages format for chat templates
        if "messages" in example:
            for msg in example["messages"]:
                text_to_tokenize += f"{msg['role']}: {msg['content']}\n"
        else:
            text_to_tokenize = example.get("text", "")
            
        if tokenizer:
            tokens = tokenizer.encode(text_to_tokenize)
            token_count = len(tokens)
        else:
            # Fallback approximation (4 chars ~= 1 token)
            token_count = len(text_to_tokenize) // 4
            
        if token_count <= MAX_TOKENS:
            valid_examples.append(example)
        else:
            print(f"Dropping example due to length constraint ({token_count} > {MAX_TOKENS})")
            dropped += 1
            
    with open(filename, 'w', encoding='utf-8') as f:
        for ex in valid_examples:
            f.write(json.dumps(ex) + '\n')
            
    print(f"Saved {len(valid_examples)} examples to {filename}. Dropped {dropped} due to token limit.")


# ---------------------------------------------------------------------------
#  Helper: build a prompt string identical to AgentEngine.buildBoundedPrompt
# ---------------------------------------------------------------------------
def _build_prompt(instruction: str, sections: Dict[str, str]) -> str:
    """
    Mirrors the AgentEngine render() logic:
        instruction.trim()
        \n\n## LABEL\ncontent.trim()
        ...
    """
    parts = [instruction.strip()]
    for label, content in sections.items():
        parts.append(f"## {label}\n{content.strip()}")
    return "\n\n".join(parts)


# ===========================================================================
#  ORCHESTRATOR DATA  — must exactly match executeOrchestrator() at inference
# ===========================================================================
def _orch_instruction(run_seq: int = 1, iteration: int = 1) -> str:
    """Builds the exact instruction string used by AgentEngine.executeOrchestrator."""
    return "\n".join([
        "ROLE: Orchestrator.",
        f"RUN_SEQUENCE: {run_seq}; ITERATION: {iteration}.",
        "Choose exactly one next action from write_code, run_tests, or done.",
        "For write_code, provide targetFile and optionally command.",
        "For run_tests, optionally provide command. A command is executable plus arguments, never shell syntax.",
        "Use done only when the project objective is demonstrably complete.",
        "Return exactly one JSON object in this property order:",
        '{"action":"write_code|run_tests|done","targetFile":"optional","command":"optional"}',
        "Do not include Markdown, commentary, or additional properties.",
    ])


def generate_orchestrator_data() -> List[Dict[str, Any]]:
    """
    Generates synthetic dataset for the Orchestrator adapter.
    Every example mirrors the exact prompt layout produced by
    AgentEngine.buildBoundedPrompt + memorySections at inference time.
    """
    dataset: List[Dict[str, Any]] = []

    def add(instruction: str, user_prompt: str, target: dict):
        dataset.append({
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": json.dumps(target)},
            ]
        })

    # ------------------------------------------------------------------
    # 1. Brand-new project → write_code with verification command
    # ------------------------------------------------------------------
    add(
        _orch_instruction(1, 1),
        _build_prompt("", {
            "ARCHITECTURE": "No architecture specified.",
            "PROJECT STATE": (
                "Objective: Scaffold a Node.js Express backend.\n"
                "Current Status: Repository initialized. Please write the initial server.ts file."
            ),
            "KNOWN BUGS": "None.",
        }),
        {"action": "write_code", "targetFile": "server.ts", "command": "npx tsc server.ts --noEmit"},
    )

    # ------------------------------------------------------------------
    # 2. Brand-new project, simpler objective
    # ------------------------------------------------------------------
    add(
        _orch_instruction(1, 1),
        _build_prompt("", {
            "ARCHITECTURE": "Single-file CLI utility.",
            "PROJECT STATE": (
                "Objective: Create a Node.js CLI tool that converts CSV to JSON.\n"
                "Current Status: Not started."
            ),
            "KNOWN BUGS": "None.",
        }),
        {"action": "write_code", "targetFile": "convert.ts", "command": "npx tsc convert.ts --noEmit"},
    )

    # ------------------------------------------------------------------
    # 3. Server file exists, needs a utility module
    # ------------------------------------------------------------------
    add(
        _orch_instruction(1, 2),
        _build_prompt("", {
            "ARCHITECTURE": "Express backend with utilities in utils/.",
            "PROJECT STATE": (
                "Objective: Add a date-formatting utility module.\n"
                "Current Status: server.ts exists and compiles."
            ),
            "KNOWN BUGS": "None.",
        }),
        {"action": "write_code", "targetFile": "utils/dateFormat.ts", "command": "npx tsc --noEmit"},
    )

    # ------------------------------------------------------------------
    # 4. Existing file needs a bug fix
    # ------------------------------------------------------------------
    add(
        _orch_instruction(1, 3),
        _build_prompt("", {
            "ARCHITECTURE": "Express backend.",
            "PROJECT STATE": (
                "Objective: Fix the port configuration bug in server.ts.\n"
                "Current Status: server.ts exists but crashes on startup."
            ),
            "KNOWN BUGS": "Error: listen EADDRINUSE: address already in use :::3000",
        }),
        {"action": "write_code", "targetFile": "server.ts", "command": "node server.js"},
    )

    # ------------------------------------------------------------------
    # 5. After a successful write, run tests
    # ------------------------------------------------------------------
    add(
        _orch_instruction(1, 2),
        _build_prompt("", {
            "ARCHITECTURE": "Express REST API with Jest tests.",
            "PROJECT STATE": (
                "Objective: Implement user registration endpoint.\n"
                "Current Status: server.ts updated with /register route.\n\n"
                "## Verified AI change — 2026-01-15T10:00:00.000Z\n"
                "- Target: `server.ts`\n"
                "- Verification: `npx tsc --noEmit`\n"
                "- Attempts: 1"
            ),
            "KNOWN BUGS": "None.",
        }),
        {"action": "run_tests", "command": "npx jest --passWithNoTests"},
    )

    # ------------------------------------------------------------------
    # 6. Run tests with a simple node command
    # ------------------------------------------------------------------
    add(
        _orch_instruction(1, 2),
        _build_prompt("", {
            "ARCHITECTURE": "Single-file Node.js script.",
            "PROJECT STATE": (
                "Objective: Create a working calculator module.\n"
                "Current Status: calculator.ts written and compiled.\n\n"
                "## Verified AI change — 2026-01-15T10:00:00.000Z\n"
                "- Target: `calculator.ts`\n"
                "- Verification: `npx tsc calculator.ts --noEmit`\n"
                "- Attempts: 1"
            ),
            "KNOWN BUGS": "None.",
        }),
        {"action": "run_tests", "command": "node calculator.js"},
    )

    # ------------------------------------------------------------------
    # 7. All tasks complete → done
    # ------------------------------------------------------------------
    add(
        _orch_instruction(1, 4),
        _build_prompt("", {
            "ARCHITECTURE": "Express REST API.",
            "PROJECT STATE": (
                "Objective: Implement user registration endpoint.\n"
                "Current Status: All endpoints implemented and tests passing.\n\n"
                "## Verified AI change — 2026-01-15T10:00:00.000Z\n"
                "- Target: `server.ts`\n"
                "- Verification: `npx tsc --noEmit`\n"
                "- Attempts: 1\n\n"
                "## Verified AI change — 2026-01-15T10:05:00.000Z\n"
                "- Target: `routes/register.ts`\n"
                "- Verification: `npx jest`\n"
                "- Attempts: 1"
            ),
            "KNOWN BUGS": "None.",
        }),
        {"action": "done"},
    )

    # ------------------------------------------------------------------
    # 8. write_code for a config file
    # ------------------------------------------------------------------
    add(
        _orch_instruction(1, 1),
        _build_prompt("", {
            "ARCHITECTURE": "TypeScript project.",
            "PROJECT STATE": (
                "Objective: Initialize TypeScript configuration.\n"
                "Current Status: No tsconfig.json exists."
            ),
            "KNOWN BUGS": "None.",
        }),
        {"action": "write_code", "targetFile": "tsconfig.json", "command": "npx tsc --noEmit"},
    )

    # ------------------------------------------------------------------
    # 9. write_code for a test file
    # ------------------------------------------------------------------
    add(
        _orch_instruction(1, 3),
        _build_prompt("", {
            "ARCHITECTURE": "Express backend with Jest.",
            "PROJECT STATE": (
                "Objective: Add unit tests for the auth module.\n"
                "Current Status: auth.ts exists and compiles."
            ),
            "KNOWN BUGS": "None.",
        }),
        {"action": "write_code", "targetFile": "auth.test.ts", "command": "npx jest auth.test.ts"},
    )

    # ------------------------------------------------------------------
    # 10. write_code for middleware
    # ------------------------------------------------------------------
    add(
        _orch_instruction(1, 2),
        _build_prompt("", {
            "ARCHITECTURE": "Express with middleware chain.",
            "PROJECT STATE": (
                "Objective: Add request logging middleware.\n"
                "Current Status: server.ts running, no middleware."
            ),
            "KNOWN BUGS": "None.",
        }),
        {"action": "write_code", "targetFile": "middleware/logger.ts", "command": "npx tsc --noEmit"},
    )

    # ------------------------------------------------------------------
    # 11. After tests fail, write_code to fix the file
    # ------------------------------------------------------------------
    add(
        _orch_instruction(1, 3),
        _build_prompt("", {
            "ARCHITECTURE": "Express REST API with Jest.",
            "PROJECT STATE": (
                "Objective: Implement user registration endpoint.\n"
                "Current Status: Tests failing for /register route."
            ),
            "KNOWN BUGS": (
                "## Verification failure — 2026-01-15T10:10:00.000Z\n"
                "- Command: `npx jest`\n"
                "- Exit code: 1\n\n"
                "### Diagnostic\n"
                "```text\n"
                "FAIL  register.test.ts\n"
                "  ● POST /register › returns 201\n"
                "    Expected: 201, Received: 500\n"
                "```"
            ),
        }),
        {"action": "write_code", "targetFile": "routes/register.ts", "command": "npx jest"},
    )

    # ------------------------------------------------------------------
    # 12. write_code for a package.json script
    # ------------------------------------------------------------------
    add(
        _orch_instruction(1, 1),
        _build_prompt("", {
            "ARCHITECTURE": "Node.js project.",
            "PROJECT STATE": (
                "Objective: Set up the project with a proper package.json.\n"
                "Current Status: Empty directory, no package.json."
            ),
            "KNOWN BUGS": "None.",
        }),
        {"action": "write_code", "targetFile": "package.json", "command": "node -e \"require('./package.json')\""},
    )

    # ------------------------------------------------------------------
    # 13. run_tests with tsc for type checking
    # ------------------------------------------------------------------
    add(
        _orch_instruction(1, 3),
        _build_prompt("", {
            "ARCHITECTURE": "TypeScript monorepo.",
            "PROJECT STATE": (
                "Objective: Ensure all TypeScript files compile.\n"
                "Current Status: Multiple files written.\n\n"
                "## Verified AI change — 2026-01-15T10:00:00.000Z\n"
                "- Target: `src/index.ts`\n"
                "- Verification: `npx tsc --noEmit`\n"
                "- Attempts: 1"
            ),
            "KNOWN BUGS": "None.",
        }),
        {"action": "run_tests", "command": "npx tsc --noEmit"},
    )

    # ------------------------------------------------------------------
    # 14. write_code for a .env or config
    # ------------------------------------------------------------------
    add(
        _orch_instruction(1, 2),
        _build_prompt("", {
            "ARCHITECTURE": "Express backend with dotenv.",
            "PROJECT STATE": (
                "Objective: Add environment configuration.\n"
                "Current Status: server.ts references process.env but no config exists."
            ),
            "KNOWN BUGS": "None.",
        }),
        {"action": "write_code", "targetFile": "config.ts", "command": "npx tsc config.ts --noEmit"},
    )

    # ------------------------------------------------------------------
    # 15. done after all verification passes
    # ------------------------------------------------------------------
    add(
        _orch_instruction(1, 5),
        _build_prompt("", {
            "ARCHITECTURE": "CLI utility.",
            "PROJECT STATE": (
                "Objective: Create a CSV-to-JSON converter.\n"
                "Current Status: Converter complete and tested.\n\n"
                "## Verified AI change — 2026-01-15T10:00:00.000Z\n"
                "- Target: `convert.ts`\n"
                "- Verification: `npx tsc convert.ts --noEmit`\n"
                "- Attempts: 1\n\n"
                "## Verified AI change — 2026-01-15T10:05:00.000Z\n"
                "- Target: `convert.test.ts`\n"
                "- Verification: `npx jest`\n"
                "- Attempts: 1"
            ),
            "KNOWN BUGS": "None.",
        }),
        {"action": "done"},
    )

    # ------------------------------------------------------------------
    # 16. write_code for an index.ts entry point
    # ------------------------------------------------------------------
    add(
        _orch_instruction(1, 1),
        _build_prompt("", {
            "ARCHITECTURE": "Modular TypeScript project.",
            "PROJECT STATE": (
                "Objective: Create the main entry point that imports all modules.\n"
                "Current Status: Utility modules exist, no entry point."
            ),
            "KNOWN BUGS": "None.",
        }),
        {"action": "write_code", "targetFile": "src/index.ts", "command": "npx tsc --noEmit"},
    )

    # ------------------------------------------------------------------
    # 17. write_code for a README
    # ------------------------------------------------------------------
    add(
        _orch_instruction(1, 4),
        _build_prompt("", {
            "ARCHITECTURE": "Express REST API.",
            "PROJECT STATE": (
                "Objective: Add project documentation.\n"
                "Current Status: Server code complete, tests passing, no docs."
            ),
            "KNOWN BUGS": "None.",
        }),
        {"action": "write_code", "targetFile": "README.md", "command": "node --version"},
    )

    # ------------------------------------------------------------------
    # 18. run_tests with npx eslint
    # ------------------------------------------------------------------
    add(
        _orch_instruction(1, 3),
        _build_prompt("", {
            "ARCHITECTURE": "TypeScript with ESLint.",
            "PROJECT STATE": (
                "Objective: Ensure code quality.\n"
                "Current Status: All files written, need linting pass."
            ),
            "KNOWN BUGS": "None.",
        }),
        {"action": "run_tests", "command": "npx eslint src/"},
    )

    # ------------------------------------------------------------------
    # 19. write_code to fix a TypeScript compile error
    # ------------------------------------------------------------------
    add(
        _orch_instruction(2, 1),
        _build_prompt("", {
            "ARCHITECTURE": "Express backend.",
            "PROJECT STATE": (
                "Objective: Fix TypeScript errors in server.ts.\n"
                "Current Status: server.ts has type errors."
            ),
            "KNOWN BUGS": (
                "## Verification failure — 2026-01-15T10:10:00.000Z\n"
                "- Command: `npx tsc --noEmit`\n"
                "- Exit code: 1\n\n"
                "### Diagnostic\n"
                "```text\n"
                "server.ts(12,5): error TS2322: Type 'string' is not assignable to type 'number'.\n"
                "```"
            ),
        }),
        {"action": "write_code", "targetFile": "server.ts", "command": "npx tsc --noEmit"},
    )

    # ------------------------------------------------------------------
    # 20. run_tests for the specific test scenario in the user's project
    # ------------------------------------------------------------------
    add(
        _orch_instruction(1, 2),
        _build_prompt("", {
            "ARCHITECTURE": "Node.js Express backend with Redis and WebSockets.",
            "PROJECT STATE": (
                "Objective: Scaffold a Node.js Express backend. Implement a unified notification "
                "aggregator using Redis for background job processing and WebSockets for real-time delivery.\n"
                "Current Status: Repository initialized and all npm packages installed. "
                "server.ts written.\n\n"
                "## Verified AI change — 2026-07-02T03:50:15.389Z\n"
                "- Target: `server.ts`\n"
                "- Verification: `node --version`\n"
                "- Attempts: 1"
            ),
            "KNOWN BUGS": "None.",
        }),
        {"action": "run_tests", "command": "npx tsc server.ts --noEmit"},
    )

    return dataset


# ===========================================================================
#  CODER DIFF DATA  — must exactly match executeCoder() at inference (existing file)
# ===========================================================================
def _coder_diff_instruction(target_file: str, run_seq: int = 1, iteration: int = 1, attempt: int = 1) -> str:
    """Builds the exact instruction string used by AgentEngine.executeCoder for existing files."""
    return "\n".join([
        "ROLE: Coder.",
        f"RUN_SEQUENCE: {run_seq}; ITERATION: {iteration}; ATTEMPT: {attempt}.",
        f"TARGET_FILE: {target_file}",
        "Generate one valid Unified Diff for only TARGET_FILE.",
        "Delta generation is mandatory. Do not rewrite an entire file longer than 100 lines.",
        "Keep unchanged context lines in each hunk. Never emit shell commands.",
        'Return exactly {"diff":"<JSON-escaped unified diff>"}.',
        "Do not include Markdown fences or additional properties.",
    ])


def generate_coder_data() -> List[Dict[str, Any]]:
    """
    Generates synthetic dataset for the Coder adapter (diff mode).
    Target output is JSON-wrapped unified diffs matching CODER_GBNF.
    """
    dataset: List[Dict[str, Any]] = []

    def add(instruction: str, user_prompt: str, diff: str):
        dataset.append({
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": json.dumps({"diff": diff})},
            ]
        })

    # ------------------------------------------------------------------
    # 1. Change port in Express server
    # ------------------------------------------------------------------
    add(
        _coder_diff_instruction("server.ts"),
        _build_prompt("", {
            "PROJECT STATE": "Objective: Change the port to use env variable.\nCurrent Status: server.ts hardcodes port 3000.",
            "CURRENT TARGET (6 lines)": (
                "import express from 'express';\n"
                "const app = express();\n"
                "\n"
                "app.listen(3000, () => {\n"
                "  console.log('Server running');\n"
                "});"
            ),
            "KNOWN BUGS": "None.",
        }),
        (
            "--- server.ts\n"
            "+++ server.ts\n"
            "@@ -3,4 +3,5 @@\n"
            " const app = express();\n"
            " \n"
            "-app.listen(3000, () => {\n"
            "-  console.log('Server running');\n"
            "+const PORT = process.env.PORT || 8080;\n"
            "+app.listen(PORT, () => {\n"
            "+  console.log(`Server running on port ${PORT}`);\n"
            " });"
        ),
    )

    # ------------------------------------------------------------------
    # 2. Add an import statement
    # ------------------------------------------------------------------
    add(
        _coder_diff_instruction("server.ts"),
        _build_prompt("", {
            "PROJECT STATE": "Objective: Add cors middleware.\nCurrent Status: server.ts needs cors import.",
            "CURRENT TARGET (6 lines)": (
                "import express from 'express';\n"
                "const app = express();\n"
                "\n"
                "app.get('/', (req, res) => res.send('OK'));\n"
                "\n"
                "app.listen(3000);"
            ),
            "KNOWN BUGS": "None.",
        }),
        (
            "--- server.ts\n"
            "+++ server.ts\n"
            "@@ -1,4 +1,6 @@\n"
            " import express from 'express';\n"
            "+import cors from 'cors';\n"
            " const app = express();\n"
            "+app.use(cors());\n"
            " \n"
            " app.get('/', (req, res) => res.send('OK'));"
        ),
    )

    # ------------------------------------------------------------------
    # 3. Fix a type error
    # ------------------------------------------------------------------
    add(
        _coder_diff_instruction("utils.ts", 1, 2, 1),
        _build_prompt("", {
            "PROJECT STATE": "Objective: Fix type error in formatDate.\nCurrent Status: utils.ts has a type error.",
            "CURRENT TARGET (5 lines)": (
                "export function formatDate(date: string): string {\n"
                "  const d = new Date(date);\n"
                "  return d.toISOString;\n"
                "}\n"
                ""
            ),
            "KNOWN BUGS": (
                "## Verification failure\n"
                "- Command: `npx tsc --noEmit`\n"
                "- Exit code: 1\n\n"
                "### Diagnostic\n"
                "```text\n"
                "utils.ts(3,3): error TS2322: Type '() => string' is not assignable to type 'string'.\n"
                "```"
            ),
        }),
        (
            "--- utils.ts\n"
            "+++ utils.ts\n"
            "@@ -1,4 +1,4 @@\n"
            " export function formatDate(date: string): string {\n"
            "   const d = new Date(date);\n"
            "-  return d.toISOString;\n"
            "+  return d.toISOString();\n"
            " }"
        ),
    )

    # ------------------------------------------------------------------
    # 4. Add error handling
    # ------------------------------------------------------------------
    add(
        _coder_diff_instruction("server.ts", 1, 1, 1),
        _build_prompt("", {
            "PROJECT STATE": "Objective: Add error handling middleware.\nCurrent Status: server.ts has no error handler.",
            "CURRENT TARGET (8 lines)": (
                "import express from 'express';\n"
                "const app = express();\n"
                "\n"
                "app.get('/', (req, res) => {\n"
                "  res.json({ status: 'ok' });\n"
                "});\n"
                "\n"
                "app.listen(3000);"
            ),
            "KNOWN BUGS": "None.",
        }),
        (
            "--- server.ts\n"
            "+++ server.ts\n"
            "@@ -6,3 +6,8 @@\n"
            " });\n"
            " \n"
            "+app.use((err: Error, req: express.Request, res: express.Response, next: express.NextFunction) => {\n"
            "+  console.error(err.stack);\n"
            "+  res.status(500).json({ error: 'Internal server error' });\n"
            "+});\n"
            "+\n"
            " app.listen(3000);"
        ),
    )

    # ------------------------------------------------------------------
    # 5. Add a new route
    # ------------------------------------------------------------------
    add(
        _coder_diff_instruction("server.ts", 1, 2, 1),
        _build_prompt("", {
            "PROJECT STATE": "Objective: Add a health check endpoint.\nCurrent Status: server.ts has only the root route.",
            "CURRENT TARGET (8 lines)": (
                "import express from 'express';\n"
                "const app = express();\n"
                "\n"
                "app.get('/', (req, res) => {\n"
                "  res.json({ status: 'ok' });\n"
                "});\n"
                "\n"
                "app.listen(3000);"
            ),
            "KNOWN BUGS": "None.",
        }),
        (
            "--- server.ts\n"
            "+++ server.ts\n"
            "@@ -5,4 +5,8 @@\n"
            "   res.json({ status: 'ok' });\n"
            " });\n"
            " \n"
            "+app.get('/health', (req, res) => {\n"
            "+  res.json({ uptime: process.uptime(), timestamp: Date.now() });\n"
            "+});\n"
            "+\n"
            " app.listen(3000);"
        ),
    )

    # ------------------------------------------------------------------
    # 6. Rename a variable
    # ------------------------------------------------------------------
    add(
        _coder_diff_instruction("config.ts", 1, 1, 1),
        _build_prompt("", {
            "PROJECT STATE": "Objective: Rename PORT constant to SERVER_PORT.\nCurrent Status: config.ts uses PORT.",
            "CURRENT TARGET (3 lines)": (
                "export const PORT = process.env.PORT || 3000;\n"
                "export const HOST = '0.0.0.0';\n"
                "export const NODE_ENV = process.env.NODE_ENV || 'development';"
            ),
            "KNOWN BUGS": "None.",
        }),
        (
            "--- config.ts\n"
            "+++ config.ts\n"
            "@@ -1,3 +1,3 @@\n"
            "-export const PORT = process.env.PORT || 3000;\n"
            "+export const SERVER_PORT = process.env.PORT || 3000;\n"
            " export const HOST = '0.0.0.0';\n"
            " export const NODE_ENV = process.env.NODE_ENV || 'development';"
        ),
    )

    return dataset


# ===========================================================================
#  CODER SCAFFOLD DATA  — must exactly match executeCoder() for new files
# ===========================================================================
def _coder_scaffold_instruction(target_file: str, run_seq: int = 1, iteration: int = 1, attempt: int = 1) -> str:
    """Builds the exact instruction string used by AgentEngine.executeCoder for new files."""
    return "\n".join([
        "ROLE: Coder.",
        f"RUN_SEQUENCE: {run_seq}; ITERATION: {iteration}; ATTEMPT: {attempt}.",
        f"TARGET_FILE: {target_file}",
        "Generate the complete, raw file contents for this new file.",
        'Return exactly {"code":"<JSON-escaped file string>", "explanation":"<short reason>"}',
        "Do not include Markdown fences or additional properties.",
    ])


def generate_scaffold_data() -> List[Dict[str, Any]]:
    """
    Generates synthetic dataset for the Coder adapter's scaffold branch.
    Target output is a full file JSON structure matching SCAFFOLD_GBNF.
    All examples are kept concise to stay well within the 2048-token output limit.
    """
    dataset: List[Dict[str, Any]] = []

    def add(instruction: str, user_prompt: str, code: str, explanation: str):
        dataset.append({
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": json.dumps({"code": code, "explanation": explanation})},
            ]
        })

    # ------------------------------------------------------------------
    # 1. Minimal Express server (matches the user's actual test scenario)
    # ------------------------------------------------------------------
    add(
        _coder_scaffold_instruction("server.ts"),
        _build_prompt("", {
            "PROJECT STATE": (
                "Objective: Scaffold a Node.js Express backend.\n"
                "Current Status: Repository initialized. Please write the initial server.ts file."
            ),
            "ARCHITECTURE": "No architecture specified.",
            "KNOWN BUGS": "None.",
        }),
        (
            "import express from 'express';\n"
            "\n"
            "const app = express();\n"
            "const PORT = process.env.PORT || 3000;\n"
            "\n"
            "app.get('/', (req, res) => {\n"
            "  res.json({ status: 'ok' });\n"
            "});\n"
            "\n"
            "app.listen(PORT, () => {\n"
            "  console.log(`Server running on port ${PORT}`);\n"
            "});\n"
        ),
        "Scaffolded a minimal Express server with a health-check root route.",
    )

    # ------------------------------------------------------------------
    # 2. tsconfig.json
    # ------------------------------------------------------------------
    add(
        _coder_scaffold_instruction("tsconfig.json"),
        _build_prompt("", {
            "PROJECT STATE": (
                "Objective: Initialize TypeScript configuration.\n"
                "Current Status: No tsconfig.json exists."
            ),
            "ARCHITECTURE": "TypeScript project.",
            "KNOWN BUGS": "None.",
        }),
        (
            '{\n'
            '  "compilerOptions": {\n'
            '    "target": "ES2022",\n'
            '    "module": "Node16",\n'
            '    "moduleResolution": "Node16",\n'
            '    "outDir": "./dist",\n'
            '    "rootDir": "./src",\n'
            '    "strict": true,\n'
            '    "esModuleInterop": true,\n'
            '    "skipLibCheck": true,\n'
            '    "forceConsistentCasingInFileNames": true\n'
            '  },\n'
            '  "include": ["src/**/*"],\n'
            '  "exclude": ["node_modules", "dist"]\n'
            '}\n'
        ),
        "Created a standard TypeScript configuration targeting ES2022 with Node16 modules.",
    )

    # ------------------------------------------------------------------
    # 3. Utility module
    # ------------------------------------------------------------------
    add(
        _coder_scaffold_instruction("utils/logger.ts"),
        _build_prompt("", {
            "PROJECT STATE": (
                "Objective: Add a structured logger utility.\n"
                "Current Status: server.ts uses console.log directly."
            ),
            "ARCHITECTURE": "Express backend with utilities in utils/.",
            "KNOWN BUGS": "None.",
        }),
        (
            "type LogLevel = 'info' | 'warn' | 'error';\n"
            "\n"
            "function log(level: LogLevel, message: string, meta?: Record<string, unknown>): void {\n"
            "  const entry = { timestamp: new Date().toISOString(), level, message, ...meta };\n"
            "  console.log(JSON.stringify(entry));\n"
            "}\n"
            "\n"
            "export const logger = {\n"
            "  info: (msg: string, meta?: Record<string, unknown>) => log('info', msg, meta),\n"
            "  warn: (msg: string, meta?: Record<string, unknown>) => log('warn', msg, meta),\n"
            "  error: (msg: string, meta?: Record<string, unknown>) => log('error', msg, meta),\n"
            "};\n"
        ),
        "Created a structured JSON logger with info, warn, and error levels.",
    )

    # ------------------------------------------------------------------
    # 4. Simple test file
    # ------------------------------------------------------------------
    add(
        _coder_scaffold_instruction("calculator.test.ts"),
        _build_prompt("", {
            "PROJECT STATE": (
                "Objective: Add unit tests for calculator module.\n"
                "Current Status: calculator.ts exists with add and multiply functions."
            ),
            "ARCHITECTURE": "Node.js with Jest.",
            "KNOWN BUGS": "None.",
        }),
        (
            "import { add, multiply } from './calculator';\n"
            "\n"
            "describe('calculator', () => {\n"
            "  test('add returns sum', () => {\n"
            "    expect(add(2, 3)).toBe(5);\n"
            "  });\n"
            "\n"
            "  test('multiply returns product', () => {\n"
            "    expect(multiply(4, 5)).toBe(20);\n"
            "  });\n"
            "});\n"
        ),
        "Scaffolded Jest tests for add and multiply functions.",
    )

    # ------------------------------------------------------------------
    # 5. Express middleware
    # ------------------------------------------------------------------
    add(
        _coder_scaffold_instruction("middleware/auth.ts"),
        _build_prompt("", {
            "PROJECT STATE": (
                "Objective: Add authentication middleware.\n"
                "Current Status: server.ts has no auth layer."
            ),
            "ARCHITECTURE": "Express backend with middleware chain.",
            "KNOWN BUGS": "None.",
        }),
        (
            "import { Request, Response, NextFunction } from 'express';\n"
            "\n"
            "export function authMiddleware(req: Request, res: Response, next: NextFunction): void {\n"
            "  const token = req.headers.authorization?.split(' ')[1];\n"
            "  if (!token) {\n"
            "    res.status(401).json({ error: 'Missing authorization token' });\n"
            "    return;\n"
            "  }\n"
            "  next();\n"
            "}\n"
        ),
        "Created a basic Bearer-token auth middleware that checks the Authorization header.",
    )

    # ------------------------------------------------------------------
    # 6. Config module
    # ------------------------------------------------------------------
    add(
        _coder_scaffold_instruction("config.ts"),
        _build_prompt("", {
            "PROJECT STATE": (
                "Objective: Add centralized configuration.\n"
                "Current Status: server.ts references process.env variables directly."
            ),
            "ARCHITECTURE": "Express backend with dotenv.",
            "KNOWN BUGS": "None.",
        }),
        (
            "export const config = {\n"
            "  port: parseInt(process.env.PORT || '3000', 10),\n"
            "  host: process.env.HOST || '0.0.0.0',\n"
            "  nodeEnv: process.env.NODE_ENV || 'development',\n"
            "  logLevel: process.env.LOG_LEVEL || 'info',\n"
            "} as const;\n"
        ),
        "Created a typed configuration object reading from environment variables.",
    )

    return dataset


def main():
    print("Starting dataset preparation for LoRA adapters...")
    tokenizer = load_tokenizer()
    
    orchestrator_data = generate_orchestrator_data()
    coder_data = generate_coder_data()
    scaffold_data = generate_scaffold_data()
    
    # Combine coder diff data and scaffold data for the Coder adapter training
    combined_coder_data = coder_data + scaffold_data
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    verify_and_save(orchestrator_data, "data/orchestrator_dataset.jsonl", tokenizer)
    verify_and_save(combined_coder_data, "data/coder_dataset.jsonl", tokenizer)
    
    print(f"\nDataset summary:")
    print(f"  Orchestrator: {len(orchestrator_data)} examples")
    print(f"  Coder (diff): {len(coder_data)} examples")
    print(f"  Coder (scaffold): {len(scaffold_data)} examples")
    print(f"  Coder (combined): {len(combined_coder_data)} examples")
    print("\nDataset preparation complete.")

if __name__ == "__main__":
    main()
