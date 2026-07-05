import { InferenceGateway } from './InferenceGateway';
import * as fs from 'fs';
import * as path from 'path';

async function runTest() {
    const gateway = new InferenceGateway('http://127.0.0.1:8080');

    console.log("Loading GBNF grammars...");
    const orchestratorGrammar = fs.readFileSync(path.join(__dirname, 'grammars/orchestrator.gbnf'), 'utf-8');
    const coderGrammar = fs.readFileSync(path.join(__dirname, 'grammars/coder.gbnf'), 'utf-8');

    // ====================================================================
    // TEST 1: ORCHESTRATOR 
    // ====================================================================
    console.log("\n[1/2] Swapping to Orchestrator LoRA...");
    await gateway.swapLoRA('orchestrator');
    
    // Simulate what WorkspaceMemory.ts would generate:
    const orchestratorPrompt = `
You are the Orchestrator AI. Given the workspace state (project.md, known_bugs.md), output strict JSON determining the next action. Schema: { 'action': 'write_code' | 'run_tests' | 'done', 'targetFile': string, 'reasoning': string }

# project.md
Task: Build an express server.
Status: Need to write server.ts.
# known_bugs.md
None.
`.trim();

    console.log("Sending prompt to Orchestrator...");
    const orchestratorOutput = await gateway.requestCompletion(orchestratorPrompt, orchestratorGrammar);
    console.log("Orchestrator JSON Result:", orchestratorOutput);


    // ====================================================================
    // TEST 2: CODER
    // ====================================================================
    console.log("\n[2/2] Swapping to Coder LoRA...");
    await gateway.swapLoRA('coder');

    const coderPrompt = `
You are the Coder AI. Given a file and an instruction, output ONLY a unified diff to apply the changes. Do not output any markdown code blocks, explanations, or full file rewrites. Output raw diff only.

File: server.ts
\`\`\`typescript
const x = 1;
\`\`\`
Instruction: change x to 2.
`.trim();

    console.log("Sending prompt to Coder...");
    const coderOutput = await gateway.requestCompletion(coderPrompt, coderGrammar);
    console.log("Coder JSON Result:", coderOutput);
}

runTest().catch(console.error);
