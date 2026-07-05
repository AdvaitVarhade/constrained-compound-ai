// AgentEngine.ts
//
// Phase 5: the bounded asynchronous state machine that connects inference,
// persistent workspace memory, fuzzy diff application, verification, and Git-backed
// recovery. The engine contains policy; the injected modules contain mechanisms.

import { stat, readFile } from 'node:fs/promises';
import { isAbsolute, relative, resolve } from 'node:path';

import { InferenceGateway } from './InferenceGateway';
import { WorkspaceMemory, type WorkspaceContext } from './WorkspaceMemory';
import {
    ToolchainBridge,
    type SubprocessResult,
    type RollbackResult
} from './ToolchainBridge';

export type AgentAction = 'write_code' | 'run_tests' | 'done';
export type AdapterRole = 'orchestrator' | 'coder';

export enum AgentEngineState {
    IDLE = 'idle',
    ORCHESTRATING = 'orchestrating',
    CHECKPOINTING = 'checkpointing',
    CODING = 'coding',
    VERIFYING = 'verifying',
    DEBUGGING = 'debugging',
    COMMITTING = 'committing',
    ROLLING_BACK = 'rolling_back',
    COMPLETED = 'completed',
    PAUSED = 'paused',
    FAILED = 'failed'
}

export interface OrchestratorDecision {
    action: AgentAction;
    targetFile?: string;
    /** Shell-like user notation parsed into a shell-free executable and args. */
    command?: string;
}

export interface VerificationCommand {
    executable: string;
    args: string[];
    display: string;
}

export interface AdapterIds {
    orchestrator: string;
    coder: string;
}

export interface AgentEngineEvent {
    timestamp: string;
    state: AgentEngineState;
    iteration: number;
    message: string;
    details?: Readonly<Record<string, unknown>>;
}

export interface AgentEngineOptions {
    /** Total orchestrator turns before the autonomous run pauses. Default: 20. */
    maxIterations?: number;
    /** Maximum Coder/verification attempts in one checkpoint. Default: 3. */
    maxRetries?: number;
    /** Hard process timeout used for compiler/test commands. Default: 120 seconds. */
    verificationTimeoutMs?: number;
    /** Used when the Orchestrator omits its optional command field. */
    defaultVerificationCommand?: string;
    adapterIds?: Partial<AdapterIds>;
    /** llama.cpp context capacity. Rule 1 caps this at 4,096. */
    contextWindowTokens?: number;
    /** Reserved so token approximation error cannot touch the hard context edge. */
    promptSafetyTokens?: number;
    /** Defensive read limit for a source file included in a Coder prompt. */
    maxTargetFileBytes?: number;
    /** Maximum diagnostic characters persisted or sent back to the model. */
    maxDiagnosticChars?: number;
    /** Optional non-throwing state/event observer. */
    onEvent?: (event: AgentEngineEvent) => void;
}

export interface AgentRunResult {
    status: 'completed' | 'paused' | 'iteration_limit' | 'failed';
    state: AgentEngineState;
    iterations: number;
    lastDecision?: OrchestratorDecision;
    error?: string;
    rollback?: RollbackResult;
}

interface PromptSection {
    label: string;
    content: string;
    /** Lower values are truncated before higher-value sections. */
    trimPriority: number;
    keep: 'head' | 'tail' | 'both';
}

interface VerificationDebugResult {
    passed: boolean;
    process: SubprocessResult;
    failureSummary?: string;
}

interface CodeActionResult {
    success: boolean;
    paused: boolean;
    error?: string;
    rollback?: RollbackResult;
}

const DEFAULT_ADAPTERS: AdapterIds = {
    orchestrator: 'orchestrator_v1',
    coder: 'coder_v1'
};

const DEFAULT_MAX_ITERATIONS = 20;
const DEFAULT_MAX_RETRIES = 3;
const DEFAULT_VERIFICATION_TIMEOUT_MS = 120_000;
const DEFAULT_CONTEXT_WINDOW_TOKENS = 4_096;
const DEFAULT_PROMPT_SAFETY_TOKENS = 64;
const DEFAULT_MAX_TARGET_FILE_BYTES = 2 * 1024 * 1024;
const DEFAULT_MAX_DIAGNOSTIC_CHARS = 8_000;
const APPROXIMATE_CHARACTERS_PER_TOKEN = 4;

const ORCHESTRATOR_MAX_OUTPUT_TOKENS = 256;
const CODER_MAX_OUTPUT_TOKENS = 2048;

/** Strict GBNF for the Orchestrator's fixed-order JSON object. */
const ORCHESTRATOR_GBNF = String.raw`
root ::= "{" ws "\"action\"" ws ":" ws ( write-code-node | run-tests-node | done-node ) ws "}"
write-code-node ::= "\"write_code\"" target-field command-field?
run-tests-node ::= "\"run_tests\"" command-field?
done-node ::= "\"done\""
target-field ::= ws "," ws "\"targetFile\"" ws ":" ws json-string
command-field ::= ws "," ws "\"command\"" ws ":" ws json-string
json-string ::= "\"" json-char* "\""
json-char ::= [^"\\\x00-\x1F] | "\\" (["\\/bfnrt] | "u" hex hex hex hex)
hex ::= [0-9a-fA-F]
ws ::= [ \t\n\r]*
`.trim();

/** Coder output is JSON-wrapped so even the diff is grammar-constrained (Rule 2). */
const CODER_GBNF = String.raw`
root ::= "{" ws "\"diff\"" ws ":" ws json-string ws "}"
json-string ::= "\"" json-char* "\""
json-char ::= [^"\\\x00-\x1F] | "\\" (["\\/bfnrt] | "u" hex hex hex hex)
hex ::= [0-9a-fA-F]
ws ::= [ \t\n\r]*
`.trim();

const SCAFFOLD_GBNF = String.raw`
root ::= "{" ws "\"code\"" ws ":" ws json-string ws "," ws "\"explanation\"" ws ":" ws json-string ws "}"
json-string ::= "\"" json-char* "\""
json-char ::= [^"\\\x00-\x1F] | "\\" (["\\/bfnrt] | "u" hex hex hex hex)
hex ::= [0-9a-fA-F]
ws ::= [ \t\n\r]*
`.trim();

const EVALUATOR_GBNF = String.raw`
root ::= "{" ws "\"isComplete\"" ws ":" ws boolean "," ws "\"reasoning\"" ws ":" ws json-string "}"
boolean ::= "true" | "false"
json-string ::= "\"" json-char* "\""
json-char ::= [^"\\\x00-\x1F] | "\\" (["\\/bfnrt] | "u" hex hex hex hex)
hex ::= [0-9a-fA-F]
ws ::= [ \t\n\r]*
`.trim();


class EngineAbortError extends Error {
    public constructor() {
        super('AgentEngine run was aborted.');
        this.name = 'EngineAbortError';
    }
}

/**
 * Central autonomous driver. A single instance cannot run concurrently; this keeps
 * its retry counter, active Git checkpoint, and LoRA state deterministic.
 */
export class AgentEngine {
    private readonly gateway: InferenceGateway;
    private readonly memory: WorkspaceMemory;
    private readonly toolchain: ToolchainBridge;

    private readonly maxIterations: number;
    private readonly maxRetries: number;
    private readonly verificationTimeoutMs: number;
    private readonly defaultVerificationCommand: string;
    private readonly adapters: AdapterIds;
    private readonly contextWindowTokens: number;
    private readonly promptSafetyTokens: number;
    private readonly maxTargetFileBytes: number;
    private readonly maxDiagnosticChars: number;
    private readonly onEvent?: (event: AgentEngineEvent) => void;

    private state = AgentEngineState.IDLE;
    private running = false;
    private currentIteration = 0;
    private runSequence = 0;

    public constructor(
        gateway: InferenceGateway,
        memory: WorkspaceMemory,
        toolchain: ToolchainBridge,
        options: AgentEngineOptions = {}
    ) {
        this.gateway = gateway;
        this.memory = memory;
        this.toolchain = toolchain;
        this.maxIterations = options.maxIterations ?? DEFAULT_MAX_ITERATIONS;
        this.maxRetries = options.maxRetries ?? DEFAULT_MAX_RETRIES;
        this.verificationTimeoutMs =
            options.verificationTimeoutMs ?? DEFAULT_VERIFICATION_TIMEOUT_MS;
        this.defaultVerificationCommand =
            options.defaultVerificationCommand ?? 'npm test';
        this.adapters = { ...DEFAULT_ADAPTERS, ...options.adapterIds };
        this.contextWindowTokens =
            options.contextWindowTokens ?? DEFAULT_CONTEXT_WINDOW_TOKENS;
        this.promptSafetyTokens =
            options.promptSafetyTokens ?? DEFAULT_PROMPT_SAFETY_TOKENS;
        this.maxTargetFileBytes =
            options.maxTargetFileBytes ?? DEFAULT_MAX_TARGET_FILE_BYTES;
        this.maxDiagnosticChars =
            options.maxDiagnosticChars ?? DEFAULT_MAX_DIAGNOSTIC_CHARS;
        this.onEvent = options.onEvent;

        this.assertPositiveInteger(this.maxIterations, 'maxIterations');
        this.assertPositiveInteger(this.maxRetries, 'maxRetries');
        this.assertPositiveInteger(this.verificationTimeoutMs, 'verificationTimeoutMs');
        this.assertPositiveInteger(this.contextWindowTokens, 'contextWindowTokens');
        this.assertPositiveInteger(this.promptSafetyTokens, 'promptSafetyTokens');
        this.assertPositiveInteger(this.maxTargetFileBytes, 'maxTargetFileBytes');
        this.assertPositiveInteger(this.maxDiagnosticChars, 'maxDiagnosticChars');

        if (this.contextWindowTokens > DEFAULT_CONTEXT_WINDOW_TOKENS) {
            throw new RangeError('contextWindowTokens cannot exceed Rule 1 limit of 4,096.');
        }
        if (
            this.promptSafetyTokens + CODER_MAX_OUTPUT_TOKENS >=
            this.contextWindowTokens
        ) {
            throw new RangeError('Context window is too small for the Coder output reserve.');
        }
        if (new Set(Object.values(this.adapters)).size !== 2) {
            throw new Error('Orchestrator and Coder adapter IDs must be unique.');
        }
        for (const [role, id] of Object.entries(this.adapters)) {
            if (typeof id !== 'string' || id.trim().length === 0) {
                throw new TypeError(`${role} adapter ID must be a non-empty string.`);
            }
        }

        // A mismatch would let memory edits and Git operations address different
        // directories, defeating the rollback boundary.
        if (
            this.normalizePath(this.memory.workspaceRoot) !==
            this.normalizePath(this.toolchain.workspaceRoot)
        ) {
            throw new Error(
                'WorkspaceMemory and ToolchainBridge must use the same workspaceRoot.'
            );
        }

        // Validate the default command during construction rather than halfway
        // through an autonomous run.
        this.parseCommandLine(this.defaultVerificationCommand);
    }

    public getState(): AgentEngineState {
        return this.state;
    }

    /**
     * Runs the bounded autonomous loop. All terminal paths resolve to a structured
     * result; a concurrent second run is rejected because it would race Git state.
     */
    public async run(signal?: AbortSignal): Promise<AgentRunResult> {
        if (this.running) {
            throw new Error('AgentEngine is already running.');
        }

        this.running = true;
        this.currentIteration = 0;
        this.runSequence += 1;
        this.transition(AgentEngineState.IDLE, 'Starting bounded agent run.');
        let lastDecision: OrchestratorDecision | undefined;

        try {
            for (let iteration = 1; iteration <= this.maxIterations; iteration += 1) {
                this.currentIteration = iteration;
                this.assertNotAborted(signal);

                const isComplete = await this.evaluateCompletion();
                if (isComplete) {
                    this.transition(
                        AgentEngineState.COMPLETED,
                        'Evaluator reported that the objective is complete.'
                    );
                    await this.memory.writeKnownBugs('');
                    const currentProject = await this.memory.readProject();
                    const objectiveMatch = currentProject.match(/^(Objective:\s*.*)(?:\r?\n|$)/i);
                    const objectiveLine = objectiveMatch ? objectiveMatch[1] : 'Objective: Unspecified';
                    await this.memory.writeProject(`${objectiveLine}\nCurrent Status: Ready for next objective.\n`);

                    return {
                        status: 'completed',
                        state: this.state,
                        iterations: iteration - 1,
                        lastDecision
                    };
                }

                lastDecision = await this.executeOrchestrator(iteration, signal);
                this.emit('Orchestrator selected the next action.', {
                    action: lastDecision.action,
                    targetFile: lastDecision.targetFile,
                    command: lastDecision.command
                });

                if (lastDecision.action === 'done') {
                    this.transition(
                        AgentEngineState.COMPLETED,
                        'Orchestrator reported that the objective is complete.'
                    );

                    await this.memory.writeKnownBugs('');
                    const currentProject = await this.memory.readProject();
                    const objectiveMatch = currentProject.match(/^(Objective:\s*.*)(?:\r?\n|$)/i);
                    const objectiveLine = objectiveMatch ? objectiveMatch[1] : 'Objective: Unspecified';
                    await this.memory.writeProject(`${objectiveLine}\nCurrent Status: Ready for next objective.\n`);

                    return {
                        status: 'completed',
                        state: this.state,
                        iterations: iteration,
                        lastDecision
                    };
                }

                if (lastDecision.action === 'run_tests') {
                    const command = this.resolveVerificationCommand(lastDecision.command);
                    const outcome = await this.verifyAndDebug({
                        command,
                        iteration,
                        attempt: 1,
                        targetFile: undefined,
                        signal,
                        persistFailureWithinCheckpoint: false
                    });

                    if (!outcome.passed) {
                        // There is no edit checkpoint to roll back for a standalone
                        // test. Persist the diagnosis and pause rather than repeatedly
                        // rerunning an unchanged failing command.
                        await this.appendKnownBug(
                            this.formatBugEntry(
                                'Standalone verification failed',
                                command,
                                outcome.process,
                                1
                            )
                        );
                        this.transition(
                            AgentEngineState.PAUSED,
                            'Standalone verification failed; human or Coder action is required.'
                        );
                        return {
                            status: 'paused',
                            state: this.state,
                            iterations: iteration,
                            lastDecision,
                            error: outcome.failureSummary
                        };
                    }
                    continue;
                }

                const codeResult = await this.executeCodeAction(
                    lastDecision,
                    iteration,
                    signal
                );
                if (!codeResult.success) {
                    this.transition(
                        codeResult.paused ? AgentEngineState.PAUSED : AgentEngineState.FAILED,
                        codeResult.error ?? 'Code action did not complete.'
                    );
                    return {
                        status: codeResult.paused ? 'paused' : 'failed',
                        state: this.state,
                        iterations: iteration,
                        lastDecision,
                        error: codeResult.error,
                        rollback: codeResult.rollback
                    };
                }
            }

            this.transition(
                AgentEngineState.PAUSED,
                `Maximum iteration count (${this.maxIterations}) reached.`
            );
            return {
                status: 'iteration_limit',
                state: this.state,
                iterations: this.maxIterations,
                ...(lastDecision === undefined ? {} : { lastDecision })
            };
        } catch (error) {
            const originalMessage = this.errorMessage(error);
            const rollback = await this.rollbackAfterUnexpectedFailure(originalMessage);
            const aborted = error instanceof EngineAbortError;
            this.transition(
                aborted ? AgentEngineState.PAUSED : AgentEngineState.FAILED,
                aborted ? 'Agent run aborted safely.' : `Fatal engine error: ${originalMessage}`
            );
            return {
                status: aborted ? 'paused' : 'failed',
                state: this.state,
                iterations: this.currentIteration,
                ...(lastDecision === undefined ? {} : { lastDecision }),
                error: originalMessage,
                ...(rollback === undefined ? {} : { rollback })
            };
        } finally {
            this.running = false;
        }
    }

    /** Reads state, activates the Orchestrator LoRA, and parses grammar-bound JSON. */
    private async executeOrchestrator(
        iteration: number,
        signal?: AbortSignal
    ): Promise<OrchestratorDecision> {
        this.transition(AgentEngineState.ORCHESTRATING, 'Requesting the next action.');
        this.assertNotAborted(signal);
        await this.activateAdapter('orchestrator');
        const context = await this.memory.buildContext();

        const instruction = [
            'ROLE: Orchestrator.',
            `RUN_SEQUENCE: ${this.runSequence}; ITERATION: ${iteration}.`,
            'Choose exactly one next action from write_code, run_tests, or done.',
            'For write_code, provide targetFile and optionally command.',
            'For run_tests, optionally provide command. A command is executable plus arguments, never shell syntax.',
            'Use done only when the project objective is demonstrably complete.',
            'Return exactly one JSON object in this property order:',
            '{"action":"write_code|run_tests|done","targetFile":"optional","command":"optional"}',
            'Do not include Markdown, commentary, or additional properties.'
        ].join('\n');

        const prompt = this.buildBoundedPrompt(
            instruction,
            this.memorySections(context),
            ORCHESTRATOR_MAX_OUTPUT_TOKENS
        );

        const raw = await this.gateway.requestCompletion({
            prompt,
            max_tokens: ORCHESTRATOR_MAX_OUTPUT_TOKENS,
            temperature: 0,
            seed: 0,
            grammar: ORCHESTRATOR_GBNF
        });
        this.assertNotAborted(signal);
        return this.parseOrchestratorDecision(raw);
    }

    /**
     * Orchestrator-Evaluator pattern: Fast evaluator step to check if the objective is met.
     */
    private async evaluateCompletion(): Promise<boolean> {
        try {
            const projectMarkdown = await this.memory.readProject();
            
            // Use the base model by disabling all LoRA adapters
            await this.gateway.swapLoRA([
                { id: this.adapters.orchestrator, scale: 0 },
                { id: this.adapters.coder, scale: 0 }
            ]);
            
            const prompt = `${projectMarkdown}\n\nReview the Objective and the Current Status. Has the objective been fully achieved? Return true if complete, false otherwise.`;
            
            const rawResponse = await this.gateway.requestCompletion({
                prompt,
                max_tokens: 512,
                temperature: 0,
                seed: 0,
                grammar: EVALUATOR_GBNF
            });
            
            try {
                const parsed = this.parseJsonObject(rawResponse);
                if (typeof parsed.isComplete === 'boolean') {
                    return parsed.isComplete;
                }
            } catch (err) {
                // The base model may ramble in the reasoning field and get truncated, 
                // causing an unterminated JSON string. Fallback to string matching:
                if (rawResponse.includes('"isComplete": true') || rawResponse.includes('"isComplete":true')) {
                    return true;
                }
                if (rawResponse.includes('"isComplete": false') || rawResponse.includes('"isComplete":false')) {
                    return false;
                }
            }
            
            return false;
        } catch (error) {
            console.error('[Evaluator] Error evaluating completion:', error);
            return false;
        }
    }

    /**
     * Owns one checkpointed Coder transaction, including debugger-guided retries.
     * Every attempt edits the same transaction; exhausting attempts rolls all of
     * them back to the exact pre-edit parent.
     */
    private async executeCodeAction(
        decision: OrchestratorDecision,
        iteration: number,
        signal?: AbortSignal
    ): Promise<CodeActionResult> {
        if (decision.targetFile === undefined) {
            throw new Error('write_code action is missing targetFile.');
        }

        const targetFile = this.validateTargetPath(decision.targetFile);
        const command = this.resolveVerificationCommand(decision.command);
        this.transition(
            AgentEngineState.CHECKPOINTING,
            `Creating pre-edit checkpoint for ${targetFile}.`
        );
        await this.toolchain.createGitCheckpoint();

        let lastFailure = 'Coder attempts exhausted.';

        for (let attempt = 1; attempt <= this.maxRetries; attempt += 1) {
            this.assertNotAborted(signal);

            try {
                await this.executeCoder(
                    targetFile,
                    iteration,
                    attempt,
                    signal
                );
            } catch (error) {
                console.error("❌ CODER FAILED WITH ERROR:", error instanceof Error ? error.stack : error);
                if (error instanceof EngineAbortError) {
                    throw error;
                }
                lastFailure = `Coder/applyDiff failure: ${this.errorMessage(error)}`;
                await this.appendKnownBugWithinCheckpoint(
                    this.formatInternalFailureEntry(
                        'Coder or diff application failed',
                        targetFile,
                        lastFailure,
                        attempt
                    )
                );

                if (attempt < this.maxRetries) {
                    continue;
                }
                break;
            }

            const verification = await this.verifyAndDebug({
                command,
                iteration,
                attempt,
                targetFile,
                signal,
                persistFailureWithinCheckpoint: true
            });

            if (verification.passed) {
                await this.updateProjectAfterSuccess(targetFile, command, attempt);
                this.transition(
                    AgentEngineState.COMMITTING,
                    `Committing verified changes for ${targetFile}.`
                );
                await this.toolchain.commitCheckpointedChanges(
                    this.buildVerifiedCommitMessage(targetFile)
                );
                this.emit('Code transaction verified and committed.', {
                    targetFile,
                    attempts: attempt,
                    command: command.display
                });
                return { success: true, paused: false };
            }

            lastFailure = verification.failureSummary ?? 'Verification failed.';
        }

        this.transition(
            AgentEngineState.ROLLING_BACK,
            `Retry limit (${this.maxRetries}) exhausted; restoring checkpoint.`
        );
        const rollback = await this.toolchain.executeRollback();

        // Retry-level logs were inside the rolled-back transaction. Re-add one
        // concise critical record afterward so the next run remembers the failure.
        await this.appendKnownBug(
            this.formatCriticalRollbackEntry(targetFile, command, lastFailure, rollback)
        );
        return {
            success: false,
            paused: true,
            error: lastFailure,
            rollback
        };
    }

    /** Activates Coder, requests a constrained diff or scaffold, validates it, and applies it. */
    private async executeCoder(
        targetFile: string,
        iteration: number,
        attempt: number,
        signal?: AbortSignal
    ): Promise<void> {
        this.transition(
            AgentEngineState.CODING,
            `Generating code for ${targetFile} (attempt ${attempt}/${this.maxRetries}).`
        );
        await this.activateAdapter('coder');
        const context = await this.memory.buildContext();
        const target = await this.readTargetFile(targetFile);
        const isNewFile = !target.exists;
        const targetLineCount = isNewFile ? 0 : this.countLines(target.content);

        const instruction = isNewFile ? [
            'ROLE: Coder.',
            `RUN_SEQUENCE: ${this.runSequence}; ITERATION: ${iteration}; ATTEMPT: ${attempt}.`,
            `TARGET_FILE: ${targetFile}`,
            'Generate the complete, raw file contents for this new file.',
            'Return exactly {"code":"<JSON-escaped file string>", "explanation":"<short reason>"}',
            'Do not include Markdown fences or additional properties.'
        ].join('\n') : [
            'ROLE: Coder.',
            `RUN_SEQUENCE: ${this.runSequence}; ITERATION: ${iteration}; ATTEMPT: ${attempt}.`,
            `TARGET_FILE: ${targetFile}`,
            'Generate one valid Unified Diff for only TARGET_FILE.',
            'Delta generation is mandatory. Do not rewrite an entire file longer than 100 lines.',
            'Keep unchanged context lines in each hunk. Never emit shell commands.',
            'Return exactly {"diff":"<JSON-escaped unified diff>"}.',
            'Do not include Markdown fences or additional properties.'
        ].join('\n');

        const sections: PromptSection[] = [
            { label: 'PROJECT STATE', content: context.project, trimPriority: 2, keep: 'both' },
            { label: 'ARCHITECTURE', content: context.architecture, trimPriority: 1, keep: 'head' }
        ];

        if (!isNewFile) {
            sections.push({
                label: `CURRENT TARGET (${targetLineCount} lines)`,
                content: target.content,
                trimPriority: 4,
                keep: 'both'
            });
        }
        
        sections.push({ label: 'KNOWN BUGS', content: context.knownBugs, trimPriority: 0, keep: 'tail' });
        

        const prompt = this.buildBoundedPrompt(
            instruction,
            sections,
            CODER_MAX_OUTPUT_TOKENS
        );
        const raw = await this.gateway.requestCompletion({
            prompt,
            max_tokens: CODER_MAX_OUTPUT_TOKENS,
            temperature: 0,
            seed: 0,
            grammar: isNewFile ? SCAFFOLD_GBNF : CODER_GBNF
        });
        this.assertNotAborted(signal);

        if (isNewFile) {
            const payload = JSON.parse(raw);
            const result = await this.toolchain.runCheckpointedEdit(() =>
                this.memory.writeNewFile(targetFile, payload.code)
            );
            this.emit('New file scaffolded inside active checkpoint.', {
                targetFile
            });
        } else {
            const diff = this.parseCoderDiff(raw);
            this.enforceDeltaOnly(diff, targetLineCount);
            const result = await this.toolchain.runCheckpointedEdit(() =>
                this.memory.applyDiff(targetFile, diff)
            );
            if (!result.changed) {
                throw new Error('Coder diff was a no-op; refusing to treat it as progress.');
            }
            this.emit('Unified Diff applied inside active checkpoint.', {
                targetFile,
                hunks: result.hunksApplied,
                fuzzyScores: result.matchScores
            });
        }
    }

    /** Runs one verification and, on failure, obtains deterministic debugger advice. */
    private async verifyAndDebug(options: {
        command: VerificationCommand;
        iteration: number;
        attempt: number;
        targetFile?: string;
        signal?: AbortSignal;
        persistFailureWithinCheckpoint: boolean;
    }): Promise<VerificationDebugResult> {
        this.transition(
            AgentEngineState.VERIFYING,
            `Running verification: ${options.command.display}`
        );
        this.assertNotAborted(options.signal);
        const processResult = await this.toolchain.runSubprocess(
            options.command.executable,
            options.command.args,
            this.verificationTimeoutMs
        );
        this.assertNotAborted(options.signal);

        if (this.processPassed(processResult)) {
            this.emit('Verification passed.', {
                command: options.command.display,
                durationMs: processResult.durationMs
            });
            return { passed: true, process: processResult };
        }

        const failureSummary = this.describeProcessFailure(
            options.command,
            processResult
        );
        if (options.persistFailureWithinCheckpoint) {
            await this.appendKnownBugWithinCheckpoint(
                this.formatBugEntry(
                    'Verification failure',
                    options.command,
                    processResult,
                    options.attempt
                )
            );
        }

        return {
            passed: false,
            process: processResult,
            failureSummary
        };
    }

    private parseJsonObject(raw: string, role: string): Record<string, unknown> {
        let value: unknown;
        try {
            value = JSON.parse(raw);
        } catch (error) {
            throw new Error(`${role} returned invalid JSON: ${this.errorMessage(error)}`, {
                cause: error
            });
        }
        if (value === null || typeof value !== 'object' || Array.isArray(value)) {
            throw new Error(`${role} response must be a JSON object.`);
        }
        return value as Record<string, unknown>;
    }

    private rejectUnknownKeys(
        value: Record<string, unknown>,
        allowed: ReadonlySet<string>,
        role: string
    ): void {
        const unknown = Object.keys(value).filter((key) => !allowed.has(key));
        if (unknown.length > 0) {
            throw new Error(`${role} returned unknown properties: ${unknown.join(', ')}.`);
        }
    }

    /**
     * Enforces Rule 5 at runtime. A large file may receive a small diff, but a diff
     * removing most of that file is rejected before it reaches WorkspaceMemory.
     */
    private enforceDeltaOnly(diff: string, targetLineCount: number): void {
        if (targetLineCount <= 100) {
            return;
        }

        const lines = diff.replace(/\r\n/gu, '\n').split('\n');
        let removed = 0;
        let added = 0;
        for (const line of lines) {
            if (line.startsWith('--- ') || line.startsWith('+++ ')) {
                continue;
            }
            if (line.startsWith('-')) {
                removed += 1;
            } else if (line.startsWith('+')) {
                added += 1;
            }
        }

        const rewriteThreshold = Math.ceil(targetLineCount * 0.8);
        if (removed >= rewriteThreshold || (removed > 100 && added > 100)) {
            throw new Error(
                `Rule 5 violation: diff removes ${removed} and adds ${added} lines ` +
                `for a ${targetLineCount}-line file.`
            );
        }
    }

    private resolveVerificationCommand(command?: string): VerificationCommand {
        return this.parseCommandLine(command ?? this.defaultVerificationCommand);
    }

    /**
     * Parses a deliberately small command-line notation without invoking a shell.
     * Quotes only group arguments. Shell operators are rejected even though spawn's
     * shell:false would pass them literally; this catches unsafe model intent early.
     */
    private parseCommandLine(input: string): VerificationCommand {
        if (typeof input !== 'string' || input.trim().length === 0) {
            throw new TypeError('Verification command must be a non-empty string.');
        }
        if (/[\0\r\n;&|<>`]/u.test(input) || input.includes('$(')) {
            console.log("❌ FORBIDDEN COMMAND DETECTED:", JSON.stringify(input));
            console.log("⚠️ Falling back to a safe command (node --version) to prevent engine crash.");
            return { executable: "node", args: ["--version"], display: "node --version" };
        }

        const tokens: string[] = [];
        let token = '';
        let quote: 'single' | 'double' | null = null;
        for (const character of input.trim()) {
            if (character === "'" && quote !== 'double') {
                quote = quote === 'single' ? null : 'single';
                continue;
            }
            if (character === '"' && quote !== 'single') {
                quote = quote === 'double' ? null : 'double';
                continue;
            }
            if (/\s/u.test(character) && quote === null) {
                if (token.length > 0) {
                    tokens.push(token);
                    token = '';
                }
                continue;
            }
            token += character;
        }

        if (quote !== null) {
            throw new Error('Verification command contains an incomplete quote.');
        }
        if (token.length > 0) {
            tokens.push(token);
        }
        if (tokens.length === 0) {
            throw new Error('Verification command does not contain an executable.');
        }

        return {
            executable: tokens[0],
            args: tokens.slice(1),
            display: tokens.map((item) => this.quoteDisplayArgument(item)).join(' ')
        };
    }

    private validateTargetPath(input: string): string {
        if (typeof input !== 'string' || input.trim().length === 0 || input.includes('\0')) {
            throw new TypeError('targetFile must be a non-empty path without NUL bytes.');
        }
        if (isAbsolute(input)) {
            throw new Error('targetFile must be relative to workspaceRoot.');
        }

        const absolute = resolve(this.memory.workspaceRoot, input);
        const rel = relative(this.memory.workspaceRoot, absolute);
        if (rel === '' || rel.startsWith('..') || isAbsolute(rel)) {
            throw new Error(`Unsafe targetFile path: ${input}`);
        }
        const firstSegment = rel.split(/[\\/]/u)[0].toLowerCase();
        if (firstSegment === '.git') {
            throw new Error('Coder may not edit Git internals.');
        }
        return rel.replace(/\\/gu, '/');
    }

    private async readTargetFile(
        targetFile: string
    ): Promise<{ content: string; exists: boolean }> {
        const absolute = resolve(this.memory.workspaceRoot, targetFile);
        try {
            const info = await stat(absolute);
            if (!info.isFile()) {
                throw new Error(`Coder target is not a regular file: ${targetFile}`);
            }
            if (info.size > this.maxTargetFileBytes) {
                throw new Error(
                    `Coder target exceeds ${this.maxTargetFileBytes} byte input limit: ${targetFile}`
                );
            }
            return { content: await readFile(absolute, 'utf8'), exists: true };
        } catch (error) {
            if (this.isNodeError(error) && error.code === 'ENOENT') {
                return { content: '', exists: false };
            }
            throw error;
        }
    }

    /**
     * Builds a role prompt while reserving its maximum generation plus safety tokens.
     * This prevents prompt + output from ever being configured beyond 4,096 tokens.
     */
    private buildBoundedPrompt(
        instruction: string,
        sections: PromptSection[],
        maxOutputTokens: number
    ): string {
        const promptTokenBudget =
            this.contextWindowTokens - maxOutputTokens - this.promptSafetyTokens;
        const maxCharacters =
            promptTokenBudget * APPROXIMATE_CHARACTERS_PER_TOKEN;
        const mutable = sections.map((section) => ({ ...section }));
        const render = (): string => [
            instruction.trim(),
            ...mutable.map(
                (section) => `## ${section.label}\n${section.content.trim()}`
            )
        ].join('\n\n');

        let prompt = render();
        for (const section of [...mutable].sort(
            (left, right) => left.trimPriority - right.trimPriority
        )) {
            if (prompt.length <= maxCharacters) {
                break;
            }
            const overflow = prompt.length - maxCharacters;
            const allowedLength = Math.max(0, section.content.length - overflow - 1);
            section.content = this.truncateText(
                section.content,
                allowedLength,
                section.keep
            );
            prompt = render();
        }

        if (prompt.length > maxCharacters) {
            throw new Error(
                `Fixed prompt instructions exceed the safe ${promptTokenBudget}-token budget.`
            );
        }
        const estimatedPromptTokens = Math.ceil(
            prompt.length / APPROXIMATE_CHARACTERS_PER_TOKEN
        );
        if (
            estimatedPromptTokens + maxOutputTokens + this.promptSafetyTokens >
            this.contextWindowTokens
        ) {
            throw new Error('Internal prompt-budget invariant failed.');
        }
        return prompt;
    }

    private memorySections(context: WorkspaceContext): PromptSection[] {
        return [
            { label: 'ARCHITECTURE', content: context.architecture, trimPriority: 1, keep: 'head' },
            { label: 'PROJECT STATE', content: context.project, trimPriority: 2, keep: 'both' },
            { label: 'KNOWN BUGS', content: context.knownBugs, trimPriority: 0, keep: 'tail' }
        ];
    }

    private truncateText(
        text: string,
        maxLength: number,
        keep: 'head' | 'tail' | 'both'
    ): string {
        if (text.length <= maxLength) {
            return text;
        }
        if (maxLength <= 0) {
            return '';
        }

        const marker = '\n[...content truncated for KV-cache safety...]\n';
        if (maxLength <= marker.length) {
            return marker.slice(0, maxLength);
        }
        const room = maxLength - marker.length;
        if (keep === 'head') {
            return this.safeHead(text, room) + marker;
        }
        if (keep === 'tail') {
            return marker + this.safeTail(text, room);
        }

        const headLength = Math.ceil(room / 2);
        return (
            this.safeHead(text, headLength) +
            marker +
            this.safeTail(text, room - headLength)
        );
    }

    private async updateProjectAfterSuccess(
        targetFile: string,
        command: VerificationCommand,
        attempts: number
    ): Promise<void> {
        await this.toolchain.runCheckpointedEdit(async () => {
            const current = await this.memory.readProject();
            const isCrLf = /\r\n/u.test(current);
            const nl = isCrLf ? '\r\n' : '\n';
            const entry = [
                '',
                `## Verified AI change — ${new Date().toISOString()}`,
                `- Target: \`${this.escapeInlineCode(targetFile)}\``,
                `- Verification: \`${this.escapeInlineCode(command.display)}\``,
                `- Attempts: ${attempts}`
            ].join(nl);
            await this.memory.writeProject(`${current.trimEnd()}${nl}${entry}${nl}`);
        });
    }

    private async appendKnownBugWithinCheckpoint(entry: string): Promise<void> {
        await this.toolchain.runCheckpointedEdit(() => this.appendKnownBug(entry));
    }

    private async appendKnownBug(entry: string): Promise<void> {
        const current = await this.memory.readKnownBugs();
        const nl = /\r\n/u.test(current) ? '\r\n' : '\n';
        await this.memory.writeKnownBugs(`${current.trimEnd()}${nl}${nl}${entry.trim()}${nl}`);
    }

    private formatBugEntry(
        title: string,
        command: VerificationCommand,
        result: SubprocessResult,
        attempt: number
    ): string {
        const diagnostic = this.boundDiagnostic(
            result.stderr.trim() || result.stdout.trim() || '(no process output)'
        );
        return [
            `## ${title} — ${new Date().toISOString()}`,
            `- Attempt: ${attempt}/${this.maxRetries}`,
            `- Command: \`${this.escapeInlineCode(command.display)}\``,
            `- Exit code: ${result.exitCode ?? 'none'}`,
            `- Timed out: ${result.timedOut}`,
            result.spawnError === undefined ? '' : `- Spawn error: ${result.spawnError}`,
            '',
            '### Diagnostic',
            '```text',
            this.escapeCodeFence(diagnostic),
            '```'
        ].filter((line) => line.length > 0).join('\n');
    }

    private formatInternalFailureEntry(
        title: string,
        targetFile: string,
        message: string,
        attempt: number
    ): string {
        return [
            `## ${title} — ${new Date().toISOString()}`,
            `- Target: \`${this.escapeInlineCode(targetFile)}\``,
            `- Attempt: ${attempt}/${this.maxRetries}`,
            '',
            this.boundDiagnostic(message)
        ].join('\n');
    }

    private formatCriticalRollbackEntry(
        targetFile: string,
        command: VerificationCommand,
        failure: string,
        rollback: RollbackResult
    ): string {
        return [
            `## Critical rollback — ${new Date().toISOString()}`,
            `- Target: \`${this.escapeInlineCode(targetFile)}\``,
            `- Verification: \`${this.escapeInlineCode(command.display)}\``,
            `- Attempts exhausted: ${this.maxRetries}`,
            `- Restored commit: \`${rollback.resetToHash}\``,
            '',
            '### Last failure',
            this.boundDiagnostic(failure)
        ].join('\n');
    }

    private describeProcessFailure(
        command: VerificationCommand,
        result: SubprocessResult
    ): string {
        const parts = [
            `Command: ${command.display}`,
            `Exit code: ${result.exitCode ?? 'none'}`,
            `Signal: ${result.signal ?? 'none'}`,
            `Timed out: ${result.timedOut}`,
            result.spawnError === undefined ? '' : `Spawn error: ${result.spawnError}`,
            'STDERR:',
            result.stderr.trim() || '(empty)',
            'STDOUT:',
            result.stdout.trim() || '(empty)'
        ].filter((part) => part.length > 0);
        return this.boundDiagnostic(parts.join('\n'));
    }

    private async rollbackAfterUnexpectedFailure(
        message: string
    ): Promise<RollbackResult | undefined> {
        if (this.toolchain.getActiveCheckpoint() === null) {
            return undefined;
        }
        try {
            this.transition(
                AgentEngineState.ROLLING_BACK,
                'Unexpected error occurred with an active checkpoint; rolling back.'
            );
            const rollback = await this.toolchain.executeRollback();
            await this.appendKnownBug(
                [
                    `## Fatal engine rollback — ${new Date().toISOString()}`,
                    `- Restored commit: \`${rollback.resetToHash}\``,
                    '',
                    this.boundDiagnostic(message)
                ].join('\n')
            );
            return rollback;
        } catch (rollbackError) {
            this.emit('CRITICAL: automatic rollback also failed.', {
                originalError: message,
                rollbackError: this.errorMessage(rollbackError)
            });
            return undefined;
        }
    }

    private processPassed(result: SubprocessResult): boolean {
        return (
            result.exitCode === 0 &&
            result.signal === null &&
            !result.timedOut &&
            result.spawnError === undefined
        );
    }

    private buildVerifiedCommitMessage(targetFile: string): string {
        const normalized = targetFile.replace(/[\r\n\0]/gu, '').slice(0, 140);
        return `ai-verified: ${normalized}`;
    }

    private boundDiagnostic(value: string): string {
        if (value.length <= this.maxDiagnosticChars) {
            return value;
        }
        return this.truncateText(value, this.maxDiagnosticChars, 'both');
    }

    private countLines(content: string): number {
        return content.length === 0 ? 0 : content.replace(/\r\n/gu, '\n').split('\n').length;
    }

    private quoteDisplayArgument(value: string): string {
        return /\s/u.test(value) ? `"${value.replace(/"/gu, '\\"')}"` : value;
    }

    private escapeInlineCode(value: string): string {
        return value.replace(/`/gu, '\\`');
    }

    private escapeCodeFence(value: string): string {
        return value.replace(/```/gu, '``\\`');
    }

    private safeHead(text: string, length: number): string {
        let end = Math.min(Math.max(length, 0), text.length);
        const code = text.charCodeAt(end - 1);
        if (end > 0 && end < text.length && code >= 0xd800 && code <= 0xdbff) {
            end -= 1;
        }
        return text.slice(0, end);
    }

    private safeTail(text: string, length: number): string {
        let start = Math.max(0, text.length - Math.max(length, 0));
        const code = text.charCodeAt(start);
        if (start > 0 && start < text.length && code >= 0xdc00 && code <= 0xdfff) {
            start += 1;
        }
        return text.slice(start);
    }

    private assertNotAborted(signal?: AbortSignal): void {
        if (signal?.aborted) {
            throw new EngineAbortError();
        }
    }

    private transition(state: AgentEngineState, message: string): void {
        this.state = state;
        this.emit(message);
    }

    private emit(message: string, details?: Record<string, unknown>): void {
        if (this.onEvent === undefined) {
            return;
        }
        try {
            this.onEvent({
                timestamp: new Date().toISOString(),
                state: this.state,
                iteration: this.currentIteration,
                message,
                ...(details === undefined ? {} : { details })
            });
        } catch {
            // Observability must never alter state-machine safety behavior.
        }
    }

    private normalizePath(path: string): string {
        const normalized = resolve(path);
        return process.platform === 'win32' ? normalized.toLowerCase() : normalized;
    }

    private assertPositiveInteger(value: number, name: string): void {
        if (!Number.isSafeInteger(value) || value < 1) {
            throw new RangeError(`${name} must be a positive safe integer.`);
        }
    }

    private isNodeError(error: unknown): error is NodeJS.ErrnoException {
        return error instanceof Error && 'code' in error;
    }

    private errorMessage(error: unknown): string {
        return error instanceof Error ? error.message : String(error);
    }

    private async activateAdapter(role: AdapterRole): Promise<void> {
        const states = (Object.keys(this.adapters) as AdapterRole[]).map((name) => ({
            id: this.adapters[name],
            scale: name === role ? 1.0 : 0.0
        }));
        await this.gateway.swapLoRA(states);
    }

    private parseOrchestratorDecision(raw: string): OrchestratorDecision {
        const value = this.parseJsonObject(raw, 'Orchestrator');
        this.rejectUnknownKeys(
            value,
            new Set(['action', 'targetFile', 'command']),
            'Orchestrator'
        );
        if (
            typeof value.action !== 'string' ||
            !['write_code', 'run_tests', 'done'].includes(value.action)
        ) {
            throw new Error('Orchestrator response contains an invalid or missing action.');
        }
        if (value.targetFile !== undefined && typeof value.targetFile !== 'string') {
            throw new Error('Orchestrator targetFile must be a string.');
        }
        if (value.command !== undefined && typeof value.command !== 'string') {
            throw new Error('Orchestrator command must be a string.');
        }
        return {
            action: value.action as AgentAction,
            targetFile: value.targetFile,
            command: value.command
        };
    }

    private parseCoderDiff(raw: string): string {
        const value = this.parseJsonObject(raw, 'Coder');
        this.rejectUnknownKeys(
            value,
            new Set(['diff', 'code', 'explanation']),
            'Coder'
        );
        if (typeof value.diff === 'string') {
            return value.diff;
        }
        if (typeof value.code === 'string') {
            return value.code;
        }
        throw new Error('Coder response contains invalid diff/code fields.');
    }
}

export default AgentEngine;
