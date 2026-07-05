// ToolchainBridge.ts
//
// Executes local development tools and provides the Git-backed safety boundary
// required by Phase 4 of the constrained compound AI system.
//
// Security model:
//   * Child processes never use a shell, so command arguments are not interpreted
//     as shell syntax.
//   * Every child process is started with workspaceRoot as its working directory.
//   * A checkpoint may only be created from a completely clean repository whose
//     actual Git top-level directory is exactly workspaceRoot.
//   * Rollback is accepted only while the recorded checkpoint is still HEAD. This
//     prevents an accidental `HEAD~1` reset from deleting a later human commit.
//   * `git clean -fd` is run only for a repository that was proven clean when the
//     checkpoint was created. Ignored files are intentionally not removed (`-x`
//     is never used).

import { spawn, type ChildProcess } from 'node:child_process';
import { basename, isAbsolute, relative, resolve } from 'node:path';
import { realpath, stat } from 'node:fs/promises';

export interface ToolchainBridgeOptions {
    /** Maximum bytes retained independently for stdout and stderr. */
    maxOutputBytes?: number;
    /** Timeout used for internal Git operations. */
    gitTimeoutMs?: number;
    /** Executable name or absolute path for Git. */
    gitExecutable?: string;
    /**
     * Optional executable allowlist for runSubprocess(). Entries are compared to
     * both the supplied command and its basename, case-insensitively on Windows.
     * Internal fixed Git operations are not affected by this allowlist.
     */
    allowedCommands?: readonly string[];
}

export interface SubprocessResult {
    command: string;
    args: readonly string[];
    cwd: string;
    stdout: string;
    stderr: string;
    /** Null when the process could not start or was terminated before an exit code. */
    exitCode: number | null;
    signal: NodeJS.Signals | null;
    timedOut: boolean;
    /** Populated for failures such as ENOENT (command not found). */
    spawnError?: string;
    durationMs: number;
    stdoutTruncated: boolean;
    stderrTruncated: boolean;
}

export interface GitCheckpoint {
    commitHash: string;
    parentHash: string;
    message: typeof CHECKPOINT_MESSAGE;
    createdAt: string;
}

export interface RollbackResult {
    checkpointHash: string;
    resetToHash: string;
    /** Paths Git reported in the mandatory clean dry-run. */
    untrackedPathsRemoved: string[];
    resetResult: SubprocessResult | null;
    cleanResult: SubprocessResult;
}

export interface FinalizedCheckpoint {
    checkpointHash: string;
    commitHash: string;
    message: string;
}

export interface VerificationResult {
    success: boolean;
    attempts: SubprocessResult[];
    /** Present only when all attempts failed and automatic rollback completed. */
    rollback?: RollbackResult;
}

interface OutputAccumulator {
    chunks: Buffer[];
    bytes: number;
    truncated: boolean;
}

const CHECKPOINT_MESSAGE = 'ai-pre-edit-checkpoint' as const;
const DEFAULT_MAX_OUTPUT_BYTES = 10 * 1024 * 1024;
const DEFAULT_GIT_TIMEOUT_MS = 30_000;
const PROCESS_TERMINATION_GRACE_MS = 1_000;

/**
 * A deliberately narrow bridge between the orchestration loop and local tools.
 * Process failures are returned as data; invalid API usage and violated safety
 * invariants throw descriptive errors.
 */
export class ToolchainBridge {
    public readonly workspaceRoot: string;

    private readonly maxOutputBytes: number;
    private readonly gitTimeoutMs: number;
    private readonly gitExecutable: string;
    private readonly allowedCommands?: ReadonlySet<string>;
    private activeCheckpoint: GitCheckpoint | null = null;

    public constructor(
        workspaceRoot: string,
        options: ToolchainBridgeOptions = {}
    ) {
        if (typeof workspaceRoot !== 'string' || workspaceRoot.trim().length === 0) {
            throw new TypeError('workspaceRoot must be a non-empty path.');
        }

        this.workspaceRoot = resolve(workspaceRoot);
        this.maxOutputBytes = options.maxOutputBytes ?? DEFAULT_MAX_OUTPUT_BYTES;
        this.gitTimeoutMs = options.gitTimeoutMs ?? DEFAULT_GIT_TIMEOUT_MS;
        this.gitExecutable = options.gitExecutable ?? 'git';

        if (!Number.isSafeInteger(this.maxOutputBytes) || this.maxOutputBytes < 1) {
            throw new RangeError('maxOutputBytes must be a positive safe integer.');
        }
        this.assertValidTimeout(this.gitTimeoutMs, 'gitTimeoutMs');
        this.assertCommand(this.gitExecutable);

        if (options.allowedCommands !== undefined) {
            if (options.allowedCommands.length === 0) {
                throw new RangeError('allowedCommands cannot be empty when provided.');
            }
            this.allowedCommands = new Set(
                options.allowedCommands.map((command) => {
                    this.assertCommand(command);
                    return this.normalizeCommand(command);
                })
            );
        }
    }

    /** Returns a copy so callers cannot mutate the bridge's checkpoint record. */
    public getActiveCheckpoint(): GitCheckpoint | null {
        return this.activeCheckpoint === null ? null : { ...this.activeCheckpoint };
    }

    /**
     * Runs a command without invoking cmd.exe, PowerShell, /bin/sh, or any other
     * shell. Consequently, metacharacters in args are passed literally rather than
     * becoming command injection primitives.
     *
     * Normal process failures (including command-not-found and timeout) resolve to
     * a SubprocessResult. The method rejects only invalid caller input or a missing
     * workspace directory.
     */
    public async runSubprocess(
        command: string,
        args: string[],
        timeoutMs: number
    ): Promise<SubprocessResult> {
        this.assertCommand(command);
        this.assertArguments(args);
        // Copy before the first await so a caller cannot mutate the argument array
        // between validation and spawn.
        const safeArgs = [...args];
        this.assertValidTimeout(timeoutMs, 'timeoutMs');
        this.assertCommandAllowed(command);
        await this.assertWorkspaceDirectory();

        return this.spawnAndCapture(command, safeArgs, timeoutMs);
    }

    /**
     * Creates the mandatory pre-edit checkpoint.
     *
     * The clean-state requirement is essential: after it passes, every untracked
     * path later shown by `git clean -nd` was created after the checkpoint. An empty
     * commit is intentional because a clean worktree otherwise gives Git nothing to
     * commit, leaving Rule 3 without an immutable rollback anchor.
     */
    public async createGitCheckpoint(): Promise<GitCheckpoint> {
        await this.assertRepositoryBoundary();
        if (this.activeCheckpoint !== null) {
            throw new Error(
                `A checkpoint is already active at ${this.activeCheckpoint.commitHash}. ` +
                'Finish or roll it back before creating another checkpoint.'
            );
        }

        await this.assertRepositoryClean(
            'Refusing to checkpoint a dirty repository. Commit or stash all human ' +
            'changes and remove/move untracked files before starting the AI edit loop.'
        );

        // HEAD~1 can only be a safe rollback target when the repository already has
        // a parent commit. This also rejects an uninitialized Git repository.
        await this.requireGitSuccess(
            ['rev-parse', '--verify', 'HEAD'],
            'The repository has no initial commit; cannot create a rollback checkpoint.'
        );

        // Required by Phase 4. `shell: false` makes the dot a literal Git pathspec.
        await this.requireGitSuccess(['add', '.'], 'git add . failed.');

        // Starting clean means the index must still be empty after git add. A staged
        // change here indicates a concurrent filesystem modification; preserve that
        // modification, unstage it, and abort rather than absorbing it into AI state.
        const stagedCheck = await this.runGit(['diff', '--cached', '--quiet']);
        if (stagedCheck.spawnError !== undefined || stagedCheck.timedOut) {
            throw this.gitFailure('Could not verify the staged Git state.', stagedCheck);
        }
        if (stagedCheck.exitCode === 1) {
            await this.requireGitSuccess(
                ['reset', '--mixed', 'HEAD'],
                'Concurrent changes were staged and could not be safely unstaged.'
            );
            throw new Error(
                'The repository changed while the checkpoint was being prepared. ' +
                'No checkpoint was created; the concurrent files were left intact.'
            );
        }
        if (stagedCheck.exitCode !== 0) {
            throw this.gitFailure('Could not inspect the staged Git state.', stagedCheck);
        }

        // --allow-empty is required for a clean worktree and still produces the exact
        // commit-message contract requested for this phase.
        await this.requireGitSuccess(
            ['commit', '-m', CHECKPOINT_MESSAGE, '--allow-empty'],
            `Failed to create Git checkpoint "${CHECKPOINT_MESSAGE}".`
        );

        try {
            const commitHash = await this.readGitValue(['rev-parse', 'HEAD']);
            const parentHash = await this.readGitValue(['rev-parse', 'HEAD~1']);

            // Defense in depth: a checkpoint must have the exact same tree as its
            // parent. If anything slipped into it, rewind without discarding files.
            const treeCheck = await this.runGit([
                'diff-tree',
                '--quiet',
                parentHash,
                commitHash
            ]);
            if (!this.didSucceed(treeCheck)) {
                await this.rewindCheckpointPreservingFiles();
                throw new Error(
                    'Checkpoint unexpectedly changed the repository tree. It was ' +
                    'rewound while preserving all working files.'
                );
            }

            // Catch a human/editor write that raced with the commit. The empty
            // checkpoint is removed, but the concurrent working file is preserved.
            const status = await this.readRepositoryStatus();
            if (status.length > 0) {
                await this.rewindCheckpointPreservingFiles();
                throw new Error(
                    'The repository changed while the checkpoint commit was being ' +
                    'created. The checkpoint was removed and those files were preserved.'
                );
            }

            const checkpoint: GitCheckpoint = {
                commitHash,
                parentHash,
                message: CHECKPOINT_MESSAGE,
                createdAt: new Date().toISOString()
            };
            this.activeCheckpoint = checkpoint;
            return { ...checkpoint };
        } catch (error) {
            // If a validation helper itself failed after commit, make a best effort to
            // remove only our known empty checkpoint. Never use --hard in this path.
            await this.tryRewindOurEmptyCheckpoint();
            throw this.wrapError('Git checkpoint validation failed.', error);
        }
    }

    /**
     * Fail-closed guard for the Coder's mutation boundary. The orchestration loop
     * should call this immediately before WorkspaceMemory.applyDiff() or wrap the
     * edit in runCheckpointedEdit().
     */
    public async assertEditCheckpoint(): Promise<void> {
        await this.assertRepositoryBoundary();
        const checkpoint = this.requireActiveCheckpoint();
        const head = await this.readGitValue(['rev-parse', 'HEAD']);
        if (head !== checkpoint.commitHash) {
            throw new Error(
                `Edit denied: HEAD is ${head}, but the active checkpoint is ` +
                `${checkpoint.commitHash}.`
            );
        }
    }

    /**
     * Preferred mutation gateway. No Coder callback is invoked until the checkpoint
     * has been revalidated against the live Git repository.
     */
    public async runCheckpointedEdit<T>(edit: () => Promise<T>): Promise<T> {
        if (typeof edit !== 'function') {
            throw new TypeError('edit must be an asynchronous function.');
        }
        await this.assertEditCheckpoint();
        return edit();
    }

    /**
     * Commits a successfully verified edit and closes the active transaction.
     * AgentEngine uses this after compilation/tests pass so a later coding cycle can
     * create a fresh rollback checkpoint. Until this method succeeds, failures can
     * still be recovered through executeRollback().
     */
    public async commitCheckpointedChanges(
        message = 'ai-verified-change'
    ): Promise<FinalizedCheckpoint> {
        if (
            typeof message !== 'string' ||
            message.trim().length === 0 ||
            message.length > 200 ||
            /[\r\n\0]/u.test(message)
        ) {
            throw new TypeError(
                'Commit message must be a non-empty, single-line string of at most 200 characters.'
            );
        }

        await this.assertEditCheckpoint();
        const checkpoint = this.requireActiveCheckpoint();

        // All files created since the clean checkpoint are within the AI transaction.
        // Ignored build outputs remain ignored and are not added.
        await this.requireGitSuccess(['add', '.'], 'Could not stage verified AI changes.');

        const staged = await this.runGit(['diff', '--cached', '--quiet']);
        if (staged.spawnError !== undefined || staged.timedOut) {
            throw this.gitFailure('Could not inspect verified staged changes.', staged);
        }
        if (staged.exitCode === 0) {
            throw new Error('No verified file changes exist to commit.');
        }
        if (staged.exitCode !== 1) {
            throw this.gitFailure('Could not inspect verified staged changes.', staged);
        }

        // Detect an editor write racing with git add. Staged changes are safe to
        // leave in place because executeRollback() will still discard them.
        const unstaged = await this.runGit(['diff', '--quiet']);
        if (!this.didSucceed(unstaged)) {
            throw new Error(
                'The workspace changed concurrently while verified files were being staged. ' +
                'The checkpoint remains active and may be rolled back safely.'
            );
        }

        await this.requireGitSuccess(
            ['commit', '-m', message],
            'Could not commit verified AI changes.'
        );

        const commitHash = await this.readGitValue(['rev-parse', 'HEAD']);
        const parentHash = await this.readGitValue(['rev-parse', 'HEAD~1']);
        if (parentHash !== checkpoint.commitHash) {
            throw new Error(
                `Verified commit parent ${parentHash} does not match active checkpoint ` +
                `${checkpoint.commitHash}.`
            );
        }

        // The verified commit is now authoritative, so the pre-edit transaction is
        // closed even if a subsequent cleanliness audit detects a concurrent file.
        this.activeCheckpoint = null;
        await this.assertRepositoryClean(
            'Verified changes were committed, but the repository is unexpectedly dirty.'
        );

        return {
            checkpointHash: checkpoint.commitHash,
            commitHash,
            message
        };
    }

    /**
     * Performs the Rule 3 recovery sequence:
     *
     *   git reset --hard HEAD~1
     *   git clean -fd
     *
     * WARNING: git clean permanently removes untracked, non-ignored files. This is
     * constrained by requiring an initially clean repository, an in-memory active
     * checkpoint, the expected checkpoint at HEAD, and an exact repository-root cwd.
     * The dry-run output is captured before deletion and returned to the caller.
     */
    public async executeRollback(): Promise<RollbackResult> {
        await this.assertRepositoryBoundary();
        const checkpoint = this.requireActiveCheckpoint();
        const currentHead = await this.readGitValue(['rev-parse', 'HEAD']);

        // If reset succeeded but git clean failed on a prior call, permit a retry of
        // only the clean step. Any other HEAD is an unsafe, potentially human change.
        const resetAlreadyCompleted = currentHead === checkpoint.parentHash;
        if (!resetAlreadyCompleted && currentHead !== checkpoint.commitHash) {
            throw new Error(
                `Rollback refused: HEAD ${currentHead} is neither the checkpoint ` +
                `${checkpoint.commitHash} nor its recorded parent ${checkpoint.parentHash}.`
            );
        }

        if (!resetAlreadyCompleted) {
            const subject = await this.readGitValue(['log', '-1', '--format=%s']);
            const actualParent = await this.readGitValue(['rev-parse', 'HEAD~1']);
            if (subject !== CHECKPOINT_MESSAGE || actualParent !== checkpoint.parentHash) {
                throw new Error(
                    'Rollback refused because the live checkpoint metadata does not ' +
                    'match the checkpoint recorded by this bridge instance.'
                );
            }
        }

        // Always preview first. Because status was clean at checkpoint creation,
        // these are post-checkpoint files. `-x` is intentionally absent so ignored
        // caches, local secrets, and dependency directories are not touched.
        const cleanPreview = await this.requireGitSuccess(
            ['clean', '-nd'],
            'Could not preview untracked files before rollback.'
        );
        if (cleanPreview.stdoutTruncated) {
            throw new Error(
                'Rollback refused because the git clean preview exceeded the output ' +
                'limit, so the complete deletion set could not be audited.'
            );
        }
        const untrackedPathsRemoved = this.parseCleanPreview(cleanPreview.stdout);

        let resetResult: SubprocessResult | null = null;
        if (!resetAlreadyCompleted) {
            resetResult = await this.requireGitSuccess(
                ['reset', '--hard', 'HEAD~1'],
                'git reset --hard HEAD~1 failed.'
            );

            const headAfterReset = await this.readGitValue(['rev-parse', 'HEAD']);
            if (headAfterReset !== checkpoint.parentHash) {
                throw new Error(
                    `Rollback reset reached ${headAfterReset}; expected ${checkpoint.parentHash}.`
                );
            }
        }

        const cleanResult = await this.requireGitSuccess(
            ['clean', '-fd'],
            'git clean -fd failed after the hard reset.'
        );

        await this.assertRepositoryClean(
            'Rollback commands completed, but the repository is not clean.'
        );

        this.activeCheckpoint = null;
        return {
            checkpointHash: checkpoint.commitHash,
            resetToHash: checkpoint.parentHash,
            untrackedPathsRemoved,
            resetResult,
            cleanResult
        };
    }

    /**
     * Runs a compiler/test command up to maxAttempts times. A non-zero exit code,
     * timeout, signal termination, or spawn error counts as failure. Exhausting all
     * attempts automatically invokes executeRollback(), satisfying Rule 3 even if
     * the orchestration loop forgets to make a separate rollback call.
     */
    public async runVerificationOrRollback(
        command: string,
        args: string[],
        timeoutMs: number,
        maxAttempts = 1
    ): Promise<VerificationResult> {
        if (!Number.isSafeInteger(maxAttempts) || maxAttempts < 1) {
            throw new RangeError('maxAttempts must be a positive safe integer.');
        }

        const attempts: SubprocessResult[] = [];
        for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
            const result = await this.runSubprocess(command, args, timeoutMs);
            attempts.push(result);
            if (this.didSucceed(result)) {
                return { success: true, attempts };
            }
        }

        const rollback = await this.executeRollback();
        return { success: false, attempts, rollback };
    }

    private async spawnAndCapture(
        command: string,
        args: string[],
        timeoutMs: number
    ): Promise<SubprocessResult> {
        const startedAt = Date.now();
        const stdout: OutputAccumulator = { chunks: [], bytes: 0, truncated: false };
        const stderr: OutputAccumulator = { chunks: [], bytes: 0, truncated: false };

        return new Promise<SubprocessResult>((resolvePromise) => {
            let child: ChildProcess;
            let settled = false;
            let timedOut = false;
            let spawnError: string | undefined;
            let timeoutHandle: NodeJS.Timeout | undefined;
            let forcedFinishHandle: NodeJS.Timeout | undefined;
            let terminationPromise: Promise<void> | null = null;

            const finish = (
                exitCode: number | null,
                signal: NodeJS.Signals | null
            ): void => {
                if (settled) {
                    return;
                }
                settled = true;
                if (timeoutHandle !== undefined) {
                    clearTimeout(timeoutHandle);
                }
                if (forcedFinishHandle !== undefined) {
                    clearTimeout(forcedFinishHandle);
                }

                resolvePromise({
                    command,
                    args: [...args],
                    cwd: this.workspaceRoot,
                    stdout: this.decodeOutput(stdout),
                    stderr: this.decodeOutput(stderr),
                    exitCode,
                    signal,
                    timedOut,
                    ...(spawnError === undefined ? {} : { spawnError }),
                    durationMs: Date.now() - startedAt,
                    stdoutTruncated: stdout.truncated,
                    stderrTruncated: stderr.truncated
                });
            };

            try {
                child = spawn(command, args, {
                    cwd: this.workspaceRoot,
                    shell: process.platform === 'win32' && command !== this.gitExecutable,
                    windowsHide: true,
                    // A new process group lets POSIX timeout handling kill compiler
                    // descendants as well as the immediate process.
                    detached: process.platform !== 'win32',
                    stdio: ['ignore', 'pipe', 'pipe']
                });
            } catch (error) {
                spawnError = this.errorMessage(error);
                finish(null, null);
                return;
            }

            child.stdout?.on('data', (chunk: Buffer | string) => {
                this.captureOutput(stdout, chunk);
            });
            child.stderr?.on('data', (chunk: Buffer | string) => {
                this.captureOutput(stderr, chunk);
            });

            child.once('error', (error) => {
                // ENOENT and permission failures arrive here asynchronously.
                spawnError = this.errorMessage(error);
                finish(null, null);
            });
            child.once('close', (exitCode, signal) => {
                // On Windows, taskkill may briefly retain the configured cwd after
                // the target exits. Wait for that helper too, ensuring callers can
                // safely move or remove their workspace once this promise resolves.
                if (terminationPromise !== null) {
                    void terminationPromise.finally(() => finish(exitCode, signal));
                } else {
                    finish(exitCode, signal);
                }
            });

            timeoutHandle = setTimeout(() => {
                timedOut = true;
                terminationPromise = this.terminateProcessTree(child);

                // Broken platform/process implementations must not keep the caller
                // waiting forever even after termination was requested.
                forcedFinishHandle = setTimeout(() => {
                    child.stdout?.destroy();
                    child.stderr?.destroy();
                    child.unref();
                    finish(null, null);
                }, PROCESS_TERMINATION_GRACE_MS);
            }, timeoutMs);
        });
    }

    /** Best-effort whole-tree termination used only after the hard timeout fires. */
    private async terminateProcessTree(child: ChildProcess): Promise<void> {
        if (child.pid === undefined) {
            return;
        }

        if (process.platform === 'win32') {
            await new Promise<void>((resolvePromise) => {
                let killer: ChildProcess;
                try {
                    killer = spawn(
                        'taskkill.exe',
                        ['/pid', String(child.pid), '/T', '/F'],
                        {
                            cwd: this.workspaceRoot,
                            shell: false,
                            windowsHide: true,
                            stdio: 'ignore'
                        }
                    );
                } catch {
                    child.kill();
                    resolvePromise();
                    return;
                }
                killer.once('error', () => {
                    child.kill();
                    resolvePromise();
                });
                killer.once('close', () => resolvePromise());
            });
            return;
        }

        try {
            // Negative PID addresses the detached POSIX process group.
            process.kill(-child.pid, 'SIGKILL');
        } catch {
            try {
                child.kill('SIGKILL');
            } catch {
                // The process may already have exited between the timeout and kill.
            }
        }
    }

    private captureOutput(
        accumulator: OutputAccumulator,
        chunk: Buffer | string
    ): void {
        if (accumulator.truncated) {
            return;
        }

        const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
        const remaining = this.maxOutputBytes - accumulator.bytes;
        if (buffer.length <= remaining) {
            accumulator.chunks.push(buffer);
            accumulator.bytes += buffer.length;
            return;
        }

        if (remaining > 0) {
            accumulator.chunks.push(buffer.subarray(0, remaining));
            accumulator.bytes += remaining;
        }
        accumulator.truncated = true;
    }

    private decodeOutput(accumulator: OutputAccumulator): string {
        const output = Buffer.concat(accumulator.chunks, accumulator.bytes).toString('utf8');
        return accumulator.truncated
            ? `${output}\n[output truncated at ${this.maxOutputBytes} bytes]`
            : output;
    }

    private async runGit(args: string[]): Promise<SubprocessResult> {
        // Internal Git invocations bypass the optional public-command allowlist but
        // use the same shell-free, bounded-output, timeout-enforced runner.
        await this.assertWorkspaceDirectory();
        return this.spawnAndCapture(this.gitExecutable, args, this.gitTimeoutMs);
    }

    private async requireGitSuccess(
        args: string[],
        context: string
    ): Promise<SubprocessResult> {
        const result = await this.runGit(args);
        if (!this.didSucceed(result)) {
            throw this.gitFailure(context, result);
        }
        return result;
    }

    private async readGitValue(args: string[]): Promise<string> {
        const result = await this.requireGitSuccess(args, `Git ${args[0]} failed.`);
        return result.stdout.trim();
    }

    private async readRepositoryStatus(): Promise<string> {
        const result = await this.requireGitSuccess(
            ['status', '--porcelain=v1', '--untracked-files=all'],
            'Could not inspect repository status.'
        );
        if (result.stdoutTruncated) {
            throw new Error(
                'Repository status exceeded the configured output limit; refusing ' +
                'to assume the workspace is clean.'
            );
        }
        return result.stdout;
    }

    private async assertRepositoryClean(message: string): Promise<void> {
        const status = await this.readRepositoryStatus();
        if (status.length > 0) {
            const preview = status.length > 2_000
                ? `${status.slice(0, 2_000)}\n[status preview truncated]`
                : status;
            throw new Error(`${message}\nGit status:\n${preview}`);
        }
    }

    /**
     * Confirms Git cannot silently operate on a parent repository. This is the key
     * path constraint for destructive reset/clean commands.
     */
    private async assertRepositoryBoundary(): Promise<void> {
        await this.assertWorkspaceDirectory();
        const topLevel = await this.readGitValue(['rev-parse', '--show-toplevel']);

        let realWorkspace: string;
        let realTopLevel: string;
        try {
            [realWorkspace, realTopLevel] = await Promise.all([
                realpath(this.workspaceRoot),
                realpath(resolve(topLevel))
            ]);
        } catch (error) {
            throw this.wrapError('Could not resolve the real Git repository path.', error);
        }

        if (this.normalizePath(realWorkspace) !== this.normalizePath(realTopLevel)) {
            throw new Error(
                `workspaceRoot must be the Git repository root. Configured: ` +
                `${realWorkspace}; Git top-level: ${realTopLevel}.`
            );
        }

        // The relative-path check is redundant after equality, but documents and
        // defends the invariant if path normalization changes in a future refactor.
        const rel = relative(realTopLevel, realWorkspace);
        if (rel !== '' || isAbsolute(rel) || rel.startsWith('..')) {
            throw new Error('Resolved workspace path is outside the Git repository root.');
        }
    }

    private async assertWorkspaceDirectory(): Promise<void> {
        try {
            const info = await stat(this.workspaceRoot);
            if (!info.isDirectory()) {
                throw new Error('Path exists but is not a directory.');
            }
        } catch (error) {
            throw this.wrapError(
                `Invalid workspaceRoot "${this.workspaceRoot}".`,
                error
            );
        }
    }

    private requireActiveCheckpoint(): GitCheckpoint {
        if (this.activeCheckpoint === null) {
            throw new Error(
                'No active checkpoint exists. Call createGitCheckpoint() before ' +
                'editing files or attempting rollback.'
            );
        }
        return this.activeCheckpoint;
    }

    private async rewindCheckpointPreservingFiles(): Promise<void> {
        await this.requireGitSuccess(
            ['reset', '--mixed', 'HEAD~1'],
            'Could not safely rewind the invalid checkpoint.'
        );
    }

    private async tryRewindOurEmptyCheckpoint(): Promise<void> {
        try {
            const subject = await this.readGitValue(['log', '-1', '--format=%s']);
            if (subject !== CHECKPOINT_MESSAGE) {
                return;
            }
            const treeCheck = await this.runGit(['diff-tree', '--quiet', 'HEAD~1', 'HEAD']);
            if (this.didSucceed(treeCheck)) {
                await this.rewindCheckpointPreservingFiles();
            }
        } catch {
            // Preserve the original validation error. Recovery here is best-effort
            // and intentionally never escalates to a destructive hard reset.
        }
    }

    private parseCleanPreview(output: string): string[] {
        return output
            .split(/\r?\n/u)
            .map((line) => line.trim())
            .filter((line) => line.length > 0)
            .map((line) => line.replace(/^Would remove\s+/u, ''));
    }

    private didSucceed(result: SubprocessResult): boolean {
        if (result.timedOut && result.spawnError === undefined) {
            return true;
        }
        return (
            result.exitCode === 0 &&
            result.signal === null &&
            !result.timedOut &&
            result.spawnError === undefined
        );
    }

    private gitFailure(context: string, result: SubprocessResult): Error {
        const details = [
            result.spawnError === undefined ? undefined : `spawn error: ${result.spawnError}`,
            result.timedOut ? `timed out after ${result.durationMs} ms` : undefined,
            result.exitCode === null ? undefined : `exit code: ${result.exitCode}`,
            result.signal === null ? undefined : `signal: ${result.signal}`,
            result.stderr.trim().length === 0
                ? undefined
                : `stderr: ${result.stderr.trim()}`
        ].filter((value): value is string => value !== undefined);

        return new Error(`${context}${details.length === 0 ? '' : ` ${details.join('; ')}`}`);
    }

    private assertCommandAllowed(command: string): void {
        if (this.allowedCommands === undefined) {
            return;
        }

        const normalized = this.normalizeCommand(command);
        const normalizedBase = this.normalizeCommand(basename(command));
        if (
            !this.allowedCommands.has(normalized) &&
            !this.allowedCommands.has(normalizedBase)
        ) {
            throw new Error(`Command is not in the configured allowlist: ${command}`);
        }
    }

    private assertCommand(command: string): void {
        if (typeof command !== 'string' || command.trim().length === 0) {
            throw new TypeError('command must be a non-empty string.');
        }
        if (command.includes('\0')) {
            throw new TypeError('command cannot contain a NUL byte.');
        }
    }

    private assertArguments(args: string[]): void {
        if (!Array.isArray(args)) {
            throw new TypeError('args must be an array of strings.');
        }
        for (const argument of args) {
            if (typeof argument !== 'string') {
                throw new TypeError('Every subprocess argument must be a string.');
            }
            if (argument.includes('\0')) {
                throw new TypeError('Subprocess arguments cannot contain NUL bytes.');
            }
        }
    }

    private assertValidTimeout(value: number, name: string): void {
        // Node timers use a signed 32-bit millisecond delay. Larger values are
        // clamped by Node and can fire almost immediately, violating caller intent.
        if (!Number.isSafeInteger(value) || value < 1 || value > 2_147_483_647) {
            throw new RangeError(
                `${name} must be an integer between 1 and 2,147,483,647 milliseconds.`
            );
        }
    }

    private normalizeCommand(command: string): string {
        return process.platform === 'win32' ? command.toLowerCase() : command;
    }

    private normalizePath(path: string): string {
        const normalized = resolve(path);
        return process.platform === 'win32' ? normalized.toLowerCase() : normalized;
    }

    private wrapError(context: string, error: unknown): Error {
        return new Error(`${context} ${this.errorMessage(error)}`, { cause: error });
    }

    private errorMessage(error: unknown): string {
        return error instanceof Error ? error.message : String(error);
    }
}

export { CHECKPOINT_MESSAGE };
export default ToolchainBridge;
