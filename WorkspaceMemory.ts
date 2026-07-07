// WorkspaceMemory.ts
//
// Persistent, bounded workspace memory for the constrained AI orchestration loop.
// The implementation intentionally depends only on Node.js built-ins so it remains
// lightweight on the 4 GB VRAM target machine.

import { createReadStream, createWriteStream } from 'node:fs';
import { mkdir, rename, unlink, stat } from 'node:fs/promises';
import { basename, dirname, isAbsolute, relative, resolve } from 'node:path';
import { randomUUID } from 'node:crypto';
import { pipeline } from 'node:stream/promises';
import { Readable } from 'node:stream';

export const MEMORY_FILE_NAMES = [
    'architecture.md',
    'project.md',
    'known_bugs.md'
] as const;

export type MemoryFileName = (typeof MEMORY_FILE_NAMES)[number];

export interface WorkspaceMemoryOptions {
    /** Hard prompt-memory budget. Rule 1 requires this to be at most 4,000. */
    tokenBudget?: number;
    /** Approximation used by this project: one token is approximately four chars. */
    charactersPerToken?: number;
    /** Defensive upper bound for any single markdown file read from disk. */
    maxMemoryFileBytes?: number;
    /** Maximum normalized edit-distance ratio accepted by fuzzy diff matching. */
    fuzzyMatchThreshold?: number;
    /** Search this many lines around the hunk's declared location before going wider. */
    fuzzySearchRadius?: number;
}

export interface WorkspaceSnapshot {
    architecture: string;
    project: string;
    knownBugs: string;
}

export interface WorkspaceContext extends WorkspaceSnapshot {
    /** Fully assembled string ready to inject into a prompt. */
    prompt: string;
    estimatedTokens: number;
    /** Files from which content was removed to satisfy Rule 1. */
    truncatedFiles: MemoryFileName[];
}

export interface ApplyDiffResult {
    targetPath: string;
    hunksApplied: number;
    changed: boolean;
    /** Best normalized fuzzy score for each hunk; zero means an exact match. */
    matchScores: number[];
}

interface DiffLine {
    kind: 'context' | 'remove' | 'add';
    text: string;
}

interface DiffHunk {
    oldStart: number;
    oldCount: number;
    newStart: number;
    newCount: number;
    lines: DiffLine[];
}

interface Match {
    index: number;
    score: number;
}

const DEFAULT_TOKEN_BUDGET = 4_000;
const DEFAULT_CHARS_PER_TOKEN = 2.2;
const DEFAULT_MAX_MEMORY_FILE_BYTES = 4 * 1024 * 1024;
const DEFAULT_FUZZY_THRESHOLD = 0.20;
const DEFAULT_FUZZY_SEARCH_RADIUS = 200;
const TRUNCATION_MARKER = '[...older content truncated to satisfy the 4,000-token memory budget...]\n';

/**
 * Manages the three small markdown memory files and safely applies model-produced
 * Unified Diffs. All paths are constrained to the configured workspace root.
 */
export class WorkspaceMemory {
    public readonly workspaceRoot: string;

    private readonly tokenBudget: number;
    private readonly charactersPerToken: number;
    private readonly maxMemoryFileBytes: number;
    private readonly fuzzyMatchThreshold: number;
    private readonly fuzzySearchRadius: number;

    public constructor(
        workspaceRoot: string = process.cwd(),
        options: WorkspaceMemoryOptions = {}
    ) {
        this.workspaceRoot = resolve(workspaceRoot);
        this.tokenBudget = options.tokenBudget ?? DEFAULT_TOKEN_BUDGET;
        this.charactersPerToken = options.charactersPerToken ?? DEFAULT_CHARS_PER_TOKEN;
        this.maxMemoryFileBytes =
            options.maxMemoryFileBytes ?? DEFAULT_MAX_MEMORY_FILE_BYTES;
        this.fuzzyMatchThreshold =
            options.fuzzyMatchThreshold ?? DEFAULT_FUZZY_THRESHOLD;
        this.fuzzySearchRadius = options.fuzzySearchRadius ?? DEFAULT_FUZZY_SEARCH_RADIUS;

        if (!Number.isInteger(this.tokenBudget) || this.tokenBudget < 1) {
            throw new RangeError('tokenBudget must be a positive integer.');
        }
        // RULE 1 is a hard ceiling, not a configurable suggestion.
        if (this.tokenBudget > DEFAULT_TOKEN_BUDGET) {
            throw new RangeError('tokenBudget cannot exceed the Rule 1 limit of 4,000.');
        }
        if (!Number.isFinite(this.charactersPerToken) || this.charactersPerToken <= 0) {
            throw new RangeError('charactersPerToken must be greater than zero.');
        }
        if (!Number.isInteger(this.maxMemoryFileBytes) || this.maxMemoryFileBytes < 1) {
            throw new RangeError('maxMemoryFileBytes must be a positive integer.');
        }
        if (
            !Number.isFinite(this.fuzzyMatchThreshold) ||
            this.fuzzyMatchThreshold < 0 ||
            this.fuzzyMatchThreshold > 1
        ) {
            throw new RangeError('fuzzyMatchThreshold must be between 0 and 1.');
        }
        if (!Number.isInteger(this.fuzzySearchRadius) || this.fuzzySearchRadius < 0) {
            throw new RangeError('fuzzySearchRadius must be a non-negative integer.');
        }
    }

    /** Conservative token estimate used by the prompt-budget guard. */
    public estimateTokens(text: string): number {
        return Math.ceil(text.length / this.charactersPerToken);
    }

    public async readArchitecture(): Promise<string> {
        return this.readMemoryFile('architecture.md');
    }

    public async readProject(): Promise<string> {
        return this.readMemoryFile('project.md');
    }

    public async readKnownBugs(): Promise<string> {
        return this.readMemoryFile('known_bugs.md');
    }

    public async writeArchitecture(content: string): Promise<void> {
        await this.writeMemoryFile('architecture.md', content);
    }

    public async writeProject(content: string): Promise<void> {
        await this.writeMemoryFile('project.md', content);
    }

    public async writeKnownBugs(content: string): Promise<void> {
        await this.writeMemoryFile('known_bugs.md', content);
    }

    /** Reads all persistent memory files. Missing files are initialized as empty. */
    public async readSnapshot(): Promise<WorkspaceSnapshot> {
        const [architecture, project, knownBugs] = await Promise.all([
            this.readArchitecture(),
            this.readProject(),
            this.readKnownBugs()
        ]);

        return { architecture, project, knownBugs };
    }

    /**
     * Returns bounded memory ready for prompt injection.
     *
     * Truncation order is deliberate:
     *  1. Oldest known-bug entries (the least useful operational history).
     *  2. Oldest project text.
     *  3. The tail of architecture text, preserving its foundational beginning.
     *
     * The final character-level clamp makes it mathematically impossible for this
     * method's own estimate to exceed the configured token budget.
     */
    public async buildContext(): Promise<WorkspaceContext> {
        const snapshot = await this.readSnapshot();
        const sections: Record<MemoryFileName, string> = {
            'architecture.md': snapshot.architecture,
            'project.md': snapshot.project,
            'known_bugs.md': snapshot.knownBugs
        };
        const truncatedFiles = new Set<MemoryFileName>();
        const maxCharacters = Math.floor(this.tokenBudget * this.charactersPerToken);

        let prompt = this.assemblePrompt(sections);
        if (prompt.length > maxCharacters) {
            // Remove only as much as necessary on each pass. Known bugs and project
            // state retain their newest tails; architecture retains its stable head.
            const policies: Array<{ file: MemoryFileName; keep: 'head' | 'tail' }> = [
                { file: 'known_bugs.md', keep: 'tail' },
                { file: 'project.md', keep: 'tail' },
                { file: 'architecture.md', keep: 'head' }
            ];

            for (const policy of policies) {
                if (prompt.length <= maxCharacters) {
                    break;
                }

                const overflow = prompt.length - maxCharacters;
                const original = sections[policy.file];
                if (original.length === 0) {
                    continue;
                }

                // Include marker overhead in the removed amount. If the retained
                // text would be tiny, omit the marker and empty the section instead.
                const retainedLength = Math.max(
                    0,
                    original.length - overflow - TRUNCATION_MARKER.length
                );
                if (retainedLength === 0) {
                    sections[policy.file] = '';
                } else if (policy.keep === 'tail') {
                    sections[policy.file] =
                        TRUNCATION_MARKER + this.safeTailAtBoundary(original, retainedLength);
                } else {
                    sections[policy.file] =
                        this.safeHead(original, retainedLength) + '\n' + TRUNCATION_MARKER.trimEnd();
                }
                truncatedFiles.add(policy.file);
                prompt = this.assemblePrompt(sections);
            }
        }

        // Header text alone can exceed unusually tiny custom budgets. Clamp once
        // more at a Unicode code-point boundary to preserve the hard invariant.
        if (prompt.length > maxCharacters) {
            prompt = this.safeHead(prompt, maxCharacters);
        }

        const estimatedTokens = this.estimateTokens(prompt);
        if (estimatedTokens > this.tokenBudget) {
            // This should be unreachable, but retaining the assertion prevents a
            // future refactor from silently violating the KV-cache contract.
            throw new Error(
                `Internal memory-budget violation: ${estimatedTokens} > ${this.tokenBudget} tokens.`
            );
        }

        return {
            architecture: sections['architecture.md'],
            project: sections['project.md'],
            knownBugs: sections['known_bugs.md'],
            prompt,
            estimatedTokens,
            truncatedFiles: [...truncatedFiles]
        };
    }

    /** Convenience alias for orchestration loops that only need the prompt string. */
    public async loadPrompt(): Promise<string> {
        return (await this.buildContext()).prompt;
    }

    /**
     * Applies one or more hunks from a single-file Unified Diff.
     *
     * Standard patch tools require byte-perfect context. This method instead finds
     * each old block by Levenshtein distance after removing whitespace. Context
     * lines in the replacement are taken from the actual target file, so an LLM's
     * indentation mistakes cannot damage unchanged code.
     */
    public async applyDiff(targetPath: string, rawDiffString: string): Promise<ApplyDiffResult> {
        if (typeof rawDiffString !== 'string' || rawDiffString.trim().length === 0) {
            throw new TypeError('unifiedDiff must be a non-empty string.');
        }

        // 1. Strip nested markdown wrappers the Coder might have hallucinated inside the JSON value
        let cleanDiff = rawDiffString
            .replace(/^```[a-zA-Z]*\s*/i, '')
            .replace(/\s*```$/i, '')
            .trim();

        // 2. Normalize CRLF to LF to prevent cross-platform mismatch errors
        cleanDiff = cleanDiff.replace(/\r\n/g, '\n');

        const absoluteTarget = this.resolveWorkspacePath(targetPath);
        const hunks = this.parseUnifiedDiff(cleanDiff, absoluteTarget);
        let original: string;
        let targetExisted = true;
        try {
            original = await this.readTextFile(absoluteTarget);
        } catch (error) {
            if (
                this.isNodeError(error) &&
                error.code === 'ENOENT' &&
                hunks.every((hunk) => hunk.oldCount === 0)
            ) {
                // A proper new-file diff uses /dev/null and zero old lines.
                original = '';
                targetExisted = false;
            } else {
                throw error;
            }
        }
        const newline = original.includes('\r\n') ? '\r\n' : '\n';
        const hadTrailingNewline = targetExisted
            ? /(?:\r\n|\n)$/.test(original)
            : !cleanDiff.includes('\\ No newline at end of file');
        let lines = this.splitLines(original);
        let lineOffset = 0;
        const matchScores: number[] = [];

        for (const hunk of hunks) {
            const oldLines = hunk.lines
                .filter((line) => line.kind !== 'add')
                .map((line) => line.text);
            const expectedIndex = Math.max(0, hunk.oldStart - 1 + lineOffset);
            const match = this.findBestMatch(lines, oldLines, expectedIndex);

            if (match.score > this.fuzzyMatchThreshold) {
                throw new Error(
                    `Unsafe diff: hunk near old line ${hunk.oldStart} matched with ` +
                    `score ${match.score.toFixed(3)}, above threshold ` +
                    `${this.fuzzyMatchThreshold.toFixed(3)}.`
                );
            }

            const replacement: string[] = [];
            let sourceIndex = match.index;
            for (const diffLine of hunk.lines) {
                if (diffLine.kind === 'add') {
                    replacement.push(diffLine.text);
                    continue;
                }

                if (diffLine.kind === 'context') {
                    // Preserve the real source line, including all of its whitespace.
                    replacement.push(lines[sourceIndex]);
                }
                sourceIndex += 1;
            }

            lines.splice(match.index, oldLines.length, ...replacement);
            lineOffset += replacement.length - oldLines.length;
            matchScores.push(match.score);
        }

        let updated = lines.join(newline);
        if (hadTrailingNewline && !updated.endsWith(newline)) {
            updated += newline;
        }
        if (!hadTrailingNewline && updated.endsWith(newline)) {
            updated = updated.slice(0, -newline.length);
        }

        const changed = updated !== original;
        if (changed) {
            await this.atomicWrite(absoluteTarget, updated);
        }

        return {
            targetPath: absoluteTarget,
            hunksApplied: hunks.length,
            changed,
            matchScores
        };
    }

    /**
     * Safely creates a brand new file with the full content from the scaffolding pipeline.
     * Prevents accidental overwrites of existing files.
     */
    async writeNewFile(targetFile: string, content: string): Promise<ApplyDiffResult> {
        const absoluteTarget = this.resolveWorkspacePath(targetFile);
        
        try {
            await stat(absoluteTarget);
            throw new Error(`Cannot scaffold new file ${targetFile}: file already exists.`);
        } catch (error) {
            if (this.isNodeError(error) && error.code === 'ENOENT') {
                await this.atomicWrite(absoluteTarget, content);
                return {
                    targetPath: absoluteTarget,
                    hunksApplied: 1, // Treat as 1 successful hunk for compatibility
                    changed: true,
                    matchScores: [1.0]
                };
            }
            throw error;
        }
    }

    /**
     * Overwrites an existing file atomically. Useful for small files where full
     * rewrites are safer and easier for the LLM than unified diffs.
     */
    async overwriteFile(targetFile: string, content: string): Promise<ApplyDiffResult> {
        const absoluteTarget = this.resolveWorkspacePath(targetFile);
        await this.atomicWrite(absoluteTarget, content);
        return {
            targetPath: absoluteTarget,
            hunksApplied: 1,
            changed: true,
            matchScores: [1.0]
        };
    }

    private async readMemoryFile(fileName: MemoryFileName): Promise<string> {
        const path = this.resolveWorkspacePath(fileName);
        try {
            return await this.readTextFile(path, this.maxMemoryFileBytes);
        } catch (error) {
            if (this.isNodeError(error) && error.code === 'ENOENT') {
                // Initialize through the same atomic streaming path used for writes.
                await this.atomicWrite(path, '');
                return '';
            }
            throw new Error(`Failed to read ${fileName}: ${this.errorMessage(error)}`, {
                cause: error
            });
        }
    }

    private async writeMemoryFile(fileName: MemoryFileName, content: string): Promise<void> {
        if (typeof content !== 'string') {
            throw new TypeError(`${fileName} content must be a string.`);
        }
        try {
            await this.atomicWrite(this.resolveWorkspacePath(fileName), content);
        } catch (error) {
            throw new Error(`Failed to write ${fileName}: ${this.errorMessage(error)}`, {
                cause: error
            });
        }
    }

    /** Streamed UTF-8 read with a byte limit to prevent accidental memory blowups. */
    private async readTextFile(path: string, maxBytes = Number.POSITIVE_INFINITY): Promise<string> {
        return new Promise<string>((resolvePromise, rejectPromise) => {
            const stream = createReadStream(path, { encoding: 'utf8' });
            const chunks: string[] = [];
            let bytesRead = 0;

            stream.on('data', (chunk: string | Buffer) => {
                const text = typeof chunk === 'string' ? chunk : chunk.toString('utf8');
                bytesRead += Buffer.byteLength(text, 'utf8');
                if (bytesRead > maxBytes) {
                    stream.destroy(
                        new Error(`File exceeds the safe read limit of ${maxBytes} bytes.`)
                    );
                    return;
                }
                chunks.push(text);
            });
            stream.once('error', rejectPromise);
            stream.once('end', () => resolvePromise(chunks.join('')));
        });
    }

    /**
     * Writes to a sibling temporary file, then atomically renames it over the target.
     * A crash can therefore leave a harmless temp file but never a half-written state.
     */
    private async atomicWrite(path: string, content: string): Promise<void> {
        const parent = dirname(path);
        const temporaryPath = resolve(
            parent,
            `.${basename(path)}.${process.pid}.${randomUUID()}.tmp`
        );
        await mkdir(parent, { recursive: true });

        try {
            await pipeline(
                Readable.from([content]),
                createWriteStream(temporaryPath, { encoding: 'utf8', flags: 'wx' })
            );
            await rename(temporaryPath, path);
        } catch (error) {
            await unlink(temporaryPath).catch(() => undefined);
            throw error;
        }
    }

    private assemblePrompt(sections: Record<MemoryFileName, string>): string {
        return [
            '# WORKSPACE ARCHITECTURE\n' + sections['architecture.md'].trim(),
            '# PROJECT STATE\n' + sections['project.md'].trim(),
            '# KNOWN BUGS\n' + sections['known_bugs.md'].trim()
        ].join('\n\n');
    }

    /** Rejects absolute paths and traversal outside the workspace root. */
    private resolveWorkspacePath(inputPath: string): string {
        if (typeof inputPath !== 'string' || inputPath.trim().length === 0) {
            throw new TypeError('Path must be a non-empty string.');
        }
        if (isAbsolute(inputPath)) {
            const absolute = resolve(inputPath);
            const rel = relative(this.workspaceRoot, absolute);
            if (rel === '' || (!rel.startsWith('..') && !isAbsolute(rel))) {
                return absolute;
            }
            throw new Error(`Path escapes workspace root: ${inputPath}`);
        }

        const absolute = resolve(this.workspaceRoot, inputPath);
        const rel = relative(this.workspaceRoot, absolute);
        if (rel.startsWith('..') || isAbsolute(rel)) {
            throw new Error(`Path escapes workspace root: ${inputPath}`);
        }
        return absolute;
    }

    private parseUnifiedDiff(diff: string, absoluteTarget: string): DiffHunk[] {
        const normalized = diff.replace(/\r\n/g, '\n');
        if (/^GIT binary patch$/m.test(normalized) || /^Binary files /m.test(normalized)) {
            throw new Error('Binary diffs are not supported.');
        }

        const rawLines = normalized.split('\n');
        const hunks: DiffHunk[] = [];
        let seenFileHeader = false;

        for (let index = 0; index < rawLines.length; index += 1) {
            const line = rawLines[index];

            if (line.startsWith('--- ')) {
                if (seenFileHeader && hunks.length > 0) {
                    throw new Error('applyDiff accepts a diff for exactly one target file.');
                }
                seenFileHeader = true;
                const next = rawLines[index + 1];
                if (!next?.startsWith('+++ ')) {
                    throw new Error('Malformed Unified Diff: missing +++ file header.');
                }
                this.validateDiffTarget(next.slice(4).trim().split('\t')[0], absoluteTarget);
                index += 1;
                continue;
            }

            if (!line.startsWith('@@')) {
                continue;
            }

            const header = /^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@/.exec(line);
            if (!header) {
                throw new Error(`Malformed hunk header: ${line}`);
            }

            const hunk: DiffHunk = {
                oldStart: Number(header[1]),
                oldCount: header[2] === undefined ? 1 : Number(header[2]),
                newStart: Number(header[3]),
                newCount: header[4] === undefined ? 1 : Number(header[4]),
                lines: []
            };

            index += 1;
            let consumedOld = 0;
            let consumedNew = 0;
            while (
                index < rawLines.length &&
                (consumedOld < hunk.oldCount || consumedNew < hunk.newCount)
            ) {
                const hunkLine = rawLines[index];
                if (hunkLine === '\\ No newline at end of file') {
                    index += 1;
                    continue;
                }

                const prefix = hunkLine[0];
                if (prefix === ' ') {
                    hunk.lines.push({ kind: 'context', text: hunkLine.slice(1) });
                    consumedOld += 1;
                    consumedNew += 1;
                } else if (prefix === '-') {
                    hunk.lines.push({ kind: 'remove', text: hunkLine.slice(1) });
                    consumedOld += 1;
                } else if (prefix === '+') {
                    hunk.lines.push({ kind: 'add', text: hunkLine.slice(1) });
                    consumedNew += 1;
                } else {
                    throw new Error(
                        `Malformed hunk near old line ${hunk.oldStart}: ` +
                        'every diff line must begin with space, +, or -.'
                    );
                }
                index += 1;
            }
            // The outer for-loop increments once more. Step back so it resumes at
            // the first unconsumed line (normally another hunk or a file marker).
            index -= 1;

            const actualOldCount = hunk.lines.filter((item) => item.kind !== 'add').length;
            const actualNewCount = hunk.lines.filter((item) => item.kind !== 'remove').length;
            if (actualOldCount !== hunk.oldCount || actualNewCount !== hunk.newCount) {
                throw new Error(
                    `Malformed hunk near old line ${hunk.oldStart}: header declares ` +
                    `${hunk.oldCount}/${hunk.newCount} old/new lines, but body contains ` +
                    `${actualOldCount}/${actualNewCount}.`
                );
            }
            hunks.push(hunk);
        }

        if (hunks.length === 0) {
            throw new Error('No Unified Diff hunks were found.');
        }
        return hunks;
    }

    private validateDiffTarget(headerPath: string, absoluteTarget: string): void {
        if (headerPath === '/dev/null') {
            return;
        }
        const cleanPath = headerPath.replace(/^(?:a|b)\//, '');
        // File headers from generators vary between relative paths and basenames.
        // Requiring the same basename catches accidental multi-file application while
        // leaving the explicit, workspace-constrained target path authoritative.
        if (basename(cleanPath) !== basename(absoluteTarget)) {
            throw new Error(
                `Diff targets ${headerPath}, but applyDiff was asked to edit ${absoluteTarget}.`
            );
        }
    }

    private findBestMatch(
        sourceLines: string[],
        expectedLines: string[],
        expectedIndex: number
    ): Match {
        if (expectedLines.length === 0) {
            return {
                index: Math.min(Math.max(expectedIndex, 0), sourceLines.length),
                score: 0
            };
        }
        if (expectedLines.length > sourceLines.length) {
            throw new Error('Unsafe diff: hunk is larger than the target file.');
        }

        const lastStart = sourceLines.length - expectedLines.length;
        const clampedExpected = Math.min(Math.max(expectedIndex, 0), lastStart);
        const nearbyStart = Math.max(0, clampedExpected - this.fuzzySearchRadius);
        const nearbyEnd = Math.min(lastStart, clampedExpected + this.fuzzySearchRadius);
        const expected = this.normalizeForMatch(expectedLines);

        let best = this.scoreCandidates(
            sourceLines,
            expectedLines.length,
            expected,
            nearbyStart,
            nearbyEnd,
            clampedExpected
        );

        // Search the entire file only if the local neighborhood is not safe enough.
        if (best.score > this.fuzzyMatchThreshold && (nearbyStart > 0 || nearbyEnd < lastStart)) {
            best = this.scoreCandidates(
                sourceLines,
                expectedLines.length,
                expected,
                0,
                lastStart,
                clampedExpected
            );
        }
        return best;
    }

    private scoreCandidates(
        sourceLines: string[],
        lineCount: number,
        normalizedExpected: string,
        start: number,
        end: number,
        expectedIndex: number
    ): Match {
        let best: Match = { index: start, score: Number.POSITIVE_INFINITY };

        for (let index = start; index <= end; index += 1) {
            const candidate = this.normalizeForMatch(sourceLines.slice(index, index + lineCount));
            const denominator = Math.max(normalizedExpected.length, candidate.length, 1);
            const score = this.levenshtein(normalizedExpected, candidate) / denominator;
            const isCloser =
                Math.abs(index - expectedIndex) < Math.abs(best.index - expectedIndex);

            if (score < best.score || (score === best.score && isCloser)) {
                best = { index, score };
            }
            if (score === 0 && index === expectedIndex) {
                break;
            }
        }
        return best;
    }

    private normalizeForMatch(lines: string[]): string {
        // Bug 2 explicitly requires comparison on non-whitespace characters.
        return lines.join('\n').replace(/\s+/gu, '');
    }

    /** Memory-efficient Levenshtein distance: O(min(a,b)) storage. */
    private levenshtein(first: string, second: string): number {
        if (first === second) {
            return 0;
        }
        if (first.length === 0) {
            return second.length;
        }
        if (second.length === 0) {
            return first.length;
        }
        if (first.length > second.length) {
            return this.levenshtein(second, first);
        }

        let previous = Array.from({ length: first.length + 1 }, (_, index) => index);
        for (let secondIndex = 1; secondIndex <= second.length; secondIndex += 1) {
            const current = [secondIndex];
            for (let firstIndex = 1; firstIndex <= first.length; firstIndex += 1) {
                const substitutionCost =
                    first[firstIndex - 1] === second[secondIndex - 1] ? 0 : 1;
                current[firstIndex] = Math.min(
                    current[firstIndex - 1] + 1,
                    previous[firstIndex] + 1,
                    previous[firstIndex - 1] + substitutionCost
                );
            }
            previous = current;
        }
        return previous[first.length];
    }

    private splitLines(text: string): string[] {
        if (text.length === 0) {
            return [];
        }
        const lines = text.replace(/\r\n/g, '\n').split('\n');
        if (lines[lines.length - 1] === '') {
            lines.pop();
        }
        return lines;
    }

    /** Avoid splitting a UTF-16 surrogate pair at a truncation boundary. */
    private safeHead(text: string, maxLength: number): string {
        let end = Math.min(Math.max(maxLength, 0), text.length);
        if (end > 0 && end < text.length && this.isHighSurrogate(text.charCodeAt(end - 1))) {
            end -= 1;
        }
        return text.slice(0, end);
    }

    /** Avoid splitting a UTF-16 surrogate pair at a truncation boundary. */
    private safeTail(text: string, maxLength: number): string {
        let start = Math.max(0, text.length - Math.max(maxLength, 0));
        if (start > 0 && start < text.length && this.isLowSurrogate(text.charCodeAt(start))) {
            start += 1;
        }
        return text.slice(start);
    }

    /**
     * Retains a tail without beginning halfway through a markdown log line. A
     * heading boundary is preferred because known_bugs.md entries commonly start
     * with "##" or "###"; a plain newline is the deterministic fallback.
     */
    private safeTailAtBoundary(text: string, maxLength: number): string {
        const tail = this.safeTail(text, maxLength);
        if (tail.length === text.length || tail.length === 0) {
            return tail;
        }

        const heading = tail.search(/(?:^|\n)#{1,6}\s/u);
        if (heading >= 0) {
            return tail.slice(tail[heading] === '\n' ? heading + 1 : heading);
        }
        const firstNewline = tail.indexOf('\n');
        return firstNewline >= 0 ? tail.slice(firstNewline + 1) : '';
    }

    private isHighSurrogate(value: number): boolean {
        return value >= 0xd800 && value <= 0xdbff;
    }

    private isLowSurrogate(value: number): boolean {
        return value >= 0xdc00 && value <= 0xdfff;
    }

    private isNodeError(error: unknown): error is NodeJS.ErrnoException {
        return error instanceof Error && 'code' in error;
    }

    private errorMessage(error: unknown): string {
        return error instanceof Error ? error.message : String(error);
    }
}

export default WorkspaceMemory;
