// InferenceGateway.ts
// Handles the low-level communication with the llama-server via HTTP.
// Implements GBNF grammar enforcement and millisecond LoRA multiplexing.

export interface LoRAState {
    id: string;
    scale: number;
}

export interface CompletionOptions {
    prompt: string;
    max_tokens?: number;
    temperature?: number;
    seed?: number;
    grammar?: string; // GBNF grammar string for strict JSON enforcement
}

export class InferenceGateway {
    private serverUrl: string;

    constructor(serverUrl: string = 'http://127.0.0.1:8080') {
        this.serverUrl = serverUrl;
    }

    /**
     * Phase 3: LoRA Multiplexing
     * Hot-swaps the active persona in milliseconds.
     */
    async swapLoRA(adapters: LoRAState[]): Promise<boolean> {
        try {
            // Map the internal v1 names to the numeric adapter IDs (0 = coder, 1 = orchestrator)
            // based on the load order in start-inference-server.bat
            const payload = adapters.map(a => {
                let realId = 0; // Default to 0
                if (a.id === 'orchestrator_v1') realId = 1;
                if (a.id === 'coder_v1') realId = 0;
                if (a.id === 'debugger_v1') realId = 0; 
                return { id: realId, scale: a.scale };
            });

            console.log(`[Gateway] Swapping LoRA adapters:`, payload);
            
            const response = await fetch(`${this.serverUrl}/lora-adapters`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                console.warn(`[Gateway] LoRA swap returned non-200. Check server logs.`);
            }
            
            return true;
        } catch (error) {
            console.error('[Gateway] Failed to swap LoRA:', error);
            return false;
        }
    }

    /**
     * Strips hallucinated Markdown wrappers from the LLM output before parsing.
     */
    private sanitizeLLMOutput(rawContent: string): string {
        return rawContent
            .replace(/^```[a-zA-Z]*\s*/i, '')
            .replace(/\s*```$/i, '')
            .trim();
    }

    /**
     * Phase 2: Deterministic Grammar (GBNF)
     * Requests a completion, optionally enforcing a strict grammar.
     */
    async requestCompletion(options: CompletionOptions): Promise<string> {
        try {
            // Enforcement: Deterministic JSON Outputs
            const payload = {
                prompt: options.prompt,
                n_predict: options.max_tokens ?? 2048,
                temperature: 0, // Rigidly locked
                seed: 42,       // Rigidly locked
                grammar: options.grammar ?? undefined,
                stream: false
            };

            console.log(`[Gateway] Requesting completion...`);

            const response = await fetch(`${this.serverUrl}/completion`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`Completion failed with status: ${response.status}`);
            }

            const data = await response.json();
            return this.sanitizeLLMOutput(data.content);

        } catch (error) {
            console.error('[Gateway] Inference failed:', error);
            throw error;
        }
    }
}
