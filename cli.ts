import { InferenceGateway } from './InferenceGateway';
import { WorkspaceMemory } from './WorkspaceMemory';
import { ToolchainBridge } from './ToolchainBridge';
import { AgentEngine } from './AgentEngine';
import * as fs from 'fs';
import * as path from 'path';
import { execSync } from 'child_process';

function printUsage() {
    console.log(`
Usage: npx tsx cli.ts [options]

Options:
  -d, --dir <path>           [Required] Absolute or relative path to the target workspace folder.
  -o, --objective <text>     Objective/Goal for the AI agent (creates project.md if missing).
  -i, --max-iterations <num> Maximum iterations to run (default: 5).
  -v, --verify <command>     Default verification command (default: "npx tsc --noEmit").
  -h, --help                 Display this help menu.
`);
}

async function runCli() {
    const args = process.argv.slice(2);
    let workspaceDir: string | null = null;
    let objective: string | null = null;
    let maxIterations = 5;
    let defaultVerification = "npx tsc --noEmit";

    for (let i = 0; i < args.length; i++) {
        const arg = args[i];
        if (arg === '-h' || arg === '--help') {
            printUsage();
            process.exit(0);
        } else if (arg === '-d' || arg === '--dir') {
            workspaceDir = args[++i];
        } else if (arg === '-o' || arg === '--objective') {
            objective = args[++i];
        } else if (arg === '-i' || arg === '--max-iterations') {
            maxIterations = parseInt(args[++i], 10);
        } else if (arg === '-v' || arg === '--verify') {
            defaultVerification = args[++i];
        }
    }

    if (!workspaceDir) {
        console.error("❌ Error: Workspace directory (--dir or -d) is required.");
        printUsage();
        process.exit(1);
    }

    const absoluteDir = path.resolve(workspaceDir);
    console.log(`📂 Target Workspace: ${absoluteDir}`);

    // Create target directory if it doesn't exist
    if (!fs.existsSync(absoluteDir)) {
        console.log(`Creating directory: ${absoluteDir}`);
        fs.mkdirSync(absoluteDir, { recursive: true });
    }

    // Check if it is a git repo. If not, initialize it!
    const gitDir = path.join(absoluteDir, '.git');
    if (!fs.existsSync(gitDir)) {
        console.log(`Initializing Git repository in: ${absoluteDir}`);
        execSync('git init', { cwd: absoluteDir, stdio: 'ignore' });
        execSync('git config user.name "AIOS Agent"', { cwd: absoluteDir, stdio: 'ignore' });
        execSync('git config user.email "aios@agent.local"', { cwd: absoluteDir, stdio: 'ignore' });
    }

    // Bootstrap necessary files
    const projectPath = path.join(absoluteDir, 'project.md');
    const architecturePath = path.join(absoluteDir, 'architecture.md');
    const knownBugsPath = path.join(absoluteDir, 'known_bugs.md');
    const gitignorePath = path.join(absoluteDir, '.gitignore');

    if (!fs.existsSync(projectPath)) {
        if (!objective) {
            console.error("❌ Error: Target workspace has no project.md. Please specify --objective to initialize it.");
            process.exit(1);
        }
        console.log(`Initializing project.md with objective...`);
        fs.writeFileSync(projectPath, `Objective: ${objective}\nCurrent Status: Repository initialized.\n`);
    } else if (objective) {
        console.log(`Overwriting objective in project.md...`);
        fs.writeFileSync(projectPath, `Objective: ${objective}\nCurrent Status: Repository initialized.\n`);
    }

    if (!fs.existsSync(architecturePath)) {
        fs.writeFileSync(architecturePath, `# System Architecture\n\nSpecify system architectural patterns and guidelines here.\n`);
    }

    if (!fs.existsSync(knownBugsPath)) {
        fs.writeFileSync(knownBugsPath, '');
    }

    if (!fs.existsSync(gitignorePath)) {
        fs.writeFileSync(gitignorePath, `node_modules/\ndist/\n*.log\n`);
    }

    // Make an initial git commit if repository is dirty (so ToolchainBridge can start with clean tree)
    try {
        const status = execSync('git status --porcelain', { cwd: absoluteDir }).toString().trim();
        if (status.length > 0) {
            console.log("Committing initial files to clean the Git tree...");
            execSync('git add .', { cwd: absoluteDir, stdio: 'ignore' });
            execSync('git commit -m "Initial bootstrap commit"', { cwd: absoluteDir, stdio: 'ignore' });
        }
    } catch (err) {
        console.error("Warning: Failed to perform initial git commit:", err);
    }

    console.log("🚀 Booting local Agent OS...");

    const gateway = new InferenceGateway("http://127.0.0.1:8080");
    const memory = new WorkspaceMemory(absoluteDir, { tokenBudget: 4000 });
    const bridge = new ToolchainBridge(absoluteDir);
    const engine = new AgentEngine(gateway, memory, bridge, {
        maxIterations,
        defaultVerificationCommand: defaultVerification,
        onEvent: (event) => {
            if (event.message === 'Orchestrator selected the next action.' && event.details) {
                const details = event.details;
                console.log(`\x1b[33m[Orchestrator]\x1b[0m Action: \x1b[1m${details.action}\x1b[0m | Targets: ${details.targetFile || 'none'} | Command: ${details.command || 'default'}`);
            } else if (event.message === 'Code transaction verified and committed.' && event.details) {
                const details = event.details;
                console.log(`\x1b[32m[Verified & Committed]\x1b[0m Files: ${(details.targetFiles as string[]).join(', ')} | Attempts: ${details.attempts} | Verification: ${details.command}`);
            } else {
                console.log(`\x1b[36m[State: ${event.state}]\x1b[0m ${event.message}`);
            }
        }
    });

    console.log("🧠 Handing control to the AgentEngine...");
    try {
        const result = await engine.run();
        console.log("\n🏁 \x1b[35m[AIOS Finished]\x1b[0m Result:");
        console.log(JSON.stringify(result, null, 2));
    } catch (err) {
        console.error("\n❌ Fatal error in AgentEngine:", err);
        process.exit(1);
    }
}

runCli().catch(console.error);
