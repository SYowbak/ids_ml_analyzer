# Multi-agent Automation in VS Code

This setup lets you run multiple Copilot chat agents in parallel using isolated git worktrees.

## What it does
- Creates 4 dedicated branches from a chosen base ref.
- Creates 4 worktree folders (one per agent role).
- Adds an AGENT_BRIEF.md in each worktree with mission/scope/done criteria.
- Optionally opens each worktree in a new VS Code window.

## Script
- scripts/setup_multi_agent_worktrees.ps1

## Recommended flow
1. Create a backup branch and push it to GitHub.
2. Run the setup script.
3. Open each worktree window and run one focused agent per window.
4. Merge work sequentially after tests pass.

## Usage
From repository root in PowerShell:

```powershell
./scripts/setup_multi_agent_worktrees.ps1 -BaseRef main -WorktreeRoot ..\IDS_ML_Analyzer_agents -OpenInVSCode
```

If you want the latest remote refs first:

```powershell
./scripts/setup_multi_agent_worktrees.ps1 -BaseRef main -WorktreeRoot ..\IDS_ML_Analyzer_agents -FetchOrigin -OpenInVSCode
```

If you want to base worktrees on your current branch:

```powershell
./scripts/setup_multi_agent_worktrees.ps1 -WorktreeRoot ..\IDS_ML_Analyzer_agents -OpenInVSCode
```

## Output layout
The script creates:
- ../IDS_ML_Analyzer_agents/01-stability
- ../IDS_ML_Analyzer_agents/02-core-ml
- ../IDS_ML_Analyzer_agents/03-ui-report
- ../IDS_ML_Analyzer_agents/04-ci-docs

Each folder contains AGENT_BRIEF.md.

## Notes
- Existing worktree folder names are not overwritten.
- Branch names include a timestamp to avoid collisions.
- You can run script multiple times; each run creates a new set of branches.
