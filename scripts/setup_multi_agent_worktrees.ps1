[CmdletBinding()]
param(
    [string]$BaseRef = "",
    [string]$WorktreeRoot = "..\IDS_ML_Analyzer_agents",
    [switch]$OpenInVSCode,
    [switch]$FetchOrigin
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Git {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Args
    )

    $output = git @Args 2>&1
    if ($LASTEXITCODE -ne 0) {
        $joined = $Args -join " "
        throw "git $joined failed: $output"
    }

    if ($output -is [System.Array]) {
        return ($output -join "`n")
    }

    return [string]$output
}

function New-AgentBrief {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$Agent,
        [Parameter(Mandatory = $true)]
        [string]$BranchName
    )

    $scope = ($Agent.Scope -join "`n")
    $forbidden = ($Agent.Forbidden -join "`n")
    $done = ($Agent.Done -join "`n")

    return @"
# $($Agent.Title)

## Branch
$BranchName

## Mission
$($Agent.Mission)

## Scope
$scope

## Forbidden
$forbidden

## Definition of Done
$done

## Report Format
- Summary of changes
- Files changed
- Tests run and result
- Known risks
"@
}

$repoRoot = (Invoke-Git -Args @("rev-parse", "--show-toplevel")).Trim()
if ([string]::IsNullOrWhiteSpace($repoRoot)) {
    throw "Cannot resolve git repository root."
}

Push-Location $repoRoot
try {
    if ($FetchOrigin) {
        Invoke-Git -Args @("fetch", "origin", "--prune") | Out-Null
    }

    if ([string]::IsNullOrWhiteSpace($BaseRef)) {
        $BaseRef = (Invoke-Git -Args @("rev-parse", "--abbrev-ref", "HEAD")).Trim()
    }

    $null = Invoke-Git -Args @("rev-parse", "--verify", $BaseRef)

    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"

    $agents = @(
        @{
            Id = "stability"
            Folder = "01-stability"
            Title = "Stability Agent"
            Mission = "Increase reliability of threshold/scan behavior and regression confidence."
            Scope = @(
                "- tests/**"
                "- src/core/threshold_policy.py"
                "- src/ui/tabs/scanning.py"
            )
            Forbidden = @(
                "- Do not redesign UI layout"
                "- Do not touch unrelated data-loading code"
            )
            Done = @(
                "- Threshold policy behavior is deterministic"
                "- Regression tests pass"
                "- No compatibility breaks for old model manifests"
            )
        },
        @{
            Id = "core-ml"
            Folder = "02-core-ml"
            Title = "Core ML Agent"
            Mission = "Improve training/calibration robustness and model metadata quality."
            Scope = @(
                "- src/core/model_engine.py"
                "- src/ui/tabs/training.py"
                "- tests/test_model_engine_if_calibration.py"
            )
            Forbidden = @(
                "- Do not modify CI workflow files"
                "- Do not edit history/storage UI"
            )
            Done = @(
                "- Calibration path is explicit and tested"
                "- Metadata contract remains backward-compatible"
                "- Relevant tests pass"
            )
        },
        @{
            Id = "ui-report"
            Folder = "03-ui-report"
            Title = "UI Report Agent"
            Mission = "Polish scan report clarity and operator-facing decision signals."
            Scope = @(
                "- src/ui/tabs/home.py"
                "- src/ui/tabs/scanning.py"
                "- tests/test_scanning_report_quality_helpers.py"
            )
            Forbidden = @(
                "- Do not alter model serialization format"
                "- Do not change database schema"
            )
            Done = @(
                "- Report sections remain logically consistent"
                "- Risk explanation becomes clearer"
                "- UI helper tests pass"
            )
        },
        @{
            Id = "ci-docs"
            Folder = "04-ci-docs"
            Title = "CI and Docs Agent"
            Mission = "Harden CI gates and keep docs aligned with real runtime behavior."
            Scope = @(
                "- .github/workflows/**"
                "- README.md"
                "- docs/**"
            )
            Forbidden = @(
                "- Do not change model/scanning runtime logic"
                "- Do not modify training hyperparameters"
            )
            Done = @(
                "- CI has clear regression gate"
                "- Docs reflect current tabs/features"
                "- Commands in docs are executable"
            )
        }
    )

    $resolvedWorktreeRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $WorktreeRoot))
    if (-not (Test-Path $resolvedWorktreeRoot)) {
        New-Item -Path $resolvedWorktreeRoot -ItemType Directory | Out-Null
    }

    $created = @()

    foreach ($agent in $agents) {
        $branchName = "agent/$($agent.Id)-$timestamp"
        $worktreePath = Join-Path $resolvedWorktreeRoot $agent.Folder

        if (Test-Path $worktreePath) {
            throw "Worktree path already exists: $worktreePath"
        }

        Invoke-Git -Args @("worktree", "add", "-b", $branchName, $worktreePath, $BaseRef) | Out-Null

        $brief = New-AgentBrief -Agent $agent -BranchName $branchName
        $briefPath = Join-Path $worktreePath "AGENT_BRIEF.md"
        Set-Content -Path $briefPath -Value $brief -Encoding UTF8

        $created += [PSCustomObject]@{
            Agent = $agent.Title
            Branch = $branchName
            Path = $worktreePath
        }

        if ($OpenInVSCode) {
            $codeCmd = Get-Command code -ErrorAction SilentlyContinue
            if ($null -ne $codeCmd) {
                & code -n $worktreePath | Out-Null
            }
        }
    }

    Write-Host "Created agent worktrees:" -ForegroundColor Green
    $created | Format-Table -AutoSize

    Write-Host ""
    Write-Host "Next step in each worktree:" -ForegroundColor Yellow
    Write-Host "1) Open chat and paste AGENT_BRIEF.md" -ForegroundColor Yellow
    Write-Host "2) Ask Copilot to execute the mission end-to-end" -ForegroundColor Yellow
}
finally {
    Pop-Location
}
