$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$python = (Resolve-Path '../topic2/.venv/Scripts/python.exe').Path

function Write-Milestone([string]$Message) {
    $line = "$(Get-Date -Format o) $Message"
    Add-Content -LiteralPath 'logs/milestones.log' -Value $line -Encoding UTF8
    Write-Output $line
}

function Invoke-ExperimentStage([string[]]$Command) {
    Write-Milestone "BEGIN python -X utf8 -u $($Command -join ' ')"
    & $python -X utf8 -u @Command
    if ($LASTEXITCODE -ne 0) {
        throw "Stage '$($Command -join ' ')' exited with status $LASTEXITCODE"
    }
    Write-Milestone "DONE $($Command -join ' ')"
}

try {
    if (-not (Test-Path -LiteralPath 'data/native-full-retrieval.jsonl')) {
        Invoke-ExperimentStage @('scale_pipeline.py','native')
        Invoke-ExperimentStage @('scale_pipeline.py','capture')
    }
    if (-not (Test-Path -LiteralPath 'data/compiled.jsonl')) {
        if ($env:TOPIC2_OFFLINE_COMPILER_PID) {
            $compiler = Get-Process -Id ([int]$env:TOPIC2_OFFLINE_COMPILER_PID) -ErrorAction SilentlyContinue
            if ($compiler) { Wait-Process -InputObject $compiler }
        }
        for ($attempt = 1; $attempt -le 3; $attempt++) {
            if (Test-Path -LiteralPath 'data/compiled.jsonl') { break }
            try {
                Invoke-ExperimentStage @('scale_pipeline.py','compile')
            } catch {
                if ($attempt -eq 3) { throw }
                Write-Milestone "Retry offline compilation from its journal after $($_.Exception.Message)"
                Start-Sleep -Seconds 30
            }
        }
    }
    if (-not (Test-Path -LiteralPath 'results/config.json')) {
        Invoke-ExperimentStage @('run_abcd.py','prepare')
    }
    Invoke-ExperimentStage @('run_abcd.py','run')
    Invoke-ExperimentStage @('report_scale.py')
    Write-Milestone 'COMPLETE results/summary.json'
} catch {
    Write-Milestone "FAILED $($_.Exception.Message)"
    throw
}
