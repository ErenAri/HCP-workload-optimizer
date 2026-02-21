param(
    [string]$TraceDataset = "",
    [string]$RawTrace = "",
    [string]$OutDir = "outputs/verify",
    [int]$CapacityCpus = 64,
    [int]$BenchmarkSamples = 3,
    [double]$BenchmarkMaxDrop = 0.10,
    [switch]$StrictQuality,
    [switch]$SkipCorrectness,
    [switch]$SkipBenchmark,
    [switch]$SkipLoad,
    [switch]$SkipScience,
    [switch]$SkipRepro
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:GateResults = @()

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [string]$Exe,
        [Parameter(Mandatory = $true)]
        [string[]]$Args
    )

    Write-Host ""
    Write-Host ">>> $Label" -ForegroundColor Cyan
    Write-Host "$Exe $($Args -join ' ')" -ForegroundColor DarkGray
    & $Exe @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $Label"
    }
}

function Invoke-HpcoptCli {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [string[]]$Args
    )

    $cliArgs = @("-c", "from hpcopt.cli.main import run; run()") + $Args
    Invoke-CheckedCommand -Label $Label -Exe "python" -Args $cliArgs
}

function Invoke-Gate {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Action
    )

    $started = Get-Date
    Write-Host ""
    Write-Host "=== Gate: $Name ===" -ForegroundColor Yellow
    try {
        & $Action
        $elapsed = [math]::Round(((Get-Date) - $started).TotalSeconds, 2)
        $script:GateResults += [pscustomobject]@{
            gate = $Name
            status = "pass"
            seconds = $elapsed
        }
        Write-Host "Gate passed: $Name (${elapsed}s)" -ForegroundColor Green
    } catch {
        $elapsed = [math]::Round(((Get-Date) - $started).TotalSeconds, 2)
        $script:GateResults += [pscustomobject]@{
            gate = $Name
            status = "fail"
            seconds = $elapsed
        }
        Write-Host "Gate failed: $Name (${elapsed}s)" -ForegroundColor Red
        throw
    }
}

function Resolve-RequiredPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathValue,
        [Parameter(Mandatory = $true)]
        [string]$Label
    )
    if (-not (Test-Path -LiteralPath $PathValue)) {
        throw "$Label does not exist: $PathValue"
    }
    return (Resolve-Path -LiteralPath $PathValue).Path
}

function Assert-JsonFieldEquals {
    param(
        [Parameter(Mandatory = $true)]
        [string]$JsonPath,
        [Parameter(Mandatory = $true)]
        [string]$FieldName,
        [Parameter(Mandatory = $true)]
        [string]$Expected
    )

    $payload = Get-Content -Raw -LiteralPath $JsonPath | ConvertFrom-Json
    $actual = [string]$payload.$FieldName
    if ($actual -ne $Expected) {
        throw "Expected '$FieldName' to be '$Expected' in $JsonPath, got '$actual'."
    }
}

function Assert-JsonFieldIn {
    param(
        [Parameter(Mandatory = $true)]
        [string]$JsonPath,
        [Parameter(Mandatory = $true)]
        [string]$FieldName,
        [Parameter(Mandatory = $true)]
        [string[]]$Allowed
    )

    $payload = Get-Content -Raw -LiteralPath $JsonPath | ConvertFrom-Json
    $actual = [string]$payload.$FieldName
    if ($Allowed -notcontains $actual) {
        throw "Expected '$FieldName' in $JsonPath to be one of [$($Allowed -join ', ')], got '$actual'."
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $repoRoot
try {
    $pythonPackageRoot = Join-Path $repoRoot "python"
    if (-not $env:PYTHONPATH) {
        $env:PYTHONPATH = $pythonPackageRoot
    } elseif (-not ($env:PYTHONPATH.Split([IO.Path]::PathSeparator) -contains $pythonPackageRoot)) {
        $env:PYTHONPATH = "$pythonPackageRoot$([IO.Path]::PathSeparator)$env:PYTHONPATH"
    }

    $verifyRoot = Join-Path $repoRoot $OutDir
    $dataDir = Join-Path $verifyRoot "data"
    $reportsDir = Join-Path $verifyRoot "reports"
    $simDir = Join-Path $verifyRoot "simulations"
    foreach ($dir in @($verifyRoot, $dataDir, $reportsDir, $simDir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }

    $resolvedTraceDataset = ""
    $resolvedRawTrace = ""
    if ($TraceDataset) {
        $resolvedTraceDataset = Resolve-RequiredPath -PathValue $TraceDataset -Label "Trace dataset"
        if ($RawTrace) {
            $resolvedRawTrace = Resolve-RequiredPath -PathValue $RawTrace -Label "Raw trace"
        }
    } else {
        if ($RawTrace) {
            $resolvedRawTrace = Resolve-RequiredPath -PathValue $RawTrace -Label "Raw trace"
        } else {
            $resolvedRawTrace = Resolve-RequiredPath -PathValue "tests/fixtures/sample_trace.swf" -Label "Default raw trace fixture"
        }
        Invoke-HpcoptCli -Label "Prepare canonical dataset from raw trace" -Args @(
            "ingest", "swf",
            "--input", $resolvedRawTrace,
            "--out", $dataDir,
            "--dataset-id", "verify_sample",
            "--report-out", $reportsDir
        )
        $resolvedTraceDataset = Resolve-RequiredPath -PathValue (Join-Path $dataDir "verify_sample.parquet") -Label "Generated dataset"
    }

    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"

    if (-not $SkipCorrectness) {
        Invoke-Gate -Name "Correctness (ruff + mypy + unit/integration tests)" -Action {
            Invoke-CheckedCommand -Label "Lint (ruff)" -Exe "python" -Args @("-m", "ruff", "check", "python/")
            Invoke-CheckedCommand -Label "Type-check (mypy)" -Exe "python" -Args @(
                "-m", "mypy", "python/hpcopt/", "--ignore-missing-imports"
            )
            Invoke-CheckedCommand -Label "Tests (unit + integration)" -Exe "python" -Args @(
                "-m", "pytest", "tests/unit", "tests/integration", "-q"
            )
        }
    }

    if (-not $SkipBenchmark) {
        Invoke-Gate -Name "Efficiency (benchmark regression gate)" -Action {
            $benchmarkRunId = "verify_${stamp}_benchmark"
            $historyPath = Join-Path $reportsDir "benchmark_history.jsonl"
            $args = @(
                "report", "benchmark",
                "--trace", $resolvedTraceDataset,
                "--policy", "EASY_BACKFILL_BASELINE",
                "--capacity-cpus", "$CapacityCpus",
                "--samples", "$BenchmarkSamples",
                "--regression-max-drop", "$BenchmarkMaxDrop",
                "--history-window", "5",
                "--strict-regression",
                "--out", $reportsDir,
                "--history", $historyPath,
                "--run-id", $benchmarkRunId
            )
            if ($resolvedRawTrace) {
                $args += @("--raw-trace", $resolvedRawTrace)
            }
            Invoke-HpcoptCli -Label "Benchmark suite" -Args $args
            $reportPath = Resolve-RequiredPath -PathValue (Join-Path $reportsDir "${benchmarkRunId}_benchmark_report.json") -Label "Benchmark report"
            Assert-JsonFieldEquals -JsonPath $reportPath -FieldName "status" -Expected "pass"
            Assert-JsonFieldEquals -JsonPath $reportPath -FieldName "policy_id" -Expected "EASY_BACKFILL_BASELINE"
        }
    }

    if (-not $SkipLoad) {
        Invoke-Gate -Name "API performance (load tests)" -Action {
            Invoke-CheckedCommand -Label "Tests (load)" -Exe "python" -Args @("-m", "pytest", "tests/load", "-q")
        }
    }

    if (-not $SkipScience) {
        Invoke-Gate -Name "Scientific credibility (fidelity + recommendation)" -Action {
            $baselineRunId = "verify_${stamp}_baseline"
            $candidateRunId = "verify_${stamp}_candidate"
            $fidelityRunId = "verify_${stamp}_fidelity"
            $recommendRunId = "verify_${stamp}_recommend"

            Invoke-HpcoptCli -Label "Baseline simulation" -Args @(
                "simulate", "run",
                "--trace", $resolvedTraceDataset,
                "--policy", "EASY_BACKFILL_BASELINE",
                "--capacity-cpus", "$CapacityCpus",
                "--out", $simDir,
                "--report-out", $reportsDir,
                "--run-id", $baselineRunId,
                "--strict-invariants"
            )
            Invoke-HpcoptCli -Label "Candidate simulation" -Args @(
                "simulate", "run",
                "--trace", $resolvedTraceDataset,
                "--policy", "ML_BACKFILL_P50",
                "--capacity-cpus", "$CapacityCpus",
                "--out", $simDir,
                "--report-out", $reportsDir,
                "--run-id", $candidateRunId,
                "--strict-invariants"
            )
            Invoke-HpcoptCli -Label "Fidelity gate" -Args @(
                "simulate", "fidelity-gate",
                "--trace", $resolvedTraceDataset,
                "--capacity-cpus", "$CapacityCpus",
                "--out", $reportsDir,
                "--run-id", $fidelityRunId
            )

            $baselineReport = Resolve-RequiredPath -PathValue (Join-Path $reportsDir "${baselineRunId}_easy_backfill_baseline_sim_report.json") -Label "Baseline simulation report"
            $candidateReport = Resolve-RequiredPath -PathValue (Join-Path $reportsDir "${candidateRunId}_ml_backfill_p50_sim_report.json") -Label "Candidate simulation report"
            $fidelityReport = Resolve-RequiredPath -PathValue (Join-Path $reportsDir "${fidelityRunId}_fidelity_report.json") -Label "Fidelity report"

            Invoke-HpcoptCli -Label "Recommendation generation" -Args @(
                "recommend", "generate",
                "--baseline-report", $baselineReport,
                "--candidate-report", $candidateReport,
                "--fidelity-report", $fidelityReport,
                "--out", $reportsDir,
                "--run-id", $recommendRunId
            )
            $recommendationReport = Resolve-RequiredPath -PathValue (Join-Path $reportsDir "${recommendRunId}_recommendation_report.json") -Label "Recommendation report"
            Assert-JsonFieldIn -JsonPath $fidelityReport -FieldName "status" -Allowed @("pass", "fail")
            Assert-JsonFieldIn -JsonPath $recommendationReport -FieldName "status" -Allowed @("accepted", "blocked")
            if ($StrictQuality) {
                Assert-JsonFieldEquals -JsonPath $fidelityReport -FieldName "status" -Expected "pass"
                Assert-JsonFieldEquals -JsonPath $recommendationReport -FieldName "status" -Expected "accepted"
            }
        }
    }

    if (-not $SkipRepro) {
        Invoke-Gate -Name "Reproducibility (suite test)" -Action {
            Invoke-CheckedCommand -Label "Tests (reproducibility)" -Exe "python" -Args @(
                "-m", "pytest", "tests/unit/test_reproducibility_suite.py", "-q"
            )
        }
    }

    Write-Host ""
    Write-Host "Verification summary" -ForegroundColor Magenta
    $script:GateResults | Format-Table -AutoSize | Out-Host
    Write-Host ""
    Write-Host "All selected gates passed." -ForegroundColor Green
} finally {
    Pop-Location
}
