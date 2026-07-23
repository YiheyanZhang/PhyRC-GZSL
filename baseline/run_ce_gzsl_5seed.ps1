param(
    [string]$Python = "D:\miniconda3\envs\spectral\python.exe"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$outputDir = Join-Path $root "checkpoints\baselines\ce_gzsl"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$datasets = @(
    @{Name="paviau"; Config="configs/paviau_p1.yaml"; Attr="data/processed/PaviaU_structured_attributes.json"; Pattern="checkpoints/paviau_p1_backbone_s{unseen}.pt"; Classes=1..9},
    @{Name="houston"; Config="configs/houston_p1.yaml"; Attr="data/processed/Houston_structured_attributes.json"; Pattern="checkpoints/houston_p1_backbone_s{unseen}.pt"; Classes=1..15},
    @{Name="longkou"; Config="configs/longkou_p1.yaml"; Attr="data/processed/LongKou_structured_attributes.json"; Pattern="checkpoints/longkou_p1_backbone_s{unseen}.pt"; Classes=1..9}
)

foreach ($dataset in $datasets) {
    $processes = @()
    foreach ($seed in 42..46) {
        $output = "checkpoints/baselines/ce_gzsl/$($dataset.Name)_seed$seed.json"
        $stdout = Join-Path $outputDir "$($dataset.Name)_seed$seed.stdout.log"
        $stderr = Join-Path $outputDir "$($dataset.Name)_seed$seed.stderr.log"
        $arguments = @(
            "-u", "baseline/evaluate_ce_gzsl.py",
            "--config", $dataset.Config,
            "--attributes", $dataset.Attr,
            "--unseen-classes"
        ) + $dataset.Classes + @(
            "--backbone-pattern", $dataset.Pattern,
            "--seed", $seed,
            "--output", $output,
            "--gan-epochs", 100,
            "--classifier-epochs", 25,
            "--synthetic-per-class", 100,
            "--batch-size", 2048,
            "--hidden-dim", 1024,
            "--embedding-dim", 512,
            "--projection-dim", 128,
            "--relation-hidden-dim", 512
        )
        $process = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
        $processes += @{Process=$process; Seed=$seed; ErrorLog=$stderr}
    }
    foreach ($entry in $processes) {
        $entry.Process.WaitForExit()
        if ($entry.Process.ExitCode -ne 0) {
            $details = Get-Content -Raw $entry.ErrorLog
            throw "$($dataset.Name) seed $($entry.Seed) failed: $details"
        }
    }
    Write-Output "completed $($dataset.Name) seeds 42-46"
}
