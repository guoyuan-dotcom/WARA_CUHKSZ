param(
  [Parameter(Mandatory = $false)]
  [string]$Topic,

  [Parameter(Mandatory = $false)]
  [string]$Phase1Run,

  [switch]$SkipPhase1,

  [Parameter(Mandatory = $false)]
  [string]$ModelProfile = "kimi-k2.6-no-thinking",

  [Parameter(Mandatory = $false)]
  [string]$RunId
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$envKeys = @(
  "KIMI_API_KEY",
  "KIMI_BASE_URL",
  "MOONSHOT_API_KEY",
  "MOONSHOT_BASE_URL",
  "OPENAI_API_KEY",
  "OPENAI_BASE_URL",
  "DEEPSEEK_API_KEY",
  "DEEPSEEK_BASE_URL"
)
foreach ($name in $envKeys) {
  if (-not [Environment]::GetEnvironmentVariable($name, "Process")) {
    $userValue = [Environment]::GetEnvironmentVariable($name, "User")
    if ($userValue) {
      [Environment]::SetEnvironmentVariable($name, $userValue, "Process")
    }
  }
}
$argsList = @((Join-Path $root "run_wara_pipeline.py"))
if ($Topic) {
  $argsList += @("--topic", $Topic)
}
if ($Phase1Run) {
  $argsList += @("--phase1-run", $Phase1Run)
}
if ($SkipPhase1) {
  $argsList += "--skip-phase1"
}
if ($ModelProfile) {
  $argsList += @("--model-profile", $ModelProfile)
}
if ($RunId) {
  $argsList += @("--run-id", $RunId)
}
python @argsList
exit $LASTEXITCODE
