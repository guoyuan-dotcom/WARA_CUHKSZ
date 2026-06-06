param(
  [Parameter(Mandatory = $false)]
  [string]$Phase2Run,

  [Parameter(Mandatory = $false)]
  [string]$Topic,

  [Parameter(Mandatory = $false)]
  [string]$ModelProfile,

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

$argsList = @((Join-Path $root "scripts\run_phase3_pipeline.py"))
if ($Phase2Run) {
  $argsList += @("--phase2-run", $Phase2Run)
}
if ($Topic) {
  $argsList += @("--topic", $Topic)
}
if ($ModelProfile) {
  $argsList += @("--model-profile", $ModelProfile)
}
if ($RunId) {
  $argsList += @("--run-id", $RunId)
}
python @argsList
exit $LASTEXITCODE
