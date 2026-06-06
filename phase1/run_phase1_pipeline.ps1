param(
  [Parameter(Mandatory = $true)]
  [string]$Topic,

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
$script = Join-Path $root "scripts\run_phase1_pipeline.py"
$argsList = @($script, "--topic", $Topic, "--model-profile", $ModelProfile)
if ($RunId) {
  $argsList += @("--run-id", $RunId)
}

& python @argsList
exit $LASTEXITCODE
