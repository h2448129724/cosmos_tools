param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ArgsList
)

python "$PSScriptRoot\cabf_flow.py" predict-points @ArgsList
