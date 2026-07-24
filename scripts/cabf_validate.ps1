param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ArgsList
)

python "$PSScriptRoot\cabf_validate.py" @ArgsList
