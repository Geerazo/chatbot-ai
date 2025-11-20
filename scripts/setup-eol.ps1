# scripts/setup-eol.ps1
param(
  [string]$RepoPath = "."
)

Set-Location $RepoPath

# 1) Política local recomendada en Windows:
#    - core.autocrlf=true: convierte CRLF<->LF en checkout/commit
#    - core.eol=lf: el repo queda en LF (refuerza .gitattributes)
git config core.autocrlf true
git config core.eol lf

# 2) Asegurar que .gitattributes esté trackeado
if (-not (Test-Path ".gitattributes")) {
  Write-Host "No se encontró .gitattributes en $((Get-Location).Path)" -ForegroundColor Yellow
  exit 1
}

# 3) Renormalizar el árbol de trabajo (aplica reglas nuevas a todos los files)
git add --renormalize .
git commit -m "chore: enforce LF with .gitattributes and renormalize" --no-verify
