# Charge .env puis lance le serveur MLflow (à lancer depuis la racine du projet)
Get-Content .env | Where-Object { $_ -match '^\s*[^#\s].*=' } | ForEach-Object {
    $k, $v = $_ -split '=', 2
    Set-Item -Path "Env:$($k.Trim())" -Value $v.Trim()
}
if (-not $env:MLFLOW_ARTIFACT_BUCKET) {
    Write-Error "MLFLOW_ARTIFACT_BUCKET est absent de .env"
    exit 1
}
$root = "s3://$($env:MLFLOW_ARTIFACT_BUCKET)"
Write-Host "Artefacts MLflow -> $root   (endpoint $env:MLFLOW_S3_ENDPOINT_URL)"
mlflow server --host 127.0.0.1 --port 5000 `
  --backend-store-uri sqlite:///mlflow.db `
  --default-artifact-root $root