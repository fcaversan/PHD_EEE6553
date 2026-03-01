# Add CUDA 11.2 to system PATH
$cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin"
$existingPath = [Environment]::GetEnvironmentVariable("PATH", "Machine")

if ($existingPath -notlike "*$cudaPath*") {
    [Environment]::SetEnvironmentVariable("PATH", "$cudaPath;$existingPath", "Machine")
    Write-Host "CUDA 11.2 added to system PATH successfully!"
} else {
    Write-Host "CUDA 11.2 is already in PATH"
}
