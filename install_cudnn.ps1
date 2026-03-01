# Copy cuDNN files to CUDA 11.2
$cudnnPath = "C:\Users\fabio\Downloads\cudnn_temp\cuda"
$cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2"

Write-Host "Copying cuDNN files..."
Copy-Item "$cudnnPath\bin\*.dll" -Destination "$cudaPath\bin" -Force
Write-Host "DLLs copied"
Copy-Item "$cudnnPath\include\*.h" -Destination "$cudaPath\include" -Force
Write-Host "Headers copied"
Copy-Item "$cudnnPath\lib\x64\*.lib" -Destination "$cudaPath\lib\x64" -Force
Write-Host "Libraries copied"
Write-Host "cuDNN installation complete!"
