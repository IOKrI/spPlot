py -m ensurepip --upgrade
py -m pip install numpy pandas matplotlib scipy
((get-content -path spPlot.bat -raw) -replace 'SPDIR', (pwd | select-string -pattern 'C:*')) | set-content -path spPlot.bat
Write-Output "Setup complete. Have fun using spPlot!"
