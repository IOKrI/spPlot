# spPlot
A little python tool to plot data exported from spTool
README.MD work in Progress

This program requires a working python installation. In case none is present, it can be obtained from https://www.python.org/downloads/.

# Installation on Linux: 
Execute ```install_linux.sh```.
This will install the python dependencies needed and add the command "spPlot" to your bashrc.
The Program then can be executed by running the command ```spPlot``` from a terminal from the directory you want to work in.

# Installation on Windows:
If the execution of scripts is allowed, open a powershell in the main program folder and execute the ```setup_windows.ps1``` script by ruunning 
```
./setup_windows.ps1
```
If the execution is not allowed, you have to execute the code manually inside the powershell:
```
py -m ensurepip --upgrade
py -m pip install numpy pandas matplotlib scipy
```
This will check if ```pip``` is installed and then install the packages needed to run ```spPlot```.
After that, execute
```
((get-content -path spPlot.bat -raw) -replace 'SPDIR', (pwd | select-string -pattern 'C:*')) | set-content -path spPlot.bat
```
to set the path to the directory ```spPlot``` is located in the file ```spPlot.bat```.

The Program can be run by executing ```spPlot.bat``` from anywhere on your PC (e.g. the Desktop). 


WIP:
- GUI overhaul
- history of saved images
- manual (maybe)
