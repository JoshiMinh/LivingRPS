Set objShell = CreateObject("WScript.Shell")
objShell.Run "cmd /c py main.py", 0, True
Set objShell = Nothing