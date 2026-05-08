@echo off
rem Optional local machine settings. Copy this file to local_settings.bat.
rem local_settings.bat is ignored by git.

rem If onnxruntime-gpu cannot find cudnn64_9.dll, point this to a folder
rem that contains cudnn64_9.dll and the related CUDA DLLs.
rem Example:
rem set "FACE_COMPARE_CUDA_DLL_DIRS=C:\path\to\torch\lib"
