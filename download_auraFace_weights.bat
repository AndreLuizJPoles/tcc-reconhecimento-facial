@echo off
setlocal

set "MODEL_DIR=%~dp0Models"
if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

echo ============================================
echo Baixando AuraFace FP32 (glintr100.onnx)...
echo ============================================
curl -L -o "%MODEL_DIR%\glintr100.onnx" "https://huggingface.co/fal/AuraFace-v1/resolve/main/glintr100.onnx"
if %errorlevel% neq 0 (
    echo [ERRO] Falha no download do AuraFace FP32.
) else (
    echo [OK] AuraFace salvo em %MODEL_DIR%\glintr100.onnx
)

echo.
echo ============================================
echo Baixando YuNet (face_detection_yunet_2023mar.onnx)...
echo ============================================
curl -L -o "%MODEL_DIR%\face_detection_yunet_2023mar.onnx" "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
if %errorlevel% neq 0 (
    echo [ERRO] Falha no download do YuNet.
) else (
    echo [OK] YuNet salvo em %MODEL_DIR%\face_detection_yunet_2023mar.onnx
)

echo.
echo Download concluido.
pause
endlocal