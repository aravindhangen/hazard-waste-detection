@echo off
setlocal
cd /d "%~dp0"

echo === Hazard Waste Detection - GitHub Push ===
echo Remote: https://github.com/aravindhangen/hazard-waste-detection.git
echo.
echo BEFORE running this script:
echo   1. Open https://github.com/new
echo   2. Repository name: hazard-waste-detection
echo   3. Leave README / gitignore / license UNCHECKED
echo   4. Click Create repository
echo.
pause

git branch -M main
git remote set-url origin https://github.com/aravindhangen/hazard-waste-detection.git

echo.
echo Pushing to GitHub (Git LFS may upload best.pt - this can take several minutes)...
git push -u origin main

if errorlevel 1 (
  echo.
  echo PUSH FAILED.
  echo - Create the empty repo at https://github.com/new if you have not yet
  echo - Use a Personal Access Token as password when Git prompts for credentials
  echo   Create token: https://github.com/settings/tokens
  exit /b 1
)

echo.
echo SUCCESS. Repo: https://github.com/aravindhangen/hazard-waste-detection
endlocal
