@echo off
setlocal
cd /d "%~dp0"

echo === Hazard Waste Detection - GitHub Push ===
echo Remote: https://github.com/aravindhangen/hazard-waste-detection.git
echo Branch: main (3 commits, includes YOLOv9 best.pt via Git LFS)
echo.
echo When Git asks for credentials:
echo   Username: your GitHub username
echo   Password: Personal Access Token (NOT your GitHub password)
echo   Create token: https://github.com/settings/tokens  (repo scope)
echo.
pause

git branch -M main
git remote set-url origin https://github.com/aravindhangen/hazard-waste-detection.git

echo.
echo Pushing to GitHub (Git LFS uploads best.pt ~212 MB - may take 5-15 minutes)...
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
