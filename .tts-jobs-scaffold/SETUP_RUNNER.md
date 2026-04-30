# Self-Hosted Runner Setup (WSL2)

## Prerequisites

- WSL2 with Ubuntu
- chatterbox-pipeline installed at `/home/stephan/projekte/chatterbox-pipeline/` with working venv
- `ffmpeg` installed (`sudo apt install ffmpeg`)
- NVIDIA GPU with CUDA functional

## Step 1: Create the private repo

```bash
gh repo create tts-pipeline-jobs --private --clone
cd tts-pipeline-jobs
```

## Step 2: Copy scaffold into the repo

```bash
# From wherever you have the scaffold files:
cp -r <scaffold>/.github <scaffold>/jobs <scaffold>/README.md <scaffold>/SETUP_RUNNER.md .
git add .
git commit -m "Initial: workflow + project structure"
git push
```

## Step 3: Install the GitHub Actions Runner

1. Go to: https://github.com/MoebiusSt/tts-pipeline-jobs/settings/actions/runners/new
2. Select **Linux** / **x64**
3. Follow GitHub's instructions:

```bash
mkdir -p ~/actions-runner && cd ~/actions-runner

# Download (check GitHub page for current version + token)
curl -o actions-runner-linux-x64.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.322.0/actions-runner-linux-x64-2.322.0.tar.gz
tar xzf actions-runner-linux-x64.tar.gz

# Configure with your repo-scoped token
./config.sh --url https://github.com/MoebiusSt/tts-pipeline-jobs \
  --token YOUR_TOKEN_FROM_GITHUB

# Install as systemd service (auto-start on WSL boot)
sudo ./svc.sh install
sudo ./svc.sh start
```

## Step 4: Verify

```bash
# Check service status
sudo ./svc.sh status

# Or go to repo Settings → Actions → Runners — should show "Idle"
```

## Security

- Runner is scoped to this single private repo
- Only workflows defined in this repo can execute on the runner
- No external network exposure of your GPU
- `CBPIPE_DIR` in the workflow is the only filesystem path accessed
