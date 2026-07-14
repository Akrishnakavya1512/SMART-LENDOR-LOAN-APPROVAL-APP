# GitHub and Deployment Guide

This project is ready to upload to GitHub and deploy to a Python web host.
GitHub stores the code, but it does not run Flask applications by itself. For a
live website, connect the GitHub repository to IBM Cloud, Render, Railway, or a
similar host.

## Before Uploading

Make sure these files are present:

- `app.py`
- `requirements.txt`
- `requirements-training.txt`
- `Procfile`
- `runtime.txt`
- `manifest.yml`
- `render.yaml`
- `.python-version`
- `data/loan_applicants_sample.csv`
- `models/smart_lender_model.pkl`
- `templates/`
- `static/`
- `src/`

Do not upload `.venv/`, `__pycache__/`, `work/`, or `outputs/`. They are already
ignored in `.gitignore`.

## Option 1: Publish With GitHub Desktop

1. Install GitHub Desktop from `https://desktop.github.com/`.
2. Open GitHub Desktop and sign in.
3. Choose **File > Add Local Repository**.
4. Select this folder:

   ```text
   C:\Users\omtiw\Documents\Codex\2026-06-27\smart-lender-is-a-machine-learning
   ```

5. If asked, choose **Create a repository**.
6. Commit all files with the message:

   ```text
   Initial Smart Lender Flask ML app
   ```

7. Click **Publish repository**.

## Option 2: Publish With Git Commands

Create a new empty repository on GitHub first. Do not add a README there because
this project already has one.

Then run:

```powershell
cd "C:\Users\omtiw\Documents\Codex\2026-06-27\smart-lender-is-a-machine-learning"
git init
git add .
git commit -m "Initial Smart Lender Flask ML app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/smart-lender.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

## Deploy on Render

1. Push the project to GitHub.
2. Go to `https://render.com/`.
3. Create a new **Web Service**.
4. Connect your GitHub repository.
5. Render can use `render.yaml` automatically. If entering values manually:

   ```text
   Build Command: pip install -r requirements.txt
   Start Command: gunicorn app:app
   Health Check Path: /health
   ```

6. Add an environment variable:

   ```text
   SECRET_KEY = any-long-random-text
   ```

7. If the service was created manually, also add:

   ```text
   PYTHON_VERSION = 3.12.13
   ```

8. Deploy.

## Deploy on IBM Cloud

1. Install and log in to the IBM Cloud CLI.
2. Target Cloud Foundry:

   ```powershell
   ibmcloud login
   ibmcloud target --cf
   ```

3. From the project folder, run:

   ```powershell
   ibmcloud cf push
   ```

The included `manifest.yml` runs the app with:

```text
gunicorn app:app
```

## Keep It Working After Publishing

- Keep `models/smart_lender_model.pkl` in the repository unless your host trains
  the model during deployment.
- Do not commit `.venv/`; the host installs packages from `requirements.txt`.
- Keep `requirements.txt` updated whenever you add a package.
- Keep training-only libraries in `requirements-training.txt` so cloud deploys stay faster.
- Use `/health` to confirm the deployed app can load the model.
- If the model file is missing, run `python -m src.train_model` locally and push
  the updated `models/` files.
