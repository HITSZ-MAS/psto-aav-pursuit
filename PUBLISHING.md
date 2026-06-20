# Publishing

Create an empty GitHub repository first, then run the commands below from this directory.

```bash
git init
git branch -M main
git add .
git commit -m "Release paper code"
git remote add origin git@github.com:<your-user-or-org>/<repo-name>.git
git push -u origin main
```

If the server has no Git identity configured, set it locally before committing:

```bash
git config user.name "<your name>"
git config user.email "<your email or GitHub noreply email>"
```

Do not commit generated files, checkpoints, W&B folders, Isaac Sim caches, or local experiment
outputs. They are already covered by `.gitignore`.
