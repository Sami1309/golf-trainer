# Manual Deployment Steps

## Recommended Host

Use GitHub Pages deployed by GitHub Actions.

Why this is the right first deploy:

- The app is already static: `frontend/index.html`, browser JS, and JSON model files.
- GitHub Pages gives a free HTTPS URL, which iPhone Safari needs for microphone access.
- No backend, database, secrets, build service, or paid account is needed.
- The included Actions workflow runs static checks before publishing `frontend/`.

Cloudflare Pages is the fallback if you need easier private-repo hosting or branch preview URLs. For the current range-testing goal, GitHub Pages is the minimal path.

## One-Time GitHub Setup

1. Push this repo to GitHub.
2. In GitHub, open the repo settings.
3. Go to **Pages**.
4. Under **Build and deployment**, set **Source** to **GitHub Actions**.
5. Push to `main`, or run **Actions** -> **Deploy Pages** -> **Run workflow**.
6. Open the deployed URL shown in the deploy job summary.

No repository secrets are required.

## Before Pushing

Run this locally:

```sh
npm run check
git status --short --ignored
```

Stage paths explicitly. Do not use broad `git add .` until you have confirmed raw `.m4a`, `.MOV`, generated WAV clips, exports, and session dumps are still ignored.

## iPhone Test

1. Open the GitHub Pages HTTPS URL in iPhone Safari.
2. Add a tester name in the query string if useful, for example:

   ```text
   https://OWNER.github.io/REPO/?tester=sam
   ```

3. Tap **Start recording**.
4. Allow microphone access.
5. Keep Safari in the foreground during the session.
6. Hit one calibration shot and confirm it.
7. Hit real shots and deliberate non-shot transients.
8. Export all detections at the end of the session.

The models run locally in the browser from:

- `frontend/models/stage1b_detector.json`
- `frontend/models/stage2_pure_fat.json`

## Optional Custom Domain

Add the custom domain in GitHub repo settings under **Pages**, then point DNS at GitHub Pages. Wait for GitHub to issue HTTPS before iPhone testing.

## Known Deployment Tradeoffs

- ZIP export depends on JSZip loaded from jsDelivr. If range connectivity is weak or export must work fully offline after page load, vendor JSZip into `frontend/vendor/` in a later pass.
- Browser storage is local to the iPhone and site origin. Export important sessions before clearing Safari data or changing domains.
- The CI does not retrain models. It checks syntax, required frontend assets, and model/frontend feature compatibility. Retraining remains an intentional local workflow because raw media and generated WAV payloads are not normal Git artifacts.
