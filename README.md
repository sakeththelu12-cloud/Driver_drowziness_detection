# Driver Drowsiness Detection

Real-time driver drowsiness detection using a fine-tuned EfficientNetV2-S CNN.
Training happens in Python/TensorFlow (Kaggle/Colab), and the trained model
runs **live in the browser** via TensorFlow.js — using the driver's webcam —
so the whole thing can be hosted for free on **GitHub Pages** with no backend
server required.

---

## 📁 Project Structure

```
drowsiness-detection/
├── README.md                  ← you are here
├── requirements.txt           ← Python deps for training
├── model/
│   ├── train.py                ← trains & saves the Keras model
│   └── convert_to_tfjs.py      ← converts .keras model → TensorFlow.js format
├── frontend/                   ← static site (this is what GitHub Pages serves)
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   └── model/                  ← put your converted tfjs model files here
├── backend/                     ← OPTIONAL: Flask API if you'd rather run
│   ├── app.py                     inference server-side instead of in-browser
│   └── requirements.txt
└── .gitignore
```

---

## 🗺️ Full Roadmap (start to finish)

### Step 1 — Train the model
1. Open Kaggle Notebook (or Colab) with a GPU runtime.
2. Attach the dataset: `rakibuleceruet/drowsiness-prediction-dataset`.
3. Run `model/train.py` (identical logic to what you already had — warm-up +
   fine-tune phases with EfficientNetV2-S). It saves `drowsiness_model.keras`.
4. Download `drowsiness_model.keras` to your local machine.

### Step 2 — Convert the model for the browser
Browsers can't run `.keras` files directly — convert to TensorFlow.js format:

```bash
pip install tensorflowjs
python model/convert_to_tfjs.py \
    --input drowsiness_model.keras \
    --output frontend/model
```

This produces `frontend/model/model.json` + `.bin` weight shard(s).
**These files are what actually get committed to GitHub and loaded by the
browser at runtime.**

> ⚠️ EfficientNetV2-S is ~80MB+. GitHub has a 100MB per-file limit and a soft
> repo-size warning around 1GB — a converted model this size usually fits,
> but if you hit limits, use [Git LFS](https://git-lfs.github.com/) for the
> `.bin` files (instructions below), or switch to a lighter backbone
> (e.g. MobileNetV2/EfficientNetV2-B0) for a smaller model.

### Step 3 — Test the frontend locally
```bash
cd frontend
python -m http.server 8000
```
Open `http://localhost:8000` in your browser, allow webcam access, and
confirm predictions are showing.

### Step 4 — Push to GitHub
```bash
cd drowsiness-detection
git init
git add .
git commit -m "Initial commit: drowsiness detection app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

### Step 5 — Deploy on GitHub Pages
1. Go to your repo on GitHub → **Settings → Pages**.
2. Under "Build and deployment" → Source: **Deploy from a branch**.
3. Branch: `main`, Folder: `/frontend` (or move `frontend/` contents to
   `/docs` and select that folder — GitHub Pages only offers `/root` or
   `/docs` as folder choices, so the simplest path is renaming `frontend/`
   to `docs/` before pushing, or using a GitHub Action to publish `frontend/`
   to a `gh-pages` branch — see note below).
4. Save. GitHub gives you a live URL like:
   `https://<your-username>.github.io/<your-repo>/`
5. Wait 1–2 minutes, then visit the URL — it's live.

**Simplest folder setup:** just rename `frontend/` → `docs/` in this repo
before pushing, and point GitHub Pages at the `docs/` folder. That avoids
any GitHub Actions complexity.

### Step 6 (optional) — Using Git LFS for a large model
```bash
git lfs install
git lfs track "frontend/model/*.bin"
git add .gitattributes
git add frontend/model
git commit -m "Add model weights via LFS"
git push
```

---

## 🖥️ Frontend features (what you get)

- Live webcam feed in the browser
- Runs the CNN **client-side** (TensorFlow.js) — no server, no API costs,
  no round-trip latency, works offline once loaded
- Real-time "Alert ✅ / Drowsy 🚫" status with confidence score
- Rolling drowsiness score (averages last N frames to avoid flicker/false
  alarms from a single bad frame)
- Audible alarm when drowsy state is sustained for 2+ seconds
- Simple, clean, responsive UI — works on desktop and mobile browsers

---

## 🔁 Alternative: server-side inference (backend/)

If you'd rather not ship model weights to the browser (e.g. bigger models,
privacy, or you want to log predictions), use the included Flask API instead:

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Then point `frontend/script.js`'s `MODE` constant to `"server"` and set
`API_URL` to your backend's address. Note: **this backend cannot be hosted
on GitHub Pages** (Pages is static-file-only) — deploy it separately on a
service like Render, Railway, Fly.io, or Hugging Face Spaces, then have your
GitHub Pages frontend call that API URL.

---

## 📊 Model details

- Backbone: `EfficientNetV2S` (ImageNet weights), two-phase training:
  1. Warm-up: frozen backbone, train the classification head
  2. Fine-tune: unfreeze the last 30% of backbone layers at a low LR
- Input: 224×224×3 RGB, `include_preprocessing=True` (no manual `/255`)
- Output: single sigmoid unit — `0 = Alert`, `1 = Drowsy`
- Metrics tracked: accuracy, AUC, precision, recall

---

## ✅ Quick checklist

- [ ] Train model in `model/train.py`
- [ ] Convert to TF.js with `model/convert_to_tfjs.py`
- [ ] Drop converted files into `frontend/model/`
- [ ] Test locally with `python -m http.server`
- [ ] `git init`, commit, push to GitHub
- [ ] Rename `frontend/` → `docs/` (or set up Pages source correctly)
- [ ] Enable GitHub Pages in repo settings
- [ ] Share your live `github.io` link 🎉
