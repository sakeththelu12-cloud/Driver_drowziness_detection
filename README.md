# Model files go here

After training (`model/train.py`) and converting (`model/convert_to_tfjs.py`),
place the generated `model.json` and `*.bin` shard files directly in this
folder. `frontend/script.js` loads them from `model/model.json` by default.

This folder is intentionally empty in the repo until you generate your own
trained model — the training dataset and resulting weights aren't included
here.
