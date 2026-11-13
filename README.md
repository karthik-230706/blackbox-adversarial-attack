# Black-Box Adversarial Attack on a Pretrained CNN

Hard-label (decision-based) black-box attack that produces one adversarial PNG for each of the 1000 clean images.
The attacker only sends images to the CNN and reads the **predicted label**. Scores, weights, architecture and
intermediate outputs are never used.

## Repository contents

| Path | Description |
|---|---|
| `attack.ipynb` | Complete attack code with final parameters |
| `adversarial_images/` | 1000 adversarial `.png` files (same filenames as the clean images) |
| `results.csv` | Per-image log: labels, success, L2, Linf, number of queries |
| `make_report.py` | Builds `report.pdf` from `results.csv` |
| `report.pdf` | Short description of the method and results |
| `requirements.txt` | Python dependencies |

## How to run

1. Install dependencies: `pip install -r requirements.txt`
2. Place the shared `model.pt` next to the notebook and unzip the clean images into `clean_images/`.
3. Open `attack.ipynb`, check the *Configuration* cell, and make sure `preprocess()` matches the preprocessing
   used in the organisers' notebook (normalisation / resizing).
4. Run all cells. The run is resumable: images already in `adversarial_images/` are skipped.
5. Run `python make_report.py` to regenerate `report.pdf` with your actual numbers.

## Method (summary)

Untargeted, L2, hard-label **HopSkipJump**-style attack, implemented from scratch:

1. **Initialisation:** find any image the model labels differently from the clean image (random uniform noise),
   then binary-search along the line to the clean image to land just on the adversarial side of the decision boundary.
2. **Iterate** (`N_ITERS` times): estimate the boundary's gradient direction from label-only queries around the
   current point (Monte-Carlo sign estimate with baseline subtraction), take a geometric-progression step along it,
   then binary-search back toward the clean image. Each iteration shrinks the perturbation.
3. **Quantise and verify:** round to 8-bit, re-query the exact PNG pixel values, and nudge slightly further past
   the boundary only if the rounded image is no longer misclassified.
4. **Retry:** an image that fails is re-attacked with a new seed and more iterations.

Queries are counted per image (clean-label query, initialisation, gradient estimates, line searches and the
quantisation checks) and logged in `results.csv`, which supports the tie-breaking criterion (fewer inferences).

## Final parameters

| Parameter | Value |
|---|---|
| Iterations | 10 (+5 per retry) |
| Initial / max evaluations per gradient estimate | 20 / 200 |
| Gamma | 1.0 |
| Binary-search tolerance | 1e-3 |
| Max retries | 3 |
| Seed | 0 |

## Results

Fill in after running the notebook (the last cell prints these values):

| Metric | Value |
|---|---|
| Attack success rate | _ % |
| Mean L2 (pixel range 0-1) | _ |
| Mean Linf (0-255 scale) | _ |
| Mean PSNR / SSIM | _ / _ |
| Mean queries per image | _ |
