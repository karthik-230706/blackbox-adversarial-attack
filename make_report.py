"""Builds report.pdf. Reads results.csv (if present) and fills in the measured numbers."""
import csv, os
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

rows = list(csv.DictReader(open("results.csv"))) if os.path.exists("results.csv") else []
if rows:
    n = len(rows)
    sr = 100 * np.mean([int(r["success"]) for r in rows])
    l2 = np.mean([float(r["l2"]) for r in rows]); linf = np.mean([float(r["linf_255"]) for r in rows])
    q = np.mean([int(r["queries"]) for r in rows])
    stats = [["Images attacked", str(n)], ["Attack success rate", f"{sr:.2f}%"],
             ["Mean L2 distortion (pixels in 0-1)", f"{l2:.4f}"], ["Mean Linf distortion (0-255 scale)", f"{linf:.2f}"],
             ["Mean model queries per image", f"{q:.0f}"]]
else:
    stats = [["Images attacked", "run the notebook, then rerun make_report.py"]]

st = getSampleStyleSheet()
doc = SimpleDocTemplate("report.pdf", pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm,
                        title="Black-Box Adversarial Attack on a CNN")
S = []
P = lambda t, s="BodyText": S.append(Paragraph(t, st[s]))
P("Black-Box Adversarial Attack on a Pretrained CNN", "Title")
P("1. Problem", "Heading2")
P("Given a pretrained CNN that can only be queried for its predicted label, generate for each of 1000 clean images "
  "an adversarial PNG that is visually almost identical but is classified differently. Evaluation: attack success "
  "rate (50%), imperceptibility (50%), then fewer model inferences and a larger change in confidence as tie-breakers.")
P("2. Method", "Heading2")
P("Because prediction scores are not available, score-based attacks (e.g. Square Attack, NES, SimBA) cannot be used. "
  "We implement an untargeted, L2, hard-label <b>HopSkipJump</b>-style attack, which is decision-based and "
  "query-efficient.")
P("<b>Initialisation.</b> Random uniform-noise images are queried until one receives a label different from the "
  "clean image's label. A binary search on the line between the clean image and that image gives a point just on "
  "the adversarial side of the decision boundary.")
P("<b>Boundary refinement.</b> At each iteration we (a) estimate the gradient direction of the decision boundary at "
  "the current point from label-only queries: random unit directions are sampled in a small ball, each is labelled "
  "adversarial (+1) or not (-1), and the baseline-subtracted average gives the direction; (b) move along it with a "
  "geometric-progression step size (distance / sqrt(t), halved until the point stays adversarial); and (c) binary-search "
  "back toward the clean image. The number of samples per estimate grows as sqrt(t) up to a cap. Each iteration "
  "moves the adversarial point closer to the clean image, which reduces the visible noise.")
P("<b>Quantisation safety.</b> Adversarial images are rounded to 8-bit and the exact saved pixel values are re-queried. "
  "If rounding removed the adversarial effect, the perturbation is scaled up in small steps until the saved PNG is "
  "misclassified. Failed images are retried with a new seed and more iterations.")
P("3. Final parameters", "Heading2")
params = [["Iterations", "10 (+5 per retry)"], ["Gradient-estimate evaluations", "20*sqrt(t+1), capped at 200"],
          ["Gamma", "1.0"], ["Binary-search tolerance", "1e-3"], ["Retries", "3"], ["Seed", "0"]]
def tbl(data):
    t = Table(data, colWidths=[8*cm, 8*cm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                           ("FONTSIZE", (0, 0), (-1, -1), 9)]))
    return t
S.append(tbl(params))
P("4. Results", "Heading2")
S.append(tbl(stats))
P("5. Trade-offs and discussion", "Heading2")
P("Using only labels makes every gradient estimate costly, so the iteration count and per-estimate sample cap are the "
  "main levers between imperceptibility and query count: more iterations shrink the L2 noise but use more queries. "
  "The final point is kept on the adversarial side of the boundary by a minimal margin, which keeps the noise small "
  "while giving a reliably changed prediction. The attack uses only the model's output labels and never accesses "
  "weights, architecture, gradients or confidence scores.")
doc.build(S)
print("report.pdf written")
