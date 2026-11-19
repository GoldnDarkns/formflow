"""Train the entity-label and question->answer link classifiers and export ONNX.

Trains on FUNSD training pages using BOTH ground-truth entities and the
entities our own OCR pipeline produces (so the model sees realistic,
imperfect segments). Evaluates on the FUNSD test pages in both settings.
usage: python3 scripts/train.py
"""
import json, os, sys, time
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, classification_report, confusion_matrix, precision_recall_fscore_support
from skl2onnx import to_onnx
import onnxruntime as ort
import joblib
sys.path.insert(0, HERE) if 'HERE' in dir() else None

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', '..', 'data')
OUT = os.path.join(HERE, '..', 'public', 'models')
os.makedirs(OUT, exist_ok=True)
F = json.load(open(os.path.join(DATA, 'features', 'features.json')))
LABELS = ['question', 'answer', 'header', 'other']

def arr(k, part):
    r = F[k][part]
    return np.array(r['X'], dtype=np.float32), np.array(r['y']), np.array(r['g'])

report = {}

# ---------------- entity labelling ----------------
Xg, yg, _ = arr('training-gt', 'ent')
Xo, yo, _ = arr('training-ocr', 'ent')
Xtr, ytr = np.vstack([Xg, Xo]), np.concatenate([yg, yo])
models = {
    'logreg': make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0, class_weight='balanced')),
    'gbdt': GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.08, subsample=0.9, random_state=0),
}
ent = {}
for name, m in models.items():
    t = time.time(); m.fit(Xtr, ytr)
    ent[name] = {}
    for split in ['testing-gt', 'testing-ocr']:
        X, y, _ = arr(split, 'ent')
        p = m.predict(X)
        ent[name][split] = {
            'macro_f1': float(f1_score(y, p, average='macro')),
            'per_class_f1': dict(zip(LABELS, map(float, f1_score(y, p, average=None, labels=range(4))))),
            'accuracy': float((p == y).mean()),
            'confusion': confusion_matrix(y, p, labels=range(4)).tolist(),
        }
    print(f'entity {name}: {time.time()-t:.0f}s', {k: round(v['macro_f1'], 3) for k, v in ent[name].items()})
report['entity'] = ent
best_ent = models['gbdt']
X, y, _ = arr('testing-gt', 'ent')
print(classification_report(y, best_ent.predict(X), target_names=LABELS, digits=3))

# ---------------- linking ----------------
Pg, lg, _ = arr('training-gt', 'pair')
Po, lo, _ = arr('training-ocr', 'pair')
Ptr, ltr = np.vstack([Pg, Po]), np.concatenate([lg, lo])
lmodels = {
    'logreg': make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)),
    'gbdt': GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.08, subsample=0.9, random_state=0),
}

def link_f1(split, probs, thr=0.5):
    """FUNSD-style link F1: each answer keeps its best question above threshold."""
    r = F[split]['pair']
    X, y, g = np.array(r['X']), np.array(r['y']), np.array(r['g'])
    # candidates were generated per page in the same order as features; we
    # reconstruct answer identity from (page, rank features) is not possible,
    # so use the pair-level P/R/F1 at the threshold + best-per-answer decoding
    # done in the TypeScript evaluator. Here: pair-level metrics.
    pred = (probs >= thr).astype(int)
    p, rr, f, _ = precision_recall_fscore_support(y, pred, average='binary')
    return {'precision': float(p), 'recall': float(rr), 'f1': float(f)}

lnk = {}
for name, m in lmodels.items():
    t = time.time(); m.fit(Ptr, ltr)
    lnk[name] = {}
    for split in ['testing-gt', 'testing-ocr']:
        X, y, _ = arr(split, 'pair')
        lnk[name][split] = link_f1(split, m.predict_proba(X)[:, 1])
    print(f'link {name}: {time.time()-t:.0f}s', {k: round(v['f1'], 3) for k, v in lnk[name].items()})
report['link_pairlevel'] = lnk
best_link = lmodels['gbdt']

# ---------------- export ONNX ----------------
def export(model, X, path):
    onx = to_onnx(model, X[:1], options={id(model): {'zipmap': False}}, target_opset=15)
    with open(path, 'wb') as f:
        f.write(onx.SerializeToString())
    s = ort.InferenceSession(path)
    out = s.run(None, {s.get_inputs()[0].name: X[:50]})
    ref = model.predict_proba(X[:50])
    assert np.allclose(out[1], ref, atol=1e-4), 'onnx mismatch'
    print('exported', path, os.path.getsize(path) // 1024, 'KB', [i.name for i in s.get_inputs()], [o.name for o in s.get_outputs()])

sys.path.insert(0, HERE)
from tree2onnx import compile_gbdt

def export_trees(model, X, path):
    T, I, L = compile_gbdt(model, X.shape[1], path)
    s = ort.InferenceSession(path)
    out = s.run(None, {'X': X[:2000]})
    ref = model.predict_proba(X[:2000])
    err = np.abs(out[1] - ref).max()
    agree = (out[0] == ref.argmax(1)).mean()
    print('exported', path, os.path.getsize(path) // 1024, 'KB', f'trees={T} max|dp|={err:.2e} argmax agreement={agree:.4f}')
    assert agree > 0.995

export_trees(best_ent, Xtr, os.path.join(OUT, 'entity_gbdt.onnx'))
export_trees(best_link, Ptr, os.path.join(OUT, 'link_gbdt.onnx'))
joblib.dump({'entity': best_ent, 'link': best_link}, os.path.join(DATA, 'features', 'gbdt.joblib'))
# also export the logistic-regression versions for the ablation in the app eval
export(models['logreg'], Xtr, os.path.join(DATA, 'features', 'entity_logreg.onnx'))
export(lmodels['logreg'], Ptr, os.path.join(DATA, 'features', 'link_logreg.onnx'))
json.dump(report, open(os.path.join(DATA, 'features', 'train_report.json'), 'w'), indent=1)
