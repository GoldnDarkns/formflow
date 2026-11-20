"""Train and test scores side by side for the two tree models and their logistic-regression
comparison, with the same settings as train.py. Exports nothing. Writes ../data/train-vs-test.json.
usage: python scripts/train-vs-test.py
"""
import json, os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, precision_recall_fscore_support

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', '..', 'data')
F = json.load(open(os.path.join(DATA, 'features', 'features.json')))

def arr(k, part):
    r = F[k][part]
    return np.array(r['X'], dtype=np.float32), np.array(r['y'])

out = {}
Xg, yg = arr('training-gt', 'ent'); Xo, yo = arr('training-ocr', 'ent')
Xtr, ytr = np.vstack([Xg, Xo]), np.concatenate([yg, yo])
out['entity'] = {'features': int(Xtr.shape[1]), 'trainExamples': int(len(ytr))}
for name, m in {
    'logreg': make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0, class_weight='balanced')),
    'gbdt': GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.08, subsample=0.9, random_state=0),
}.items():
    m.fit(Xtr, ytr)
    r = {}
    for split in ['training-gt', 'testing-gt', 'training-ocr', 'testing-ocr']:
        X, y = arr(split, 'ent'); p = m.predict(X)
        r[split] = {'n': int(len(y)), 'accuracy': float((p == y).mean()), 'macro_f1': float(f1_score(y, p, average='macro'))}
    out['entity'][name] = r
    print('entity', name, {k: (v['n'], round(v['accuracy'], 3), round(v['macro_f1'], 3)) for k, v in r.items()})

Pg, lg = arr('training-gt', 'pair'); Po, lo = arr('training-ocr', 'pair')
Ptr, ltr = np.vstack([Pg, Po]), np.concatenate([lg, lo])
out['link'] = {'features': int(Ptr.shape[1]), 'trainExamples': int(len(ltr)), 'positiveShare': float(ltr.mean())}
for name, m in {
    'logreg': make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)),
    'gbdt': GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.08, subsample=0.9, random_state=0),
}.items():
    m.fit(Ptr, ltr)
    r = {}
    for split in ['training-gt', 'testing-gt', 'training-ocr', 'testing-ocr']:
        X, y = arr(split, 'pair'); p = (m.predict_proba(X)[:, 1] >= 0.5).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(y, p, average='binary', zero_division=0)
        r[split] = {'n': int(len(y)), 'precision': float(pr), 'recall': float(rc), 'f1': float(f1)}
    out['link'][name] = r
    print('link', name, {k: (v['n'], round(v['precision'], 3), round(v['recall'], 3), round(v['f1'], 3)) for k, v in r.items()})
json.dump(out, open(os.path.join(DATA, 'train-vs-test.json'), 'w'), indent=1)
print(out['entity']['features'], out['entity']['trainExamples'], out['link']['features'], out['link']['trainExamples'], round(out['link']['positiveShare'], 3))
