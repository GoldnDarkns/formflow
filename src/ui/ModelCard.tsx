import { useEffect, useState } from 'react';

const f = (x?: number, d = 1) => (x === undefined || Number.isNaN(x) ? '—' : (x * 100).toFixed(d) + '%');
const ms = (x?: number) => (x === undefined ? '—' : x > 1000 ? (x / 1000).toFixed(1) + ' s' : Math.round(x) + ' ms');

function Bar({ v }: { v: number }) {
  return <div className="bar"><div style={{ width: `${Math.max(0, Math.min(1, v)) * 100}%` }} /></div>;
}

export function ModelCard() {
  const [r, setR] = useState<any>(null);
  const [err, setErr] = useState('');
  useEffect(() => {
    fetch(import.meta.env.BASE_URL + 'eval/results.json').then((x) => x.json()).then(setR).catch((e) => setErr(String(e)));
  }, []);
  if (err) return <div className="card">Could not load evaluation results: {err}</div>;
  if (!r) return <div className="card">Loading evaluation results…</div>;

  const ocrNames: Record<string, string> = {
    tesseract_raw: 'Tesseract, no pre-processing',
    tesseract_upscale: 'Tesseract + upscaling',
    tesseract_full: 'Tesseract + full clean-up (classical baseline)',
    ppocr_v5_mobile: 'PP-OCRv5 mobile, neural reader',
    florence2_1x1: 'Florence-2, whole page',
    florence2_1x2: 'Florence-2, two tiles (app setting)',
  };
  const lk = r.linking_gt_entities;
  const e2e = r.end_to_end;
  const flo = (Object.entries(e2e).find(([k]) => k.endsWith('florence_1x2')) ?? Object.entries(e2e).find(([k]) => k.startsWith('gradient_boosting_with_florence'))) as [string, any] | undefined;

  return (
    <div className="prose" style={{ maxWidth: 980, margin: '0 auto' }}>
      <div className="card">
        <h1>Model card</h1>
        <p>All numbers below come from the evaluation script in the repository, run on the <b>{r.dataset}</b>, which none of the models saw during training. Generated {new Date(r.generated).toLocaleString()}.</p>
        <div className="summary">
          <div className="stat"><b>{f(lk.gradient_boosting.gold_labels.f1)}</b><span>question→answer link F1 (gold entities)</span></div>
          <div className="stat"><b>{f(lk.gradient_boosting.predicted_labels.labelAccuracy)}</b><span>entity label accuracy</span></div>
          <div className="stat"><b>{f((r.ocr.ppocr_v5_mobile ?? r.ocr.tesseract_full).cer)}</b><span>character error rate, PP-OCRv5 reader</span></div>
          {flo?.[1].wordAgreement && <div className="stat"><b>{f(flo[1].wordAgreement.agree.accuracy)}</b><span>word accuracy when both readers agree</span></div>}
        </div>
      </div>

      <div className="card">
        <h2>Intended use</h2>
        <p>Helping a back-office clerk digitise scanned paper forms and identity documents (passports, ID cards, visas, licences) into structured key–value data, with a person reviewing every flagged field before it is used. <b>Out of scope:</b> automatic decisions about people without human review, identity verification or fraud detection (the app does not check that a document is genuine), handwriting-heavy forms, and non-Latin scripts.</p>
        <h2>Models</h2>
        <ul>
          <li><b>Readers:</b> PP-OCRv5 mobile (text detector 4.6 MB + English recogniser 7.5 MB, ONNX, Apache-2.0) and Tesseract 5 LSTM (tesseract.js, Apache-2.0) as the classical baseline and for passport code lines.</li>
          <li><b>Entity labeller</b> (question / answer / header / other) and <b>link classifier</b> (question → answer): gradient-boosted trees on 82 (labeller) and 21 (linker) geometry + text features, trained on the FUNSD training split (ground-truth blocks plus blocks from both readers) and compiled to plain ONNX operators (1.9 MB in total), run with ONNX Runtime Web.</li>
          <li><b>Second reader:</b> Florence-2-base-ft (Microsoft, MIT licence, 0.23 B parameters), int8 / q4 ONNX via Transformers.js, WebGPU when available.</li>
          <li><b>MRZ:</b> rule-based ICAO 9303 parser; check digits repair common OCR confusions (O/0, I/1, S/5, B/8).</li>
        </ul>
      </div>

      <div className="card">
        <h2>Reading the text</h2>
        <div className="tablewrap"><table className="metrics">
          <thead><tr><th>Reader</th><th className="num">CER ↓</th><th className="num">Word acc. ↑</th><th className="num">Omitted ↓</th><th className="num">Hallucinated ↓</th><th className="num">Time / page</th></tr></thead>
          <tbody>
            {Object.entries(r.ocr).map(([k, v]: [string, any]) => (
              <tr key={k} className={k === 'ppocr_v5_mobile' ? 'best' : ''}>
                <td>{ocrNames[k] ?? k} <span className="muted tiny">({v.pages} pages)</span></td>
                <td className="num">{f(v.cer)}</td><td className="num">{f(v.wordAccuracy)}</td><td className="num">{f(v.omissionRate)}</td><td className="num">{f(v.hallucinationRate)}</td><td className="num">{ms(v.msPerPage)}</td>
              </tr>
            ))}
          </tbody>
        </table></div>
        <p className="small muted">Omitted = ground-truth words with no prediction over them. Hallucinated = predicted words where the form has no text. Florence-2 times are CPU-only (Node, 2 cores); WebGPU in the browser is much faster.</p>
      </div>

      <div className="card">
        <h2>Linking labels to values</h2>
        <p className="small">Standard FUNSD protocol: ground-truth text blocks; “gold labels” also gives the question/answer labels, “predicted labels” uses our labeller first.</p>
        <div className="tablewrap"><table className="metrics">
          <thead><tr><th>Method</th><th className="num">F1 (gold labels)</th><th className="num">F1 (predicted labels)</th><th className="num">Label accuracy</th></tr></thead>
          <tbody>
            {[['baseline_rules', 'Baseline: geometry + colon rules'], ['logistic_regression', 'Logistic regression'], ['gradient_boosting', 'Gradient boosting (deployed)']].map(([k, n]) => (
              <tr key={k} className={k === 'gradient_boosting' ? 'best' : ''}>
                <td>{n}</td><td className="num">{f(lk[k].gold_labels.f1)}</td><td className="num">{f(lk[k].predicted_labels.f1)}</td><td className="num">{f(lk[k].predicted_labels.labelAccuracy)}</td>
              </tr>
            ))}
          </tbody>
        </table></div>
      </div>

      <div className="card">
        <h2>From image to finished fields</h2>
        <p className="small">A field counts as correct only if both the key and the value match the ground truth with at least 80% character similarity, after OCR, grouping, labelling and linking — the strictest measure.</p>
        <div className="tablewrap"><table className="metrics">
          <thead><tr><th>Pipeline</th><th className="num">Precision</th><th className="num">Recall</th><th className="num">F1</th><th className="num">Auto-accepted correct</th><th className="num">Flagged correct</th></tr></thead>
          <tbody>
            {Object.entries(e2e).filter(([, v]: any) => typeof v === 'object').map(([k, v]: [string, any]) => (
              <tr key={k}>
                <td>{k.replace(/_/g, ' ')} <span className="muted tiny">({v.pages} pages)</span></td>
                <td className="num">{f(v.precision)}</td><td className="num">{f(v.recall)}</td><td className="num">{f(v.f1)}</td>
                <td className="num">{f(v.flagged.autoAcceptedCorrect)} <span className="muted tiny">n={v.flagged.autoAccepted}</span></td>
                <td className="num">{f(v.flagged.flaggedCorrect)} <span className="muted tiny">n={v.flagged.flaggedForReview}</span></td>
              </tr>
            ))}
          </tbody>
        </table></div>
        <p className="small muted">The review flag works as intended when auto-accepted fields are much more often correct than flagged ones: flagged fields are where the clerk's time is spent.</p>
      </div>

      {flo?.[1].wordAgreement && (
        <div className="card">
          <h2>Generative AI experiment: does a second reader help?</h2>
          <p className="small">For every Tesseract word we check whether Florence-2 read the same text in the same place, then measure how often the Tesseract word is actually right.</p>
          <div className="tablewrap"><table className="metrics">
            <thead><tr><th>Readers…</th><th className="num">Words</th><th className="num">Tesseract word correct</th><th></th></tr></thead>
            <tbody>
              {[['agree', 'agree'], ['disagree', 'disagree'], ['secondReaderSawNothing', 'Florence-2 saw nothing there']].map(([k, n]) => (
                <tr key={k}><td>{n}</td><td className="num">{flo[1].wordAgreement[k].words}</td><td className="num">{f(flo[1].wordAgreement[k].accuracy)}</td><td style={{ width: '30%' }}><Bar v={flo[1].wordAgreement[k].accuracy} /></td></tr>
              ))}
            </tbody>
          </table></div>
          <p className="small">Agreement is used as a confidence signal: fields where the two readers disagree are flagged, and Florence-2's reading is offered as a one-tap alternative. Florence-2 alone is a weaker OCR than Tesseract on these faxes (it misses more words), so it is used as a checker, not a replacement.</p>
        </div>
      )}

      {r.doc_type && (
        <div className="card">
          <h2>Recognising the document type</h2>
          <p className="small">Each page is sorted as passport, Emirates ID, ID card, driving licence, visa, form or other, from the machine-readable zone, tell-tale text and page layout. The confidence is a softmax over the evidence weights.</p>
          <div className="summary">
            <div className="stat"><b>{f(r.doc_type.funsd_forms.accuracy, 0)}</b><span>of {r.doc_type.funsd_forms.pages} FUNSD forms sorted as "form"</span></div>
            <div className="stat"><b>{f(r.doc_type.funsd_forms.meanConfidence, 0)}</b><span>mean confidence on those forms</span></div>
            {r.doc_type.specimens && <div className="stat"><b>{r.doc_type.specimens.correct}/{r.doc_type.specimens.total}</b><span>synthetic specimen IDs sorted correctly</span></div>}
          </div>
          <p className="small muted">Identity documents are so far tested only on a handful of synthetic specimens. Real passports, IDs, licences and visas need to be added to the test set before this can be trusted on them.</p>
        </div>
      )}

      {r.magic_scan && (
        <div className="card">
          <h2>Magic scan: finding and flattening the page</h2>
          <p className="small">{r.magic_scan.photos} test forms were re-photographed synthetically: tilted in perspective on a dark, cluttered desk with a diagonal shadow. The page finder is classical computer vision (threshold, flood fill from the border, largest region, corner extremes, homography). The score is the share of ground-truth words the reader then gets right.</p>
          <div className="summary">
            <div className="stat"><b>{f(r.magic_scan.pageFound, 0)}</b><span>pages found</span></div>
            <div className="stat"><b>{f(r.magic_scan.meanCornerError)}</b><span>mean corner error, of the image diagonal</span></div>
            <div className="stat"><b>{f(r.magic_scan.wordRecall.photoAsTaken)}</b><span>words read from the photo as taken</span></div>
            <div className="stat"><b>{f(r.magic_scan.wordRecall.pageCutOut)}</b><span>after cutting out the page</span></div>
            <div className="stat"><b>{f(r.magic_scan.wordRecall.cutOutAndLightingFixed)}</b><span>after also fixing the lighting</span></div>
          </div>
          <p className="small muted">These are simulated photos with a page lighter than the desk. Real photos on a white table, or with glare, will be harder, and the page finder then leaves the image as it is.</p>
        </div>
      )}

      {r.critic_vision && (
        <div className="card">
          <h2>The critic's second opinion</h2>
          <p className="small">MobileCLIP-S0's image encoder (46 MB, full precision) compares the picture with a description of each document type. It never sees the OCR text, so it is independent of the sorter. On its own it is weaker than the sorter, which is why it only raises doubts and never overrules.</p>
          <div className="summary">
            <div className="stat"><b>{r.critic_vision.funsd_forms.correct}/{r.critic_vision.funsd_forms.total}</b><span>FUNSD forms called "form"</span></div>
            <div className="stat"><b>{r.critic_vision.specimens.correct}/{r.critic_vision.specimens.total}</b><span>sample scans right</span></div>
            <div className="stat"><b>{Math.round(r.critic_vision.msPerImage)} ms</b><span>per image on CPU</span></div>
          </div>
          <p className="small muted">Two negative results worth reporting: the 8-bit version of this model lost its zero-shot ability entirely, and Florence-2's captions ("a screen with some text") were too generic to name a document type.</p>
        </div>
      )}

      {e2e.gradient_boosting?.reliability && (
        <div className="card">
          <h2>Is the confidence figure honest?</h2>
          <p className="small">Fields are grouped by the confidence the app showed, and compared with how often they were actually right (key and value both correct) on the FUNSD test forms.</p>
          <div className="tablewrap"><table className="metrics">
            <thead><tr><th>Confidence shown</th><th className="num">Fields</th><th className="num">Actually correct</th><th></th></tr></thead>
            <tbody>
              {e2e.gradient_boosting.reliability.map((b: any) => (
                <tr key={b.range}><td>{b.range}</td><td className="num">{b.fields}</td><td className="num">{f(b.correct)}</td><td style={{ width: '30%' }}><Bar v={b.correct} /></td></tr>
              ))}
            </tbody>
          </table></div>
          <p className="small">The figure ranks fields well: higher confidence is right far more often. On these old faxes it is too optimistic in absolute terms, so read it as "check the low ones first", not as a guarantee. Fields confirmed by a check digit are the exception: those are verified by arithmetic.</p>
        </div>
      )}

      <div className="card">
        <h2>Responsible AI</h2>
        <ul>
          <li><b>Privacy:</b> processing happens in the browser; images and results never leave the device. No accounts, analytics or cookies. Export is an explicit user action.</li>
          <li><b>Human in the loop:</b> low-confidence fields, reader disagreements and failed check digits are flagged and listed first; nothing is accepted on the clerk's behalf.</li>
          <li><b>Bias and subgroups:</b> Tesseract is trained mainly on Latin-script print. Names in other scripts, Arabic text on UAE documents, and handwriting are read less reliably — those fields are expected to be flagged more often and must be checked.</li>
          <li><b>Data licence:</b> FUNSD is for non-commercial research and education; this app and its models are not for commercial use.</li>
          <li><b>Accessibility:</b> every field is a labelled, keyboard-editable input; colour is never the only signal (status tags carry text).</li>
        </ul>
      </div>
    </div>
  );
}
