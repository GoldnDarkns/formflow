import { useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'motion/react';
import { Review } from './Review';
import { DOC_LABEL, DOC_TYPES, type DocType } from '../core/classify';
import { alertsFor } from '../core/facts';
import { union } from '../core/layout';
import type { DocState, EField } from './state';
import type { Template } from '../core/template';
import type { Quad } from '../core/scan';
import { critiqueOf, needsReview, trustOf } from './state';
import { DocViewer } from './DocViewer';
import { Ring, pct, typeColor } from './bits';
import { ICheck, ICopy, IDownload, IPlus, ISpark, IX } from './icons';

interface Props {
  s: DocState;
  selected: string | null;
  onSelect: (uid: string | null) => void;
  onChange: (fields: EField[]) => void;
  onBack: () => void;
  onRetype: (t: DocType) => void;
  onExportJson: () => void;
  onCopy: () => void;
  onExportLabelled: () => void;
  onReport: () => void;
  criticOn: boolean;
  onSecond: () => void;
  secondAvailable: boolean;
  onRemove: () => void;
  onTeach: (t: Template) => void;
  onForgetTemplate: (id: string) => void;
  onCorners: (q: Quad) => void;
}

const confClass = (c: number) => (c >= 0.8 ? 'hi' : c >= 0.6 ? 'mid' : 'lo');

export function ResultView(p: Props) {
  const { s } = p;
  const { doc } = s;
  const listRef = useRef<HTMLDivElement>(null);
  const [reviewing, setReviewing] = useState(false);
  const [copied, setCopied] = useState('');
  const trust = trustOf(s);
  const alerts = alertsFor(doc.docType, doc.classification.confidence, doc.facts, doc.mrz);
  const review = needsReview(s.fields);
  const crit = critiqueOf(s);

  useEffect(() => {
    if (p.selected) listRef.current?.querySelector(`[data-uid="${p.selected}"]`)?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }, [p.selected]);

  const update = (uid: string, patch: Partial<EField>) => p.onChange(s.fields.map((f) => (f.uid === uid ? { ...f, ...patch } : f)));
  const rank = (f: EField) => (f.status === 'review' && !f.confirmed ? 0 : f.status === 'empty' && !f.value ? 2 : 1);
  const sorted = [...s.fields].sort((a, b) => rank(a) - rank(b));
  const cropOf = (f: EField) => {
    const ids = [f.qId, ...s.analysis.links.filter((l) => l.q === f.qId).map((l) => l.a)].filter((x): x is number => x !== undefined);
    const boxes = ids.map((i) => s.analysis.entities[i]?.box).filter(Boolean);
    if (!boxes.length) return null;
    const b = union(boxes), pad = (b[3] - b[1]) * 0.35 + 6;
    return [b[0] - pad, b[1] - pad, b[2] - b[0] + 2 * pad, b[3] - b[1] + 2 * pad];
  };
  const confirmAll = () => p.onChange(s.fields.map((f) => (f.status !== 'review' && f.value ? { ...f, confirmed: true } : f)));

  // ---- what leads the screen: who or what this is, the facts, and whether a person still has to look ----
  const name = doc.facts.find((f) => f.key === 'Full name')?.value;
  const verified = doc.facts.filter((f) => f.proof && f.valid).length;
  const failed = doc.facts.filter((f) => f.proof && f.valid === false).length;
  const problem = alerts.find((a) => a.level === 'bad');
  const state: 'bad' | 'warn' | 'good' = failed || problem ? 'bad' : review ? 'warn' : 'good';
  const headline = failed ? `${failed} check digit${failed === 1 ? '' : 's'} failed: look closely`
    : problem ? problem.text.split('.')[0]
    : review ? `${review} field${review === 1 ? '' : 's'} to check`
    : verified ? `${verified} value${verified === 1 ? '' : 's'} verified by check digits` : 'Read. Nothing flagged.';
  const copyFact = (k: string, v: string) => { navigator.clipboard?.writeText(v).then(() => { setCopied(k); setTimeout(() => setCopied(''), 1200); }).catch(() => {}); };
  const shownFacts = doc.facts.filter((f) => f.key !== 'Full name');

  return (
    <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25 }}>
      <div className="detail-head">
        <button className="btn sm" onClick={p.onBack}>Back</button>
        <span className="grow" />
        <label className="row small" style={{ gap: '0.4rem' }}>
          <span className="muted">Sorted as</span>
          <select className="type" value={doc.docType} onChange={(e) => p.onRetype(e.target.value as DocType)} style={{ color: typeColor(doc.docType) }}>
            {DOC_TYPES.map((t) => <option key={t} value={t}>{DOC_LABEL[t]}</option>)}
          </select>
        </label>
      </div>

      <section className={'answer ' + state} style={{ ['--c' as any]: typeColor(doc.docType) }}>
        <div className="answer-head">
          <div className="grow">
            <span className="answer-type">{DOC_LABEL[doc.docType]}<small>{pct(doc.classification.confidence)} sure</small></span>
            <h1>{name ?? doc.fileName}</h1>
            {name && <p className="answer-file">{doc.fileName}{doc.turned ? ` · turned ${doc.turned}°` : ''}{doc.quad ? ' · cut out of a photo' : ''}</p>}
          </div>
          <Ring value={trust.score} size={72} stroke={7} />
        </div>
        <div className={'answer-state ' + state}>
          <span className="mk">{state === 'good' ? '✓' : '!'}</span>
          <b>{headline}</b>
        </div>
        {shownFacts.length > 0 && (
          <div className="answer-facts">
            {shownFacts.map((f) => (
              <button className="afact" key={f.key} onClick={() => copyFact(f.key, f.value)} title="Tap to copy">
                <span>{f.key}{f.proof && (f.valid ? <i className="ok">verified</i> : <i className="no">check failed</i>)}</span>
                <b>{f.value}</b>
                {copied === f.key && <motion.em initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }}>copied</motion.em>}
              </button>
            ))}
          </div>
        )}
        {alerts.filter((a) => a !== problem).length > 0 && <div className="alerts">{alerts.filter((a) => a !== problem).map((a) => <span key={a.text} className={'tag ' + a.level} style={{ whiteSpace: 'normal' }}>{a.text}</span>)}</div>}
        {doc.template && (
          <p className="answer-note tplnote"><span className="grow"><b>Read with a template:</b> {doc.template.template.name}. Each box is read directly and keeps its proper label.</span>
            {doc.template.template.id.startsWith('user-') && <button className="btn sm" onClick={() => p.onForgetTemplate(doc.template!.template.id)}>Forget this template</button>}</p>
        )}
        <div className="answer-actions">
          {review > 0
            ? <button className="btn primary big" onClick={() => setReviewing(true)}><ICheck /> Check {review} field{review === 1 ? '' : 's'}</button>
            : <button className="btn primary big" onClick={p.onCopy}><ICopy /> Copy everything</button>}
          {review > 0 && <button className="btn" onClick={p.onCopy}><ICopy /> Copy</button>}
          <button className="btn" onClick={p.onExportJson}><IDownload /> JSON</button>
          <button className="btn" onClick={p.onReport}><IDownload /> PDF</button>
        </div>
      </section>

      <div className="results">
        <DocViewer s={s} selected={p.selected} onSelect={p.onSelect} onTeach={p.onTeach} onCorners={p.onCorners} />
        <div className="stack">
          <div className="sheet stack" style={{ gap: '0.7rem' }}>
            <div className="row">
              <h2 className="grow">{review ? `${review} to check` : 'All fields'} <span className="muted small" style={{ fontFamily: 'var(--body)', fontWeight: 500 }}>of {s.fields.filter((f) => f.value).length}</span></h2>
              {s.second === 'none' && <button className="btn sm" disabled={!p.secondAvailable} title={p.secondAvailable ? '' : 'Turn on the second reader on the Scan tab first'} onClick={p.onSecond}><ISpark /> Second read</button>}
              {s.second === 'running' && <span className="tag">Florence-2 is reading</span>}
              {s.second === 'done' && <span className="tag good">Read twice, {((doc.secondMs ?? 0) / 1000).toFixed(0)}s</span>}
              {s.second === 'error' && <span className="tag bad" title={s.secondError}>Second read failed</span>}
              <button className="btn sm" onClick={confirmAll}><ICheck /> Confirm the sure ones</button>
            </div>
            <div className="fields" ref={listRef}>
              {sorted.length === 0 && <p className="muted small">No label and value pairs found. Switch the viewer to "Before / after" to judge the scan, or add fields by hand.</p>}
              {sorted.map((f) => {
                const sel = p.selected === f.uid;
                const crop = sel ? cropOf(f) : null;
                return (
                  <motion.div layout="position" key={f.uid} data-uid={f.uid} className={'field' + (f.status === 'review' && !f.confirmed ? ' review' : '') + (f.status === 'empty' && !f.value ? ' empty' : '') + (sel ? ' sel' : '')} onFocus={() => p.onSelect(f.uid)} onClick={() => p.onSelect(f.uid)}>
                    <input className="key" aria-label="Field name" value={f.key} onChange={(e) => update(f.uid, { key: e.target.value, edited: true })} />
                    <textarea className="val" aria-label={`Value for ${f.key}`} rows={Math.min(4, Math.max(1, Math.ceil(f.value.length / 30)))} value={f.value} placeholder="empty" onChange={(e) => update(f.uid, { value: e.target.value, edited: true })} />
                    <div className="side">
                      {f.confirmed
                        ? <motion.span className="conf hi" initial={{ scale: 0.6 }} animate={{ scale: 1 }} transition={{ type: 'spring', stiffness: 500, damping: 18 }}>checked</motion.span>
                        : <span className={'conf ' + confClass(f.confidence)} title="How sure FormFlow is about this field">{pct(f.confidence)}</span>}
                      <button className="btn icon ghost" title="This is correct" aria-label={`Confirm ${f.key}`} onClick={(e) => { e.stopPropagation(); update(f.uid, { confirmed: !f.confirmed, status: !f.confirmed ? 'ok' : f.status }); }}><ICheck /></button>
                      <button className="btn icon ghost" title="Delete this field" aria-label={`Delete ${f.key}`} onClick={(e) => { e.stopPropagation(); p.onChange(s.fields.filter((x) => x.uid !== f.uid)); }}><IX /></button>
                    </div>
                    {crop && (
                      <div className="crop"><svg viewBox={crop.join(' ')} preserveAspectRatio="xMidYMid meet"><image href={doc.view} width={doc.width} height={doc.height} /></svg></div>
                    )}
                    {!f.confirmed && (f.reasons.length > 0 || f.alt || f.mrzMatch) && (
                      <div className="extra">
                        {f.mrzMatch && <span className="tag good">Same as the machine-readable zone</span>}
                        {f.reasons.map((r) => <span key={r} className="tag warn">{r}</span>)}
                        {f.alt && <span className="alt">Second reader saw <q>{f.alt}</q><button className="btn sm" onClick={(e) => { e.stopPropagation(); update(f.uid, { value: f.alt!, edited: true }); }}>Use this</button></span>}
                      </div>
                    )}
                  </motion.div>
                );
              })}
            </div>
            <div className="row">
              <button className="btn sm" onClick={() => p.onChange([...s.fields, { uid: 'm' + Date.now(), key: 'New field', value: '', confidence: 1, status: 'ok', reasons: [], source: 'layout', edited: true }])}><IPlus /> Add a field</button>
            </div>
          </div>

          {/* the reasoning is one tap away, not in the way */}
          <details className="sheet fold">
            <summary><span className="grow">Why "{DOC_LABEL[doc.docType]}"?</span><span className={'verdict ' + crit.verdict}>{crit.verdict === 'holds' ? 'Second opinion agrees' : crit.verdict === 'doubt' ? 'Second opinion has doubts' : 'Second opinion disagrees'}</span></summary>
            <div className="stack" style={{ gap: '0.8rem', marginTop: '0.8rem' }}>
              <p className="small">{crit.summary}</p>
              <div className="duel">
                <div><span>Sorter, reading the text</span><b style={{ color: typeColor(doc.docType) }}>{DOC_LABEL[doc.docType]}</b> <span>{pct(doc.classification.confidence)} sure</span></div>
                <div>
                  <span>Critic, looking at the picture</span>
                  {crit.vision ? <><b style={{ color: typeColor(crit.vision.type) }}>{DOC_LABEL[crit.vision.type]}</b> <span>{pct(crit.vision.confidence)} sure</span></> : <><b className="muted">{p.criticOn ? 'Looking' : 'Switched off'}</b><span>{p.criticOn ? 'The vision model is loading or working' : 'Turn the critic on from the Scan tab'}</span></>}
                </div>
              </div>
              <ul className="checks">
                {crit.expected.map((e) => <li key={e.text} className={e.ok ? 'ok' : 'no'}><span className="mk">{e.ok ? '✓' : '!'}</span><span>{e.text}</span></li>)}
              </ul>
              <p className="small muted">For {DOC_LABEL[doc.docType].toLowerCase()}</p>
              <ul className="why">{doc.classification.evidence.map((e) => <li key={e.why}>{e.why}</li>)}</ul>
              {crit.against.length > 0 && <><p className="small muted">Against{crit.runnerUp ? `, ${pct(crit.runnerUp.prob)} chance it is a ${DOC_LABEL[crit.runnerUp.type].toLowerCase()}` : ''}</p><ul className="why">{crit.against.map((e) => <li key={e}>{e}</li>)}</ul></>}
              <div className="trust">
                <div className="parts">
                  {trust.parts.map((x) => (
                    <div className="part" key={x.label} title={x.note}>
                      <span>{x.label}</span>
                      <div className="bar"><div style={{ width: pct(x.value) }} /></div>
                      <span className="num">{pct(x.value)}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </details>
          {doc.mrz && (
            <details className="sheet fold">
              <summary><span className="grow">Code lines at the bottom of the page</span><span className={'verdict ' + (doc.mrz.valid ? 'holds' : 'doubt')}>{doc.mrz.valid ? 'Every check digit passes' : 'Not every check digit passes'}</span></summary>
              <div className="mrzlines" style={{ marginTop: '0.8rem' }}>{doc.mrz.lines.join('\n')}</div>
              <ul className="checks" style={{ marginTop: '0.6rem' }}>{doc.mrz.checks.map((c) => <li key={c.name} className={c.ok ? 'ok' : 'no'}><span className="mk">{c.ok ? '✓' : '!'}</span><span>{c.name}</span></li>)}</ul>
            </details>
          )}

          <div className="row">
            <button className="btn ghost sm" title="Saves your corrections as a labelled example, for testing or training the models" onClick={p.onExportLabelled}>Save as labelled example</button>
            <span className="grow" />
            <button className="btn ghost sm" style={{ color: 'var(--bad)' }} onClick={p.onRemove}>Remove this document</button>
          </div>
        </div>
      </div>
      <AnimatePresence>{reviewing && <Review s={s} cropOf={cropOf} onChange={p.onChange} onClose={() => setReviewing(false)} />}</AnimatePresence>
    </motion.div>
  );
}
