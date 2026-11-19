import { useEffect, useRef } from 'react';
import { motion } from 'motion/react';
import { DOC_LABEL, DOC_TYPES, type DocType } from '../core/classify';
import { alertsFor } from '../core/facts';
import { union } from '../core/layout';
import type { DocState, EField } from './state';
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
}

const confClass = (c: number) => (c >= 0.8 ? 'hi' : c >= 0.6 ? 'mid' : 'lo');

export function ResultView(p: Props) {
  const { s } = p;
  const { doc } = s;
  const listRef = useRef<HTMLDivElement>(null);
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

  return (
    <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25 }}>
      <div className="detail-head">
        <button className="btn sm" onClick={p.onBack}>Back to trays</button>
        <h1 className="grow">{doc.facts.find((f) => f.key === 'Full name')?.value ?? doc.fileName}</h1>
        <label className="row small" style={{ gap: '0.4rem' }}>
          <span className="muted">Sorted as</span>
          <select className="type" value={doc.docType} onChange={(e) => p.onRetype(e.target.value as DocType)} style={{ color: typeColor(doc.docType) }}>
            {DOC_TYPES.map((t) => <option key={t} value={t}>{DOC_LABEL[t]}</option>)}
          </select>
        </label>
      </div>

      <div className="results">
        <DocViewer s={s} selected={p.selected} onSelect={p.onSelect} />
        <div className="stack">
          <div className="sheet stack">
            <div className="trust">
              <Ring value={trust.score} size={104} stroke={10} />
              <div className="parts">
                <h2>How sure is FormFlow?</h2>
                {trust.parts.map((x) => (
                  <div className="part" key={x.label} title={x.note}>
                    <span>{x.label}</span>
                    <div className="bar"><motion.div initial={{ width: 0 }} animate={{ width: pct(x.value) }} transition={{ duration: 0.8, ease: [0.2, 0.8, 0.2, 1] }} /></div>
                    <span className="num">{pct(x.value)}</span>
                  </div>
                ))}
              </div>
            </div>
            {alerts.length > 0 && <div className="alerts">{alerts.map((a) => <span key={a.text} className={'tag ' + a.level} style={{ whiteSpace: 'normal' }}>{a.text}</span>)}</div>}
          </div>

          <div className="sheet stack" style={{ gap: '0.8rem' }}>
            <div className="row">
              <h2 className="grow">The critic</h2>
              <motion.span key={crit.verdict} className={'verdict ' + crit.verdict} initial={{ scale: 0.7, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ type: 'spring', stiffness: 420, damping: 20 }}>
                {crit.verdict === 'holds' ? 'Agrees' : crit.verdict === 'doubt' ? 'Has doubts' : 'Disagrees'}
              </motion.span>
            </div>
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
            <details>
              <summary className="small" style={{ cursor: 'pointer', fontWeight: 600 }}>The case for and against</summary>
              <p className="small muted" style={{ marginTop: '0.5rem' }}>For {DOC_LABEL[doc.docType].toLowerCase()}</p>
              <ul className="why">{doc.classification.evidence.map((e) => <li key={e.why}>{e.why}</li>)}</ul>
              {crit.against.length > 0 && <><p className="small muted" style={{ marginTop: '0.5rem' }}>Against{crit.runnerUp ? `, ${pct(crit.runnerUp.prob)} chance it is a ${DOC_LABEL[crit.runnerUp.type].toLowerCase()}` : ''}</p><ul className="why">{crit.against.map((e) => <li key={e}>{e}</li>)}</ul></>}
              {!crit.against.length && <p className="small muted" style={{ marginTop: '0.5rem' }}>Nothing on the page argues for another type.</p>}
            </details>
          </div>

          {doc.facts.length > 0 && (
            <div className="sheet stack">
              <h2>Key facts</h2>
              <div className="facts">
                {doc.facts.map((f) => (
                  <div className="fact" key={f.key}>
                    <span>{f.key}</span>
                    <b>{f.value}</b>
                    {f.proof && (f.valid ? <span className="tag good">{f.proof} ✓</span> : <span className="tag bad">{f.proof} fails</span>)}
                  </div>
                ))}
              </div>
              {doc.mrz && <div className="mrzlines">{doc.mrz.lines.join('\n')}</div>}
            </div>
          )}

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
              {sorted.length === 0 && <p className="muted small">No label and value pairs found. Switch the viewer to "What the reader saw" to judge the scan, or add fields by hand.</p>}
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

          <div className="row">
            <button className="btn primary" onClick={p.onExportJson}><IDownload /> Download JSON</button>
            <button className="btn" onClick={p.onReport}><IDownload /> PDF report</button>
            <button className="btn" onClick={p.onCopy}><ICopy /> Copy</button>
            <button className="btn ghost sm" title="Saves your corrections as a labelled example, for testing or training the models" onClick={p.onExportLabelled}>Save as labelled example</button>
            <span className="grow" />
            <button className="btn ghost sm" style={{ color: 'var(--bad)' }} onClick={p.onRemove}>Remove</button>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
