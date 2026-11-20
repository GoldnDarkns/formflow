import { AnimatePresence, motion } from 'motion/react';
import { DOC_LABEL, DOC_TYPES, type DocType } from '../core/classify';
import { alertsFor } from '../core/facts';
import { critiqueOf, needsReview, packetsOf, sidesOf, trustOf, type DocState } from './state';
import { Ring, Stamp, typeColor } from './bits';
import { ICopy, IDownload } from './icons';

interface Props {
  docs: DocState[];
  onOpen: (id: string) => void;
  onExportCsv: () => void;
  onReport: () => void;
  onZip: () => void;
  onTypePdf: (t: DocType) => void;
  onScan: () => void;
  onCopyText: (text: string, what: string) => void;
}

export function Sorted({ docs, onOpen, onExportCsv, onReport, onScan, onZip, onTypePdf, onCopyText }: Props) {
  if (!docs.length)
    return (
      <div className="empty-state">
        <h2>The trays are empty</h2>
        <p>Scan a pile of documents and they will be sorted here by type.</p>
        <button className="btn primary" onClick={onScan}>Scan documents</button>
      </div>
    );
  const types = DOC_TYPES.filter((t) => docs.some((d) => d.doc.docType === t));
  const review = docs.reduce((a, d) => a + needsReview(d.fields), 0);
  const packets = packetsOf(docs);
  const sides = sidesOf(docs);
  const plural = (n: number, w: string) => `${n} ${w}${n === 1 ? '' : 's'}`;

  return (
    <div>
      <div className="overview">
        <div>
          <h1>{plural(docs.length, 'document')}, <span>{plural(types.length, 'type')}</span></h1>
          <p className="lede" style={{ marginTop: '0.5rem' }}>{review ? `${plural(review, 'field')} still to check.` : 'Nothing left to check.'} Open a document to review it.</p>
        </div>
        <div className="row">
          <button className="btn primary" onClick={onZip} title="A folder per document type, each with its scans as one PDF and the data read from them"><IDownload /> Everything as ZIP</button>
          <button className="btn dark" onClick={onReport}><IDownload /> PDF report</button>
          <button className="btn" onClick={onExportCsv}><IDownload /> CSV</button>
        </div>
      </div>

      {packets.map((p) => (
        <motion.div key={p.name} className="packet" style={{ marginTop: '1rem' }} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <div className="row">
            <h2 className="grow">{p.name}</h2>
            <span className="tag">{p.docs.map((d) => DOC_LABEL[d.doc.docType]).join(', ')}</span>
          </div>
          <p className="small muted">These {p.docs.length} documents carry the same name, so their details are put together and compared.</p>
          <div className="profile">
            {p.profile.map((f) => (
              <button key={f.key} className="afact" title={`From the ${f.from.toLowerCase()}. Tap to copy`} onClick={() => onCopyText(f.value, f.key)}>
                <span>{f.key}{f.verified && <i className="ok">verified</i>}</span>
                <b>{f.value}</b>
              </button>
            ))}
          </div>
          <div className="row" style={{ gap: '0.4rem' }}>
            {p.soonest && <span className={'tag ' + (p.soonest.days < 0 ? 'bad' : p.soonest.days < 183 ? 'warn' : 'good')}>{p.soonest.days < 0 ? `${p.soonest.label} expired on ${p.soonest.date}` : `First to expire: ${p.soonest.label}, ${p.soonest.date}`}</span>}
            {p.missing.length > 0
              ? <span className="tag warn">Not in this pile: {p.missing.map((t) => DOC_LABEL[t]).join(', ')}</span>
              : <span className="tag good">Passport, Emirates ID and visa are all here</span>}
            <span className="grow" />
            <button className="btn sm" onClick={() => onCopyText(p.profile.map((f) => `${f.key}: ${f.value}`).join('\n'), 'Details')}><ICopy /> Copy details</button>
          </div>
          <ul className="checks">
            {p.checks.map((c) => (
              <li key={c.label} className={c.ok ? 'ok' : 'no'}>
                <span className="mk">{c.ok ? '✓' : '!'}</span>
                <span><b>{c.label} {c.ok ? 'matches' : 'does not match'}.</b> {c.detail}</span>
              </li>
            ))}
          </ul>
        </motion.div>
      ))}

      {types.map((t) => {
        const list = docs.filter((d) => d.doc.docType === t);
        return (
          <section key={t} style={{ ['--c' as any]: typeColor(t) }}>
            <div className="trayhead"><h2>{DOC_LABEL[t]}</h2><span className="muted small">{list.length}</span><button className="tabdl" onClick={() => onTypePdf(t)} title={`Download the ${list.length} ${DOC_LABEL[t]} scan${list.length === 1 ? '' : 's'} as one PDF`} aria-label={`Download ${DOC_LABEL[t]} scans as one PDF`}><IDownload /> PDF</button></div>
            <div className="grid">
              <AnimatePresence mode="popLayout">
                {list.map((d) => {
                  const trust = trustOf(d);
                  const alerts = alertsFor(d.doc.docType, d.doc.classification.confidence, d.doc.facts, d.doc.mrz, undefined, d.doc.quality).filter((a) => a.level !== 'good');
                  const n = needsReview(d.fields);
                  const who = d.doc.facts.find((f) => f.key === 'Full name')?.value;
                  return (
                    <motion.button key={d.doc.id} className="doc" layout initial={{ opacity: 0, y: -40, rotate: -6, scale: 0.9 }} animate={{ opacity: 1, y: 0, rotate: 0, scale: 1 }} exit={{ opacity: 0, scale: 0.9 }} transition={{ type: 'spring', stiffness: 260, damping: 22, delay: 0.05 * docs.indexOf(d) }} onClick={() => onOpen(d.doc.id)}>
                      <div className="thumb"><img src={d.doc.view} alt="" /><Stamp type={d.doc.docType} confidence={d.doc.classification.confidence} animateIn={false} /></div>
                      <div className="body">
                        <div className="line">
                          <div className="grow">
                            <div className="who">{who ?? d.doc.fileName}</div>
                            {who && <div className="tiny muted">{d.doc.fileName}</div>}
                          </div>
                          <Ring value={trust.score} size={46} stroke={5} />
                        </div>
                        <div className="row" style={{ gap: '0.3rem' }}>
                          {alerts.slice(0, 2).map((a) => <span key={a.text} className={'tag ' + a.level} style={{ whiteSpace: 'normal' }}>{a.text.split('.')[0]}</span>)}
                          {critiqueOf(d).verdict === 'challenged' && <span className="tag bad">Critic doubts the type</span>}
                          {sides.get(d.doc.id) && <span className="tag">{sides.get(d.doc.id)!.side === 'front' ? 'Front' : 'Back'} · other side: {sides.get(d.doc.id)!.other.doc.fileName}</span>}
                          {n > 0 && <span className="tag warn">{n} to check</span>}
                          {!alerts.length && !n && !sides.get(d.doc.id) && <span className="tag good">Ready</span>}
                        </div>
                      </div>
                    </motion.button>
                  );
                })}
              </AnimatePresence>
            </div>
          </section>
        );
      })}
    </div>
  );
}
