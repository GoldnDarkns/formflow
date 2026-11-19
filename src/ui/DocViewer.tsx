import { useEffect, useState } from 'react';
import { motion } from 'motion/react';
import { download as _d, sensitiveEntities, type DocState } from './state';
import { cx, cy } from '../core/layout';

const COLORS: Record<string, string> = { question: 'var(--q)', answer: 'var(--a)', header: 'var(--h)', other: 'var(--o)' };

interface Props {
  s: DocState;
  selected: string | null;
  onSelect: (uid: string | null) => void;
}

export function DocViewer({ s, selected, onSelect }: Props) {
  const [mode, setMode] = useState<'page' | 'reader' | 'photo'>('page');
  const processed = mode === 'reader';
  const [redact, setRedact] = useState(false);
  const [hidden, setHidden] = useState<Set<number>>(new Set());
  useEffect(() => { setRedact(false); setMode('page'); }, [s.doc.id]);
  const startRedact = () => { setHidden(sensitiveEntities(s)); setRedact(true); setMode('page'); };
  const toggleHidden = (id: number) => setHidden((h) => { const n = new Set(h); n.has(id) ? n.delete(id) : n.add(id); return n; });
  const saveRedacted = async () => {
    const img = new Image();
    img.src = s.doc.view;
    await img.decode();
    const c = document.createElement('canvas');
    c.width = img.width; c.height = img.height;
    const ctx = c.getContext('2d')!;
    ctx.drawImage(img, 0, 0);
    ctx.fillStyle = '#000';
    for (const e of s.analysis.entities) if (hidden.has(e.id)) ctx.fillRect(e.box[0] - 4, e.box[1] - 4, e.box[2] - e.box[0] + 8, e.box[3] - e.box[1] + 8);
    const a = document.createElement('a');
    a.href = c.toDataURL('image/jpeg', 0.9);
    a.download = s.doc.fileName.replace(/\.[^.]+$/, '') + '-redacted.jpg';
    a.click();
  };
  void _d;
  const [layers, setLayers] = useState({ boxes: true, links: true, second: false });
  const { doc, analysis } = s;
  const ents = analysis.entities;
  const sel = s.fields.find((f) => f.uid === selected);
  const selIds = new Set<number>(sel ? [sel.qId!, ...analysis.links.filter((l) => l.q === sel.qId).map((l) => l.a)].filter((x) => x !== undefined) : []);
  const reviewA = new Set(s.fields.filter((f) => f.status === 'review' && !f.confirmed).flatMap((f) => analysis.links.filter((l) => l.q === f.qId).map((l) => l.a)));
  const lw = Math.max(1.5, doc.width / 520);
  const fieldOf = (id: number) => s.fields.find((f) => f.qId === id || analysis.links.some((l) => l.a === id && l.q === f.qId));

  return (
    <div className="sheet viewer stack" style={{ gap: '0.7rem' }}>
      <div className="row">
        <span className="grow small muted">{doc.fileName}, read in {(doc.totalMs / 1000).toFixed(1)}s</span>
        <div className="seg">
          {doc.quad && <button className={mode === 'photo' ? 'on' : ''} onClick={() => setMode('photo')}>Photo</button>}
          <button className={mode === 'page' ? 'on' : ''} onClick={() => setMode('page')}>{doc.quad ? 'Cut out' : 'Page'}</button>
          <button className={mode === 'reader' ? 'on' : ''} onClick={() => setMode('reader')}>What the reader saw</button>
        </div>
      </div>
      {redact && (
        <div className="redactbar">
          <span className="grow">{hidden.size} areas hidden. Tap any box to hide or show it. Photos are not hidden automatically.</span>
          <button className="btn sm" style={{ color: 'var(--ink)' }} onClick={saveRedacted}>Save redacted copy</button>
          <button className="btn sm ghost" style={{ color: 'var(--sheet)' }} onClick={() => setRedact(false)}>Done</button>
        </div>
      )}
      {mode === 'photo' && doc.quad ? (
        <div className="docwrap">
          <img src={doc.original} alt="The photo as taken" />
          <svg viewBox="0 0 100 100" preserveAspectRatio="none">
            <path fillRule="evenodd" fill="rgb(8 12 30 / 0.6)" d={`M0 0H100V100H0Z M${doc.quad.map(([x, y]) => `${x * 100} ${y * 100}`).join(' L')}Z`} />
            <polygon points={doc.quad.map(([x, y]) => `${x * 100},${y * 100}`).join(' ')} fill="none" stroke="var(--marker)" strokeWidth="2.5" vectorEffect="non-scaling-stroke" />
          </svg>
        </div>
      ) : (
      <div className="docwrap">
        <img src={processed ? doc.processedView : doc.view} alt="Scanned document" />
        <svg viewBox={`0 0 ${doc.width} ${doc.height}`} preserveAspectRatio="none">
          <defs>
            <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="var(--ink-3)" /></marker>
          </defs>
          {/* highlighter pen over the selected field */}
          {ents.filter((e) => selIds.has(e.id)).map((e) => (
            <motion.rect key={'hl' + e.id} className="hl" x={e.box[0] - 5} y={e.box[1] - 4} height={e.box[3] - e.box[1] + 8} rx={4} fill="var(--marker)" initial={{ width: 0 }} animate={{ width: e.box[2] - e.box[0] + 10 }} transition={{ duration: 0.28, ease: 'easeOut' }} />
          ))}
          {layers.second && doc.second?.map((r, i) => (
            <rect key={'s' + i} x={r.box[0]} y={r.box[1]} width={r.box[2] - r.box[0]} height={r.box[3] - r.box[1]} fill="none" stroke="var(--t-visa)" strokeWidth={lw} strokeDasharray={`${lw * 3} ${lw * 2}`} />
          ))}
          {layers.links && analysis.links.map((l, i) => {
            const q = ents[l.q].box, a = ents[l.a].box;
            const right = a[0] >= q[2] - 5;
            return <line key={i} x1={right ? q[2] : cx(q)} y1={right ? cy(q) : q[3]} x2={right ? a[0] : cx(a)} y2={right ? cy(a) : a[1]} stroke="var(--ink-3)" strokeWidth={lw * 1.2} opacity={0.35 + 0.65 * l.p} markerEnd="url(#arr)" />;
          })}
          {layers.boxes && ents.map((e) => {
            const warn = reviewA.has(e.id);
            return (
              <rect key={e.id} x={e.box[0] - 2} y={e.box[1] - 2} width={e.box[2] - e.box[0] + 4} height={e.box[3] - e.box[1] + 4} rx={3}
                fill={COLORS[e.label ?? 'other']} fillOpacity={0.08} stroke={warn ? 'var(--warn)' : COLORS[e.label ?? 'other']} strokeWidth={warn ? lw * 1.8 : lw}
                strokeDasharray={warn ? `${lw * 3} ${lw * 2}` : undefined} style={{ cursor: 'pointer' }} onClick={() => (redact ? toggleHidden(e.id) : onSelect(fieldOf(e.id)?.uid ?? null))}>
                <title>{`${e.label}: ${e.text}`}</title>
              </rect>
            );
          })}
          {redact && ents.filter((e) => hidden.has(e.id)).map((e) => (
            <motion.rect key={'r' + e.id} x={e.box[0] - 4} y={e.box[1] - 4} height={e.box[3] - e.box[1] + 8} fill="#000" initial={{ width: 0 }} animate={{ width: e.box[2] - e.box[0] + 8 }} transition={{ duration: 0.25 }} style={{ cursor: 'pointer' }} onClick={() => toggleHidden(e.id)} />
          ))}
        </svg>
      </div>
      )}
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <div className="legend">
          <span><i style={{ background: 'var(--q)' }} />Label</span>
          <span><i style={{ background: 'var(--a)' }} />Value</span>
          <span><i style={{ background: 'var(--h)' }} />Heading</span>
          <span><i style={{ border: '2px dashed var(--warn)' }} />To check</span>
        </div>
        <div className="row small">
          <label><input type="checkbox" checked={layers.boxes} onChange={(e) => setLayers({ ...layers, boxes: e.target.checked })} /> Boxes</label>
          <label><input type="checkbox" checked={layers.links} onChange={(e) => setLayers({ ...layers, links: e.target.checked })} /> Links</label>
          {doc.second && <label><input type="checkbox" checked={layers.second} onChange={(e) => setLayers({ ...layers, second: e.target.checked })} /> Second reader</label>}
          {!redact && <button className="btn sm" onClick={startRedact}>Redact</button>}
        </div>
      </div>
    </div>
  );
}
