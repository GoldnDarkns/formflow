import { useEffect, useRef, useState } from 'react';
import { motion } from 'motion/react';
import { download as _d, sensitiveEntities, type DocState } from './state';
import { cx, cy } from '../core/layout';
import { ID_TYPES } from '../core/classify';
import { findPhotoArea } from '../core/photo';
import { rgbaToGray } from '../core/preprocess';
import type { Box } from '../core/types';
import type { Quad } from '../core/scan';
import { union } from '../core/layout';
import { buildTemplate, type Template } from '../core/template';

async function loadImage(src: string) {
  const img = new Image();
  img.src = src;
  await img.decode();
  return img;
}

/** Guess where the portrait photo is on an ID page (null for forms or when nothing fits). */
export async function guessPhoto(s: DocState): Promise<Box | null> {
  if (!ID_TYPES.includes(s.doc.docType)) return null;
  const img = await loadImage(s.doc.view);
  const k = Math.min(1, 640 / img.width);
  const c = document.createElement('canvas');
  c.width = Math.round(img.width * k); c.height = Math.round(img.height * k);
  const ctx = c.getContext('2d')!;
  ctx.drawImage(img, 0, 0, c.width, c.height);
  const gray = rgbaToGray(ctx.getImageData(0, 0, c.width, c.height).data, c.width, c.height);
  const sx = c.width / s.doc.width, sy = c.height / s.doc.height;
  const text = s.doc.words.map((w) => [w.box[0] * sx, w.box[1] * sy, w.box[2] * sx, w.box[3] * sy] as Box);
  const b = findPhotoArea(gray, text);
  return b ? [b[0] / sx, b[1] / sy, b[2] / sx, b[3] / sy] : null;
}

const COLORS: Record<string, string> = { question: 'var(--q)', answer: 'var(--a)', header: 'var(--h)', other: 'var(--o)' };

interface Props {
  s: DocState;
  selected: string | null;
  onSelect: (uid: string | null) => void;
  onTeach?: (t: Template) => void;
  onCorners?: (q: Quad) => void;
}

export function DocViewer({ s, selected, onSelect, onTeach, onCorners }: Props) {
  const [mode, setMode] = useState<'page' | 'reader' | 'photo'>('page');
  const processed = mode === 'reader';
  const [redact, setRedact] = useState(false);
  const [hidden, setHidden] = useState<Set<number>>(new Set());
  // extra black boxes: the guessed photo area plus any the person draws
  const [extra, setExtra] = useState<{ id: number; box: Box; photo?: boolean }[]>([]);
  const [draft, setDraft] = useState<Box | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const drag = useRef<{ x: number; y: number } | null>(null);
  const nextId = useRef(1);
  // before / after: a divider the person drags across the page to compare the scan with what the reader saw
  const [split, setSplit] = useState(50);
  const pageRef = useRef<HTMLDivElement>(null);
  const moveSplit = (ev: React.PointerEvent) => {
    const r = pageRef.current!.getBoundingClientRect();
    setSplit(Math.min(100, Math.max(0, ((ev.clientX - r.left) / r.width) * 100)));
  };
  // corner mode: drag the four page corners on the photo, then read the page again
  const FULL: Quad = [[0.04, 0.04], [0.96, 0.04], [0.96, 0.96], [0.04, 0.96]];
  const [corners, setCorners] = useState<Quad | null>(null);
  const photoRef = useRef<HTMLDivElement>(null);
  const moveCorner = (i: number, ev: React.PointerEvent) => {
    if (!(ev.buttons & 1) && ev.pointerType === 'mouse') return;
    const r = photoRef.current!.getBoundingClientRect();
    const x = Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width)), y = Math.min(1, Math.max(0, (ev.clientY - r.top) / r.height));
    setCorners((q) => q && (q.map((p, j) => (j === i ? [x, y] : p)) as Quad));
  };
  useEffect(() => { setCorners(null); }, [s.doc.id]);
  // teach mode: the person draws one labelled box per place a value goes, and the page becomes a template
  const [teach, setTeach] = useState(false);
  const [slots, setSlots] = useState<{ id: number; box: Box; label: string }[]>([]);
  const [selSlot, setSelSlot] = useState<number | null>(null);
  const [tplName, setTplName] = useState('');
  const labelRef = useRef<HTMLInputElement>(null);
  const startTeach = () => {
    // start from the boxes the app already found, so only the wrong ones need fixing
    const seed = s.fields.filter((f) => f.qId !== undefined && f.key.trim()).map((f) => {
      const boxes = s.analysis.links.filter((l) => l.q === f.qId).map((l) => s.analysis.entities[l.a]?.box).filter(Boolean) as Box[];
      return boxes.length ? { id: nextId.current++, box: union(boxes), label: f.key } : null;
    }).filter((x): x is { id: number; box: Box; label: string } => !!x);
    setSlots(seed); setSelSlot(null); setTplName(s.doc.template?.template.name ?? ''); setTeach(true); setRedact(false); setMode('page');
  };
  const saveTemplate = () => {
    const good = slots.filter((x) => x.label.trim());
    if (!good.length || !tplName.trim() || !onTeach) return;
    const W = s.doc.width, H = s.doc.height;
    const t = buildTemplate('user-' + Date.now(), tplName.trim(), s.doc.words, W, H, good.map((x) => ({ label: x.label.trim(), box: [x.box[0] / W, x.box[1] / H, x.box[2] / W, x.box[3] / H] as Box })));
    onTeach({ ...t, filled: true });
    setTeach(false);
  };
  useEffect(() => { setRedact(false); setMode('page'); setExtra([]); setTeach(false); }, [s.doc.id]);
  const startRedact = () => {
    setHidden(sensitiveEntities(s)); setRedact(true); setMode('page'); setExtra([]);
    guessPhoto(s).then((b) => { if (b) setExtra((e) => (e.some((x) => x.photo) ? e : [...e, { id: nextId.current++, box: b, photo: true }])); }).catch(() => {});
  };
  const toDoc = (ev: { clientX: number; clientY: number }) => {
    const r = svgRef.current!.getBoundingClientRect();
    return { x: ((ev.clientX - r.left) / r.width) * s.doc.width, y: ((ev.clientY - r.top) / r.height) * s.doc.height };
  };
  const onDown = (ev: React.PointerEvent<SVGRectElement>) => {
    drag.current = toDoc(ev);
    ev.currentTarget.setPointerCapture(ev.pointerId);
  };
  const onMove = (ev: React.PointerEvent<SVGRectElement>) => {
    if (!drag.current) return;
    const p = toDoc(ev), d = drag.current;
    setDraft([Math.min(d.x, p.x), Math.min(d.y, p.y), Math.max(d.x, p.x), Math.max(d.y, p.y)]);
  };
  const onUp = () => {
    drag.current = null;
    setDraft((b) => {
      if (b && b[2] - b[0] > s.doc.width * 0.015 && b[3] - b[1] > s.doc.height * 0.01) {
        if (teach) { const id = nextId.current++; setSlots((e) => [...e, { id, box: b, label: '' }]); setSelSlot(id); setTimeout(() => labelRef.current?.focus(), 30); }
        else setExtra((e) => [...e, { id: nextId.current++, box: b }]);
      }
      return null;
    });
  };
  const toggleHidden = (id: number) => setHidden((h) => { const n = new Set(h); n.has(id) ? n.delete(id) : n.add(id); return n; });
  const saveRedacted = async () => {
    const img = await loadImage(s.doc.view);
    const c = document.createElement('canvas');
    c.width = img.width; c.height = img.height;
    const ctx = c.getContext('2d')!;
    ctx.drawImage(img, 0, 0);
    ctx.fillStyle = '#000';
    for (const e of s.analysis.entities) if (hidden.has(e.id)) ctx.fillRect(e.box[0] - 4, e.box[1] - 4, e.box[2] - e.box[0] + 8, e.box[3] - e.box[1] + 8);
    for (const x of extra) ctx.fillRect(x.box[0], x.box[1], x.box[2] - x.box[0], x.box[3] - x.box[1]);
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
          {(doc.quad || onCorners) && <button className={mode === 'photo' ? 'on' : ''} onClick={() => { setMode('photo'); setRedact(false); setTeach(false); }}>Photo</button>}
          <button className={mode === 'page' ? 'on' : ''} onClick={() => setMode('page')}>{doc.quad ? 'Cut out' : 'Page'}</button>
          <button className={mode === 'reader' ? 'on' : ''} onClick={() => { setMode('reader'); setSplit(50); setRedact(false); setTeach(false); }}>Before / after</button>
        </div>
      </div>
      {redact && (
        <div className="redactbar">
          <span className="grow">{hidden.size + extra.length} areas hidden{extra.some((x) => x.photo) ? ', including the photo (a guess, check it)' : ''}. Tap a box to hide or show it. Drag on the page to black out anything else, such as a photo or signature.</span>
          <button className="btn sm" style={{ color: 'var(--ink)' }} onClick={saveRedacted}>Save redacted copy</button>
          <button className="btn sm ghost" style={{ color: 'var(--sheet)' }} onClick={() => setRedact(false)}>Done</button>
        </div>
      )}
      {teach && (
        <div className="redactbar teachbar">
          <span className="grow">Draw a box around each place a value is written and name it. Boxes found already are a starting point: tap one to rename or remove it.</span>
          <input className="tin" placeholder="Name of this form" value={tplName} onChange={(e) => setTplName(e.target.value)} aria-label="Name of this form" />
          {selSlot !== null && (
            <>
              <input ref={labelRef} className="tin" placeholder="Label for the selected box" value={slots.find((x) => x.id === selSlot)?.label ?? ''} onChange={(e) => setSlots((ss) => ss.map((x) => (x.id === selSlot ? { ...x, label: e.target.value } : x)))} aria-label="Label for the selected box" />
              <button className="btn sm ghost" style={{ color: 'var(--sheet)' }} onClick={() => { setSlots((ss) => ss.filter((x) => x.id !== selSlot)); setSelSlot(null); }}>Remove box</button>
            </>
          )}
          <button className="btn sm" style={{ color: 'var(--ink)' }} disabled={!tplName.trim() || !slots.some((x) => x.label.trim())} onClick={saveTemplate}>Save template ({slots.filter((x) => x.label.trim()).length} boxes)</button>
          <button className="btn sm ghost" style={{ color: 'var(--sheet)' }} onClick={() => setTeach(false)}>Cancel</button>
        </div>
      )}
      {mode === 'photo' && onCorners && (
        <div className="redactbar">
          {corners ? (
            <>
              <span className="grow">Drag each yellow dot onto a corner of the page, then read it again.</span>
              <button className="btn sm" style={{ color: 'var(--ink)' }} onClick={() => { onCorners(corners); setCorners(null); }}>Read again with these corners</button>
              <button className="btn sm ghost" style={{ color: 'var(--sheet)' }} onClick={() => setCorners(null)}>Cancel</button>
            </>
          ) : (
            <>
              <span className="grow">{doc.quad ? (doc.manualQuad ? 'The page corners were set by hand.' : 'The outline shows where FormFlow found the page.') : 'No page outline was found, so the whole picture was used.'} If the outline is wrong, fix it.</span>
              <button className="btn sm" style={{ color: 'var(--ink)' }} onClick={() => setCorners(doc.quad ?? FULL)}>Adjust corners</button>
            </>
          )}
        </div>
      )}
      {mode === 'photo' && (doc.quad || onCorners) ? (
        <div className="docwrap" ref={photoRef} style={corners ? { touchAction: 'none' } : undefined}>
          <img src={doc.original} alt="The photo as taken" />
          <svg viewBox="0 0 100 100" preserveAspectRatio="none">
            {(corners ?? doc.quad) && <path fillRule="evenodd" fill="rgb(8 12 30 / 0.6)" d={`M0 0H100V100H0Z M${(corners ?? doc.quad)!.map(([x, y]) => `${x * 100} ${y * 100}`).join(' L')}Z`} />}
            {(corners ?? doc.quad) && <polygon points={(corners ?? doc.quad)!.map(([x, y]) => `${x * 100},${y * 100}`).join(' ')} fill="none" stroke="var(--marker)" strokeWidth="2.5" vectorEffect="non-scaling-stroke" />}
          </svg>
          {corners?.map(([x, y], i) => (
            <div key={i} className="corner" role="slider" aria-label={`Page corner ${i + 1}`} aria-valuenow={Math.round(x * 100)} tabIndex={0} style={{ left: `${x * 100}%`, top: `${y * 100}%` }}
              onPointerDown={(ev) => ev.currentTarget.setPointerCapture(ev.pointerId)} onPointerMove={(ev) => ev.currentTarget.hasPointerCapture(ev.pointerId) && moveCorner(i, ev)} />
          ))}
        </div>
      ) : (
      <div className="docwrap" ref={pageRef}>
        <img src={processed ? doc.processedView : doc.view} alt={processed ? 'The cleaned page the reader saw' : 'Scanned document'} />
        {processed && <img className="cmp" src={doc.view} alt="The scan before clean-up" style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }} />}
        <svg ref={svgRef} viewBox={`0 0 ${doc.width} ${doc.height}`} preserveAspectRatio="none" style={redact || teach ? { touchAction: 'none' } : undefined}>
          {/* in redact mode, dragging on empty page draws a new black box */}
          {(redact || teach) && <rect x={0} y={0} width={doc.width} height={doc.height} fill="transparent" style={{ cursor: 'crosshair' }} onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp} />}
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
          {layers.links && !teach && !processed && analysis.links.filter((l) => ents[l.q].words.length > 0).map((l, i) => {
            const q = ents[l.q].box, a = ents[l.a].box;
            const right = a[0] >= q[2] - 5;
            return <line key={i} x1={right ? q[2] : cx(q)} y1={right ? cy(q) : q[3]} x2={right ? a[0] : cx(a)} y2={right ? cy(a) : a[1]} stroke="var(--ink-3)" strokeWidth={lw * 1.2} opacity={0.35 + 0.65 * l.p} markerEnd="url(#arr)" />;
          })}
          {layers.boxes && !teach && !processed && ents.filter((e) => e.box[2] - e.box[0] > 2).map((e) => {
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
          {redact && extra.map((x) => (
            <motion.rect key={'x' + x.id} x={x.box[0]} y={x.box[1]} width={x.box[2] - x.box[0]} fill="#000" initial={{ height: 0 }} animate={{ height: x.box[3] - x.box[1] }} transition={{ duration: 0.25 }} style={{ cursor: 'pointer' }} onClick={() => setExtra((e) => e.filter((y) => y.id !== x.id))}>
              <title>{x.photo ? 'Photo area (guess). Tap to show it again.' : 'Tap to remove this box.'}</title>
            </motion.rect>
          ))}
          {teach && slots.map((x) => (
            <g key={'t' + x.id} style={{ cursor: 'pointer' }} onClick={() => { setSelSlot(x.id); setTimeout(() => labelRef.current?.focus(), 30); }}>
              <rect x={x.box[0]} y={x.box[1]} width={x.box[2] - x.box[0]} height={x.box[3] - x.box[1]} rx={3} fill="var(--marker)" fillOpacity={x.id === selSlot ? 0.45 : 0.22} stroke={x.label.trim() ? 'var(--indigo)' : 'var(--warn)'} strokeWidth={x.id === selSlot ? lw * 2.4 : lw * 1.3} strokeDasharray={x.label.trim() ? undefined : `${lw * 3} ${lw * 2}`} />
              <title>{x.label || 'No label yet'}</title>
            </g>
          ))}
          {(redact || teach) && draft && <rect x={draft[0]} y={draft[1]} width={draft[2] - draft[0]} height={draft[3] - draft[1]} fill="#000" fillOpacity={0.6} stroke="var(--marker)" strokeWidth={lw} pointerEvents="none" />}
        </svg>
        {processed && (
          <>
            <span className="cmp-tag l" style={{ opacity: split > 14 ? 1 : 0 }}>Scan</span>
            <span className="cmp-tag r" style={{ opacity: split < 78 ? 1 : 0 }}>What the reader saw</span>
            <div className="cmp-bar" style={{ left: `${split}%` }} role="slider" tabIndex={0} aria-label="Compare the scan with what the reader saw" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(split)}
              onPointerDown={(ev) => { ev.currentTarget.setPointerCapture(ev.pointerId); moveSplit(ev); }} onPointerMove={(ev) => ev.currentTarget.hasPointerCapture(ev.pointerId) && moveSplit(ev)}
              onKeyDown={(ev) => { if (ev.key === 'ArrowLeft') setSplit((v) => Math.max(0, v - 4)); if (ev.key === 'ArrowRight') setSplit((v) => Math.min(100, v + 4)); }}>
              <i />
            </div>
          </>
        )}
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
          {!redact && !teach && onTeach && doc.kind === 'form' && <button className="btn sm" onClick={startTeach}>{doc.template ? 'Edit template' : 'Teach this form'}</button>}
          {!redact && !teach && <button className="btn sm" onClick={startRedact}>Redact</button>}
        </div>
      </div>
    </div>
  );
}
