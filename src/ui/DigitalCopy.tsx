// The digital copy: the scan on top, and under it the same document rebuilt from what was read,
// every line where it was printed (Arabic and code lines too, and the portrait). Each line can be
// tapped: it lights up on the scan, and a panel shows the close-up with the text, to mark it right or
// wrong or to correct it. Corrections reach the key facts, the fields and every export. The copy can be
// downloaded as a picture or a PDF.
import { useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'motion/react';
import { createPortal } from 'react-dom';
import type { Box } from '../core/types';
import { correctedText, type Correction, type Piece } from '../core/copy';
import { piecesOfState, MARKED_WRONG, type DocState, type EField } from './state';
import { guessPhoto } from './DocViewer';
import { niceName } from './bundle';
import { DOC_LABEL } from '../core/classify';
import { ICheck, IDownload, IX } from './icons';

interface Props {
  s: DocState;
  onPatch: (fn: (s: DocState) => DocState) => void;
}

/** where a line stands, from what the person said and what the readers found */
type Status = 'right' | 'fixed' | 'wrong' | 'proof' | 'unsure' | 'plain';
const UNSURE = 0.6;
const INK: Record<Status, string> = { right: '#0a7d4f', fixed: '#2648c9', wrong: '#c62e24', proof: '#0a7d4f', unsure: '#9a5a00', plain: '#1c1f26' };
const LABEL: Record<Status, string> = { right: 'checked', fixed: 'corrected', wrong: 'marked wrong', proof: 'verified', unsure: 'not sure', plain: '' };

export function DigitalCopy({ s, onPatch }: Props) {
  const { doc } = s;
  const [sel, setSel] = useState<string | null>(null);
  const [photo, setPhoto] = useState<Box | null>(null);
  const [busy, setBusy] = useState('');
  const svgRef = useRef<SVGSVGElement>(null);
  const words = doc.kind === 'id' && doc.words2?.length ? doc.words2 : doc.words;
  const pieces = useMemo(() => piecesOfState(s), [words, doc.arabic, doc.facts, doc.mrz, s.fields, s.analysis]); // eslint-disable-line react-hooks/exhaustive-deps
  const fixes = s.copy ?? {};
  useEffect(() => { let on = true; guessPhoto(s).then((b) => on && setPhoto(b)).catch(() => {}); return () => { on = false; }; }, [doc.id, doc.view]); // eslint-disable-line react-hooks/exhaustive-deps

  const factOf = (p: Piece) => (p.fact ? doc.facts.find((f) => f.key === p.fact) : undefined);
  const fieldOf = (p: Piece) => (p.field ? s.fields.find((f) => f.uid === p.field) : undefined);
  const statusOf = (p: Piece): Status => {
    const f = factOf(p), fl = fieldOf(p), c = fixes[p.id];
    if (f?.checked === 'wrong' || (fl && fl.status === 'review' && fl.reasons.includes(MARKED_WRONG)) || c?.verdict === 'wrong') return f?.edited || fl?.edited || c?.text !== undefined ? 'fixed' : 'wrong';
    if (f?.edited || fl?.edited || c?.text !== undefined) return 'fixed';
    if (f?.checked === 'right' || fl?.confirmed || c?.verdict === 'right') return 'right';
    if (p.verified) return 'proof';
    const conf = fl ? fl.confidence : p.conf;
    if (conf < UNSURE || f?.alt || (fl && fl.status === 'review')) return 'unsure';
    return 'plain';
  };
  /** the text shown for a line: a corrected fact or field value replaces what was printed for it */
  const shownText = (p: Piece) => {
    const f = factOf(p), fl = fieldOf(p);
    if (f?.edited && p.valueText) return p.text.includes(p.valueText) ? p.text.replace(p.valueText, f.value) : f.value;
    if (fl?.edited) return p.kind === 'mark' ? p.text : fl.value;
    return correctedText(p, fixes[p.id]);
  };

  const counts = pieces.reduce((a, p) => { const st = statusOf(p); a[st] = (a[st] ?? 0) + 1; return a; }, {} as Partial<Record<Status, number>>);
  const unsure = pieces.filter((p) => ['unsure', 'wrong'].includes(statusOf(p)));
  const selected = pieces.find((p) => p.id === sel) ?? null;
  const W = doc.width, H = doc.height;
  const card = doc.kind === 'id' && W > H;

  // ---- changes ----
  const setFix = (p: Piece, c: Correction | null) => onPatch((d) => { const copy = { ...(d.copy ?? {}) }; if (c) copy[p.id] = c; else delete copy[p.id]; return { ...d, copy }; });
  const judge = (p: Piece, verdict: 'right' | 'wrong', text?: string) => {
    const f = factOf(p), fl = fieldOf(p);
    if (f) {
      onPatch((d) => ({ ...d, doc: { ...d.doc, facts: d.doc.facts.map((x) => (x.key === f.key ? { ...x, checked: verdict, ...(text !== undefined && text !== x.value ? { value: text, edited: true, read: x.read ?? x.value } : {}) } : x)) } }));
    } else if (fl) {
      onPatch((d) => ({ ...d, fields: d.fields.map((x): EField => (x.uid !== fl.uid ? x
        : verdict === 'right' ? { ...x, confirmed: true, status: 'ok', ...(text !== undefined && text !== x.value ? { value: text, edited: true } : {}), reasons: x.reasons.filter((r) => r !== MARKED_WRONG) }
        : { ...x, confirmed: false, status: 'review', reasons: [...new Set([...x.reasons, MARKED_WRONG])], ...(text !== undefined && text !== x.value ? { value: text, edited: true } : {}) })) }));
    } else setFix(p, { verdict, ...(text !== undefined && text !== p.text ? { text } : {}) });
  };
  const allRight = () => {
    const sure = pieces.filter((p) => ['plain', 'proof'].includes(statusOf(p)));
    onPatch((d) => {
      const copy = { ...(d.copy ?? {}) };
      const factKeys = new Set(sure.map((p) => p.fact).filter(Boolean));
      const fieldIds = new Set(sure.map((p) => p.field).filter(Boolean));
      for (const p of sure) if (!p.fact && !p.field) copy[p.id] = { verdict: 'right' };
      return { ...d, copy, doc: { ...d.doc, facts: d.doc.facts.map((f) => (factKeys.has(f.key) && !f.checked ? { ...f, checked: 'right' as const } : f)) }, fields: d.fields.map((f) => (fieldIds.has(f.uid) ? { ...f, confirmed: true, status: 'ok' as const } : f)) };
    });
  };

  // ---- downloads ----
  const pictureBlob = async (scale = 2, type = 'image/png'): Promise<Blob> => {
    const svg = svgRef.current!.cloneNode(true) as SVGSVGElement;
    svg.querySelectorAll('[data-sel]').forEach((n) => n.remove());
    svg.setAttribute('width', String(W * scale)); svg.setAttribute('height', String(H * scale));
    svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    const url = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(svg)], { type: 'image/svg+xml' }));
    try {
      const img = new Image();
      img.src = url;
      await img.decode();
      const c = document.createElement('canvas');
      c.width = Math.round(W * scale); c.height = Math.round(H * scale);
      const x = c.getContext('2d')!;
      x.fillStyle = '#fff'; x.fillRect(0, 0, c.width, c.height);
      x.drawImage(img, 0, 0, c.width, c.height);
      return await new Promise((res, rej) => c.toBlob((b) => (b ? res(b) : rej(new Error('no picture'))), type, 0.88));
    } finally { URL.revokeObjectURL(url); }
  };
  const save = (blob: Blob, name: string) => { const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 4000); };
  const downloadPicture = async () => { setBusy('picture'); try { save(await pictureBlob(), niceName(s) + ' - digital copy.png'); } finally { setBusy(''); } };
  const downloadPdf = async () => {
    setBusy('pdf');
    try {
      const { jsPDF } = await import('jspdf');
      const pdf = new jsPDF({ unit: 'mm', format: 'a4' });
      const ascii = (t: string) => t.replace(/[^\x20-\xff]/g, '?');
      const PW = 210, M = 14, CW = PW - 2 * M;
      pdf.setFont('helvetica', 'bold'); pdf.setFontSize(15);
      pdf.text(ascii(`${DOC_LABEL[doc.docType]}: ${doc.facts.find((f) => f.key === 'Full name')?.value ?? doc.fileName}`), M, 18);
      pdf.setFont('helvetica', 'normal'); pdf.setFontSize(9); pdf.setTextColor(110, 110, 110);
      pdf.text('The scan, then the digital copy made from it on this device. Green: checked or verified. Amber: not sure. Blue: corrected. Red: marked wrong.', M, 24, { maxWidth: CW });
      const h = Math.min(105, (CW * H) / W), w = (h * W) / H;
      pdf.addImage(doc.view, 'JPEG', M + (CW - w) / 2, 30, w, h);
      const blob = await pictureBlob(Math.min(2, 1800 / Math.max(W, H)), 'image/jpeg');
      const png = await new Promise<string>((res) => { const r = new FileReader(); r.onload = () => res(String(r.result)); r.readAsDataURL(blob); });
      pdf.addImage(png, 'JPEG', M + (CW - w) / 2, 36 + h, w, h);
      // every line as text too, so it can be searched and copied (Arabic is in the picture: the PDF font has no Arabic letters)
      pdf.addPage();
      pdf.setTextColor(20, 20, 20); pdf.setFont('helvetica', 'bold'); pdf.setFontSize(12); pdf.text('Every line read', M, 18);
      pdf.setFont('helvetica', 'normal'); pdf.setFontSize(9);
      let y = 26;
      for (const p of pieces) {
        const st = statusOf(p), t = shownText(p);
        const line = `${/[؀-ۿ]/.test(t) ? '(Arabic: see page 1)' : ascii(t)}${LABEL[st] ? `   [${LABEL[st]}]` : ''}`;
        for (const row of pdf.splitTextToSize(line, CW) as string[]) { if (y > 285) { pdf.addPage(); y = 18; } pdf.text(row, M, y); y += 4.6; }
      }
      pdf.save(niceName(s) + ' - digital copy.pdf');
    } finally { setBusy(''); }
  };

  // ---- drawing ----
  const fontFor = (p: Piece) => (p.kind === 'mrz' ? 'ui-monospace, Menlo, Consolas, monospace' : 'Inter, -apple-system, "Segoe UI", Tahoma, Arial, sans-serif');
  const radius = card ? Math.min(W, H) * 0.045 : 0;
  // a line of ordinary print, so a big box (a logo, a signature space) does not get giant letters
  const lineH = (() => { const hs = pieces.filter((p) => p.kind === 'text').map((p) => p.box[3] - p.box[1]).sort((a, b) => a - b); return hs.length ? hs[hs.length >> 1] : H / 30; })();
  return (
    <section className="sheet dcopy">
      <div className="row dcopy-head">
        <h2 className="grow">Digital copy</h2>
        <span className="dcopy-count">{pieces.length} lines{counts.unsure ? ` · ${counts.unsure} not sure` : ''}{(counts.right ?? 0) + (counts.fixed ?? 0) ? ` · ${(counts.right ?? 0) + (counts.fixed ?? 0)} checked` : ''}</span>
      </div>
      <p className="small muted">Everything read from the document, where it was printed. Tap any line to check it or correct it. <span className="dkey"><span><i style={{ background: INK.proof }} />checked or verified</span><span><i style={{ background: INK.unsure }} />not sure</span><span><i style={{ background: INK.fixed }} />corrected</span><span><i style={{ background: INK.wrong }} />wrong</span></span></p>

      <div className="dcopy-pair">
        <figure className="dcopy-scan">
          <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="The scan">
            <image href={doc.view} width={W} height={H} />
            {selected && <rect x={selected.box[0] - 4} y={selected.box[1] - 4} width={selected.box[2] - selected.box[0] + 8} height={selected.box[3] - selected.box[1] + 8} fill="rgb(255 214 10 / 0.25)" stroke="#e0a800" strokeWidth={Math.max(2, W / 300)} rx={4} />}
          </svg>
          <figcaption>The scan</figcaption>
        </figure>
        <figure className="dcopy-doc">
          <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Digital copy of the document" xmlns="http://www.w3.org/2000/svg">
            <defs><clipPath id={`ph-${doc.id}`}>{photo && <rect x={photo[0]} y={photo[1]} width={photo[2] - photo[0]} height={photo[3] - photo[1]} rx={W / 120} />}</clipPath></defs>
            <rect width={W} height={H} rx={radius} fill="#fbfaf6" stroke="#d6d1c4" strokeWidth={Math.max(1, W / 500)} />
            {photo && <image href={doc.view} width={W} height={H} clipPath={`url(#ph-${doc.id})`} />}
            {pieces.map((p) => {
              const st = statusOf(p), [x0, y0, x1] = p.box, w = Math.max(4, x1 - x0);
              const h = Math.max(4, Math.min(p.box[3] - y0, lineH * (p.kind === 'mark' ? 1.3 : 2))), y1 = y0 + h;
              const rtl = p.kind === 'arabic', t = shownText(p), on = sel === p.id;
              const size = p.kind === 'arabic' ? h * 0.72 : h * 0.82;
              return (
                <g key={p.id} className="dpiece" onClick={() => setSel(p.id)} role="button" tabIndex={0} aria-label={`${t}${LABEL[st] ? ', ' + LABEL[st] : ''}`} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setSel(p.id); } }}>
                  {on && <rect data-sel="1" x={x0 - 4} y={y0 - 4} width={w + 8} height={h + 8} rx={4} fill="rgb(255 214 10 / 0.35)" />}
                  {/* the tap target is the whole line, a little taller than the text */}
                  <rect x={x0 - 2} y={y0 - h * 0.2} width={w + 4} height={h * 1.4} fill="transparent" />
                  {st === 'unsure' && <rect x={x0 - 3} y={y0 - 3} width={w + 6} height={h + 6} rx={3} fill="rgb(255 176 32 / 0.16)" stroke={INK.unsure} strokeWidth={Math.max(1, h / 14)} strokeDasharray={`${h / 4} ${h / 6}`} />}
                  <text x={rtl ? x1 : x0} y={y1 - h * 0.18} fontSize={size} fontFamily={fontFor(p)} fontWeight={p.fact || p.field ? 700 : 500} fill={INK[st]} direction={rtl ? 'rtl' : 'ltr'}
                    textLength={t.length > 2 && p.kind !== 'mark' ? w : undefined} lengthAdjust="spacingAndGlyphs">{t}</text>
                  {(st === 'right' || st === 'proof' || st === 'fixed' || st === 'wrong') && <line x1={x0} x2={x1} y1={y1 + h * 0.12} y2={y1 + h * 0.12} stroke={INK[st]} strokeWidth={Math.max(1, h / 10)} />}
                </g>
              );
            })}
          </svg>
          <figcaption>Digital copy</figcaption>
        </figure>
      </div>

      <div className="row dcopy-actions">
        {unsure.length > 0 && <button className="btn primary" onClick={() => setSel(unsure[0].id)}><ICheck /> Check the {unsure.length} not sure</button>}
        <button className="btn" onClick={allRight} title="Marks every line that is not flagged as checked"><ICheck /> The rest is right</button>
        <button className="btn" onClick={downloadPicture} disabled={!!busy}><IDownload /> {busy === 'picture' ? 'Making…' : 'Picture'}</button>
        <button className="btn" onClick={downloadPdf} disabled={!!busy}><IDownload /> {busy === 'pdf' ? 'Making…' : 'PDF'}</button>
      </div>

      {createPortal(
        <AnimatePresence>
          {selected && (
            <LinePanel key={selected.id} doc={s.doc} piece={selected} status={statusOf(selected)} text={shownText(selected)}
              factKey={selected.fact} fieldKey={fieldOf(selected)?.key} alt={factOf(selected)?.alt ?? fieldOf(selected)?.alt}
              value={factOf(selected)?.value ?? (fieldOf(selected) && selected.kind !== 'mark' ? fieldOf(selected)!.value : undefined)}
              conf={fieldOf(selected)?.confidence ?? selected.conf}
              onJudge={(v, t) => { judge(selected, v, t); const next = unsure.find((p) => p.id !== selected.id && p.box[1] >= selected.box[1]) ?? unsure.find((p) => p.id !== selected.id); setSel(v === 'right' && next ? next.id : v === 'right' ? null : selected.id); }}
              onUndo={() => { const f = factOf(selected), fl = fieldOf(selected);
                if (f) onPatch((d) => ({ ...d, doc: { ...d.doc, facts: d.doc.facts.map((x) => (x.key === f.key ? { ...x, checked: undefined } : x)) } }));
                else if (fl) onPatch((d) => ({ ...d, fields: d.fields.map((x) => (x.uid === fl.uid ? { ...x, confirmed: false, reasons: x.reasons.filter((r) => r !== MARKED_WRONG) } : x)) }));
                else setFix(selected, null); }}
              onClose={() => setSel(null)} />
          )}
        </AnimatePresence>,
        document.body,
      )}
    </section>
  );
}

function LinePanel(p: { doc: DocState['doc']; piece: Piece; status: Status; text: string; factKey?: string; fieldKey?: string; value?: string; alt?: string; conf: number;
  onJudge: (v: 'right' | 'wrong', text?: string) => void; onUndo: () => void; onClose: () => void }) {
  const editing = p.value ?? p.text;
  const [t, setT] = useState(editing);
  const [fixing, setFixing] = useState(p.status === 'wrong');
  const input = useRef<HTMLTextAreaElement>(null);
  useEffect(() => { if (fixing) input.current?.focus(); }, [fixing]);
  useEffect(() => { const k = (e: KeyboardEvent) => e.key === 'Escape' && p.onClose(); window.addEventListener('keydown', k); return () => window.removeEventListener('keydown', k); }, [p]);
  const [x0, y0, x1, y1] = p.piece.box, h = y1 - y0, pad = h * 0.9 + 8;
  const vb = [x0 - pad, y0 - pad, x1 - x0 + 2 * pad, h + 2 * pad].join(' ');
  const rtl = p.piece.kind === 'arabic' || /[؀-ۿ]/.test(t);
  const title = p.factKey ?? p.fieldKey ?? (p.piece.kind === 'mrz' ? 'Code line' : p.piece.kind === 'arabic' ? 'Line in Arabic' : 'Line read');
  const changed = t !== editing;
  return (
    <motion.div className="dpanel" role="dialog" aria-label={`Check: ${title}`} initial={{ y: '100%' }} animate={{ y: 0 }} exit={{ y: '100%' }} transition={{ type: 'spring', stiffness: 420, damping: 38 }}>
      <div className="row dpanel-head">
        <b className="grow">{title}</b>
        {p.status !== 'plain' && <span className={'dstate ' + p.status}>{LABEL[p.status]}</span>}
        <button className="btn icon ghost" onClick={p.onClose} aria-label="Close"><IX /></button>
      </div>
      <div className="dpanel-crop"><svg viewBox={vb} preserveAspectRatio="xMidYMid meet"><image href={p.doc.view} width={p.doc.width} height={p.doc.height} /><rect x={x0 - 3} y={y0 - 3} width={x1 - x0 + 6} height={h + 6} fill="none" stroke="#e0a800" strokeWidth={Math.max(1.5, h / 12)} rx={3} /></svg></div>
      {p.value !== undefined && p.factKey && <p className="small muted">The line on the document: <q dir="auto">{p.text}</q></p>}
      <p className="small muted">How sure the reader was: {Math.round(p.conf * 100)}%{p.piece.verified ? ' · proven by check digits' : ''}</p>
      {fixing || changed
        ? <textarea ref={input} className="dpanel-input" dir={rtl ? 'rtl' : 'auto'} rows={Math.min(4, Math.max(1, Math.ceil(t.length / 28)))} value={t} onChange={(e) => setT(e.target.value)} aria-label="The correct text" />
        : <p className="dpanel-text" dir="auto">{editing}</p>}
      {p.alt && <p className="small">The other reader saw <q dir="auto">{p.alt}</q> <button className="btn sm" onClick={() => { setT(p.alt!); setFixing(true); }}>Use this</button></p>}
      <div className="row dpanel-actions">
        {fixing || changed
          ? <><button className="btn primary" disabled={!t.trim()} onClick={() => p.onJudge('right', t)}><ICheck /> Save correction</button>
              <button className="btn" onClick={() => p.onJudge('wrong')}>Still wrong, leave it flagged</button></>
          : <><button className="btn primary" onClick={() => p.onJudge('right')}><ICheck /> Right</button>
              <button className="btn" onClick={() => setFixing(true)}><IX /> Wrong, correct it</button></>}
        {p.status !== 'plain' && p.status !== 'unsure' && p.status !== 'proof' && <button className="btn ghost sm" onClick={p.onUndo}>Undo</button>}
      </div>
    </motion.div>
  );
}
