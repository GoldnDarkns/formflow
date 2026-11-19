// PDF out: a clean summary report, one page per document, built in the browser (jsPDF).
import { DOC_LABEL } from '../core/classify';
import { alertsFor } from '../core/facts';
import { packetsOf, trustOf, type DocState } from './state';

const clean = (s: string) => s.replace(/[^\x20-\xff]/g, '?');

export async function downloadReport(states: DocState[], fileName = 'formflow-report.pdf') {
  (await buildReport(states)).save(fileName);
}

/** The summary report as a jsPDF document (saved directly, or put inside the ZIP export). */
export async function buildReport(states: DocState[]) {
  const { jsPDF } = await import('jspdf');
  const pdf = new jsPDF({ unit: 'mm', format: 'a4' });
  const L = 16, R = 194;
  let y = 0;
  const line = (h = 5) => { y += h; if (y > 280) { pdf.addPage(); y = 18; } };
  const text = (s: string, x: number, size = 10, style: 'normal' | 'bold' = 'normal', color: [number, number, number] = [16, 26, 58], maxW = R - x) => {
    pdf.setFont('helvetica', style); pdf.setFontSize(size); pdf.setTextColor(...color);
    const rows = pdf.splitTextToSize(clean(s), maxW) as string[];
    rows.forEach((r, i) => { if (i) line(size * 0.42); pdf.text(r, x, y); });
  };
  const grey: [number, number, number] = [110, 120, 150];

  const packets = packetsOf(states);
  if (packets.length) {
    y = 20; text('Cross-document checks', L, 18, 'bold'); line(9);
    for (const p of packets) {
      text(p.name, L, 12, 'bold'); line(6);
      for (const c of p.checks) { text(`${c.ok ? 'OK' : 'MISMATCH'}  ${c.label}: ${c.detail}`, L + 2, 10, 'normal', c.ok ? [10, 143, 91] : [211, 52, 42]); line(6); }
      line(3);
    }
    pdf.addPage();
  }
  states.forEach((s, i) => {
    if (i) pdf.addPage();
    const d = s.doc, trust = trustOf(s);
    y = 20;
    text(d.facts.find((f) => f.key === 'Full name')?.value ?? d.fileName, L, 18, 'bold', undefined, 120); line(7);
    text(`${DOC_LABEL[d.docType]} (${Math.round(d.classification.confidence * 100)}% sure)   Trust score ${Math.round(trust.score * 100)}%   File: ${d.fileName}`, L, 9, 'normal', grey, 120); line(4);
    try {
      const w = 52, h = Math.min(70, (w * d.height) / d.width);
      pdf.addImage(d.view, 'JPEG', R - w, 12, w, h);
      y = Math.max(y, 12 + h);
    } catch {}
    line(6);
    for (const a of alertsFor(d.docType, d.classification.confidence, d.facts, d.mrz)) { text(`${a.level === 'good' ? 'OK' : a.level === 'warn' ? 'CHECK' : 'PROBLEM'}  ${a.text}`, L, 10, 'normal', a.level === 'good' ? [10, 143, 91] : a.level === 'warn' ? [178, 94, 0] : [211, 52, 42]); line(5.5); }
    if (d.facts.length) {
      line(4); text('Key facts', L, 12, 'bold'); line(6);
      for (const f of d.facts) { text(f.key, L, 10, 'normal', grey); text(f.value + (f.proof ? (f.valid ? `   (${f.proof} passes)` : `   (${f.proof} FAILS)`) : ''), L + 48, 10, 'bold'); line(5.5); }
    }
    line(4); text('Fields', L, 12, 'bold'); line(6);
    for (const f of s.fields.filter((f) => f.value)) {
      const status = f.confirmed ? 'checked' : f.status === 'review' ? 'to check' : `${Math.round(f.confidence * 100)}%`;
      const y0 = y;
      text(f.key, L, 9, 'normal', grey, 52);
      const yk = y; y = y0;
      text(f.value, L + 56, 9, 'bold', undefined, 96);
      const yv = y; y = y0;
      text(status, R - 18, 9, 'normal', f.confirmed ? [10, 143, 91] : f.status === 'review' ? [178, 94, 0] : grey);
      y = Math.max(yk, yv); line(5.2);
    }
    pdf.setFontSize(8); pdf.setTextColor(...grey);
    pdf.text(`FormFlow report, ${new Date().toISOString().slice(0, 10)}. Values marked "to check" have not been confirmed by a person.`, L, 290);
  });
  return pdf;
}
