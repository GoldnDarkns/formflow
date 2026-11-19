// Grouped downloads. After a mixed pile has been sorted, the clerk usually
// wants it back in groups: one PDF with all the passports, one with all the
// Emirates IDs, and so on, or everything at once as a ZIP with a folder per
// document type. All of it is built in the browser; nothing is uploaded.
import { DOC_LABEL, DOC_TYPES, type DocType } from '../core/classify';
import { buildReport } from './report';
import { exportCsv, exportJson, type DocState } from './state';

const slug = (s: string) => s.replace(/[^\w.-]+/g, '-').replace(/^-+|-+$/g, '') || 'file';
const today = () => new Date().toISOString().slice(0, 10);

/** One PDF holding the scans themselves, one page per scan, each page shaped like its scan. */
export async function scansPdf(states: DocState[]): Promise<Blob> {
  const { jsPDF } = await import('jspdf');
  let pdf: InstanceType<typeof jsPDF> | null = null;
  for (const s of states) {
    const d = s.doc;
    const landscape = d.width > d.height;
    // A4 with a small margin; the scan keeps its own proportions
    const PW = landscape ? 297 : 210, PH = landscape ? 210 : 297, M = 8, FOOT = 7;
    if (!pdf) pdf = new jsPDF({ unit: 'mm', format: 'a4', orientation: landscape ? 'landscape' : 'portrait' });
    else pdf.addPage('a4', landscape ? 'landscape' : 'portrait');
    const k = Math.min((PW - 2 * M) / d.width, (PH - 2 * M - FOOT) / d.height);
    const w = d.width * k, h = d.height * k;
    pdf.addImage(d.view, 'JPEG', (PW - w) / 2, M, w, h);
    pdf.setFont('helvetica', 'normal'); pdf.setFontSize(8); pdf.setTextColor(110, 110, 100);
    const who = d.facts.find((f) => f.key === 'Full name')?.value;
    pdf.text(`${DOC_LABEL[d.docType]}${who ? ' · ' + who.replace(/[^\x20-\xff]/g, '?') : ''} · ${d.fileName.replace(/[^\x20-\xff]/g, '?')} · FormFlow ${today()}`, M, PH - 4);
  }
  return pdf!.output('blob');
}

function save(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

/** All scans of one document type as a single PDF. */
export async function downloadTypePdf(docs: DocState[], type: DocType) {
  const list = docs.filter((d) => d.doc.docType === type);
  if (!list.length) return;
  save(await scansPdf(list), `${slug(DOC_LABEL[type])}-${list.length}-scan${list.length === 1 ? '' : 's'}.pdf`);
}

/** Everything as one ZIP: a folder per document type (scans PDF + extracted data), plus the report and the CSV. */
export async function downloadZip(docs: DocState[]) {
  const { zipSync, strToU8 } = await import('fflate');
  const files: Record<string, Uint8Array | [Uint8Array, { level: 0 }]> = {};
  const bytes = async (b: Blob) => new Uint8Array(await b.arrayBuffer());
  for (const t of DOC_TYPES) {
    const list = docs.filter((d) => d.doc.docType === t);
    if (!list.length) continue;
    const folder = slug(DOC_LABEL[t]);
    files[`${folder}/${folder}-scans.pdf`] = [await bytes(await scansPdf(list)), { level: 0 }]; // already compressed
    files[`${folder}/${folder}-data.json`] = strToU8(JSON.stringify(list.map(exportJson), null, 2));
  }
  files['formflow-report.pdf'] = [await bytes((await buildReport(docs)).output('blob')), { level: 0 }];
  files['formflow-all-fields.csv'] = strToU8(exportCsv(docs));
  files['README.txt'] = strToU8(
    `FormFlow export, ${today()}\n\nOne folder per document type. In each folder:\n  *-scans.pdf   the scanned pages of that type, one per page\n  *-data.json   what was read from them, with confidence and review status\n\nformflow-report.pdf     summary of every document and the cross-document checks\nformflow-all-fields.csv every field of every document in one table\n\nValues marked "review" were not confirmed by a person. Built on this device; nothing was uploaded.\n`,
  );
  const z = zipSync(files as any);
  save(new Blob([z as BlobPart], { type: 'application/zip' }), `formflow-export-${today()}.zip`);
}

const IMG = /\.(jpe?g|png|webp|gif|bmp)$/i;
const MIME: Record<string, string> = { jpg: 'image/jpeg', jpeg: 'image/jpeg', png: 'image/png', webp: 'image/webp', gif: 'image/gif', bmp: 'image/bmp' };
export const MAX_ZIP_PAGES = 80;

/** Unpack a ZIP of scans: every image and PDF inside, in name order. Other files are ignored. */
export async function unzipScans(file: Blob): Promise<{ blob: Blob; name: string; pdf: boolean }[]> {
  const { unzipSync } = await import('fflate');
  const entries = unzipSync(new Uint8Array(await file.arrayBuffer()), {
    // skip folders, hidden files and the macOS resource fork before decompressing them
    filter: (f) => !f.name.endsWith('/') && !/(^|\/)(\.|__MACOSX)/.test(f.name) && (IMG.test(f.name) || /\.pdf$/i.test(f.name)),
  });
  return Object.keys(entries).sort((a, b) => a.localeCompare(b, undefined, { numeric: true })).map((name) => {
    const base = name.split('/').pop()!;
    const ext = base.split('.').pop()!.toLowerCase();
    const pdf = ext === 'pdf';
    return { blob: new Blob([entries[name] as BlobPart], { type: pdf ? 'application/pdf' : MIME[ext] ?? 'image/jpeg' }), name: base, pdf };
  });
}
