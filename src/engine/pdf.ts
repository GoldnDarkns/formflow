// PDF in: render each page of a PDF to an image, in the browser (pdf.js).
import * as pdfjs from 'pdfjs-dist';
import workerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url';

pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;

export async function pdfToImages(file: Blob, name: string, maxPages = 30): Promise<{ blob: Blob; name: string }[]> {
  const pdf = await pdfjs.getDocument({ data: await file.arrayBuffer() }).promise;
  const out: { blob: Blob; name: string }[] = [];
  const base = name.replace(/\.pdf$/i, '');
  for (let i = 1; i <= Math.min(pdf.numPages, maxPages); i++) {
    const page = await pdf.getPage(i);
    const v0 = page.getViewport({ scale: 1 });
    const viewport = page.getViewport({ scale: 2000 / Math.max(v0.width, v0.height) });
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(viewport.width);
    canvas.height = Math.round(viewport.height);
    await page.render({ canvas, canvasContext: canvas.getContext('2d')!, viewport } as any).promise;
    const blob: Blob = await new Promise((r) => canvas.toBlob((b) => r(b!), 'image/jpeg', 0.92));
    out.push({ blob, name: pdf.numPages > 1 ? `${base}, page ${i}` : base });
  }
  return out;
}
