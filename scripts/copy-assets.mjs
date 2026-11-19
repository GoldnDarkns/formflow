// Copy runtime assets (OCR engine, language data, ONNX Runtime wasm) into
// public/ so the app is fully self-hosted and works offline as a PWA.
import fs from 'node:fs';
import path from 'node:path';

const root = path.resolve(import.meta.dirname, '..');
const nm = (...p) => path.join(root, 'node_modules', ...p);
const pub = (...p) => path.join(root, 'public', ...p);
const copies = [
  [nm('tesseract.js/dist/worker.min.js'), pub('tesseract/worker.min.js')],
  ...['tesseract-core-lstm.wasm.js', 'tesseract-core-simd-lstm.wasm.js', 'tesseract-core-relaxedsimd-lstm.wasm.js'].map((f) => [nm('tesseract.js-core', f), pub('tesseract/core', f)]),
  [nm('@tesseract.js-data/eng/4.0.0_best_int/eng.traineddata.gz'), pub('tesseract/lang/eng.traineddata.gz')],
  ...['ort-wasm-simd-threaded.asyncify.wasm', 'ort-wasm-simd-threaded.asyncify.mjs'].map((f) => [nm('onnxruntime-web/dist', f), pub('ort', f)]),
];
for (const [src, dst] of copies) {
  fs.mkdirSync(path.dirname(dst), { recursive: true });
  if (!fs.existsSync(dst) || fs.statSync(dst).size !== fs.statSync(src).size) fs.copyFileSync(src, dst);
}
console.log('assets ready');
