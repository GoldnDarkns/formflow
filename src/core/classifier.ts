// Wraps the two ONNX classifiers for either onnxruntime-web (browser) or
// onnxruntime-node (scripts). Both expose the same InferenceSession / Tensor API.
import type { Classifiers } from './pipeline';

interface OrtLike {
  InferenceSession: { create(src: any, opts?: any): Promise<any> };
  Tensor: new (type: 'float32', data: Float32Array, dims: number[]) => any;
}

export async function loadClassifiers(ort: OrtLike, entitySrc: any, linkSrc: any, opts?: any): Promise<Classifiers> {
  const [es, ls] = await Promise.all([ort.InferenceSession.create(entitySrc, opts), ort.InferenceSession.create(linkSrc, opts)]);
  const run = async (s: any, X: number[][]): Promise<number[][]> => {
    if (!X.length) return [];
    const d = X[0].length;
    const flat = new Float32Array(X.length * d);
    X.forEach((r, i) => flat.set(r, i * d));
    const out = await s.run({ [s.inputNames[0]]: new ort.Tensor('float32', flat, [X.length, d]) });
    const p = out[s.outputNames[1]].data as Float32Array;
    const k = p.length / X.length;
    return X.map((_, i) => Array.from(p.subarray(i * k, i * k + k)));
  };
  return {
    entity: (X) => run(es, X),
    link: async (X) => (await run(ls, X)).map((r) => r[1]),
  };
}
