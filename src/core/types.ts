// Shared types for the FormFlow pipeline. Pure data, no DOM, so the same code
// runs in the browser worker and in the Node training / evaluation scripts.

export type Box = [number, number, number, number]; // x0, y0, x1, y1 (pixels)

export interface OcrWord {
  text: string;
  box: Box;
  conf: number; // 0..100 from Tesseract
  line: number; // index of the OCR line the word belongs to
}

export type Label = 'question' | 'answer' | 'header' | 'other';
export const LABELS: Label[] = ['question', 'answer', 'header', 'other'];

export interface Entity {
  id: number;
  text: string;
  box: Box;
  words: OcrWord[];
  label?: Label;
  probs?: number[]; // aligned with LABELS
}

export interface Link {
  q: number; // question entity id
  a: number; // answer entity id
  p: number; // link probability
}

export interface GrayImage {
  width: number;
  height: number;
  data: Uint8Array; // 0 = black, 255 = white
}

export interface SecondReading {
  text: string;
  box: Box;
}

export type FieldStatus = 'ok' | 'review' | 'empty';

export interface Field {
  key: string;
  value: string;
  qId?: number;
  aId?: number;
  confidence: number; // 0..1, calibrated for forms: the estimated chance the field is right
  raw?: number; // the score before calibration
  status: FieldStatus;
  reasons: string[]; // why it needs review
  alt?: string; // second reader's version of the value, when it differs
  source: 'layout' | 'mrz' | 'template';
}
