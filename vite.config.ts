import { defineConfig, defaultClientConditions } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';

// Relative base so the same build works on GitHub Pages (/repo/) or Cloudflare Pages (/).
export default defineConfig({
  base: './',
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['icon.svg', 'icon-192.png', 'icon-512.png'],
      manifest: {
        name: 'FormFlow — private document digitiser',
        short_name: 'FormFlow',
        description: 'Digitise scanned forms, passports, visas and licences on your device.',
        theme_color: '#24211d',
        background_color: '#f6f4ef',
        display: 'standalone',
        start_url: '.',
        scope: '.',
        icons: [
          { src: 'icon-192.png', sizes: '192x192', type: 'image/png' },
          { src: 'icon-512.png', sizes: '512x512', type: 'image/png' },
          { src: 'icon-512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' },
          { src: 'icon.svg', sizes: 'any', type: 'image/svg+xml' },
        ],
        share_target: undefined,
      },
      workbox: {
        // app shell + OCR engine + small ONNX models are precached for offline use
        globPatterns: ['**/*.{js,css,html,svg,png,jpg,onnx,gz,json,mjs,txt}'],
        // the Arabic reader (8 MB) is fetched the first time an identity document needs it, then kept
        globIgnores: ['**/ort/*.wasm', '**/models/ppocr/arabic_*'],
        maximumFileSizeToCacheInBytes: 30 * 1024 * 1024,
        runtimeCaching: [
          { urlPattern: /\/ort\/.*\.wasm$/, handler: 'CacheFirst', options: { cacheName: 'ort-wasm', expiration: { maxEntries: 4 } } },
          { urlPattern: /\/models\/ppocr\/arabic_/, handler: 'CacheFirst', options: { cacheName: 'arabic-reader', expiration: { maxEntries: 4 } } },
        ],
      },
    }),
  ],
  // use onnxruntime-web's external-wasm build: wasm is served from /ort/ (self-hosted, cached by the PWA)
  resolve: { conditions: ['onnxruntime-web-use-extern-wasm', ...defaultClientConditions] },
  worker: { format: 'es' },
  optimizeDeps: { exclude: ['onnxruntime-web', '@huggingface/transformers'] },
  build: { target: 'es2022', chunkSizeWarningLimit: 4000 },
});
