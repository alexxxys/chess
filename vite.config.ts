import { defineConfig } from 'vite';

export default defineConfig({
  // Allow onnxruntime-web WASM files to be served and used by Web Workers
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  worker: {
    format: 'es',
  },
  server: {
    headers: {
      // Required for SharedArrayBuffer (used by ONNX multi-thread backend)
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  // Copy ONNX WASM files to public directory at build time
  assetsInclude: ['**/*.onnx', '**/*.wasm'],
});
