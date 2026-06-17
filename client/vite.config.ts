import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@tensorflow/tfjs-tflite': '@tensorflow/tfjs-tflite/dist/tf-tflite.min.js'
    }
  },
  test: {
    environment: 'happy-dom',
    setupFiles: ['./src/setupTests.ts'],
    globals: true,
  }
})
