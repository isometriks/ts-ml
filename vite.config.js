// vite.config.js
import { resolve } from 'path'
import { defineConfig } from 'vite'

export default defineConfig({
  base: '/ts-ml',
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        mnist: resolve(__dirname, 'mnist.html'),
      },
    },
  },
})
