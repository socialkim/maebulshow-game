// Static build config — produces a self-contained bundle that runs on any static host
// (GitHub Pages, S3, a file server) with no dev server and no node_modules.
//
// The dev path (npx vite) still uses the browser importmap in index.html and loads
// three straight out of node_modules. This config only affects `npm run build`.
export default {
  base: './',                 // relative asset URLs so the build works under any sub-path
  build: {
    target: 'esnext',         // main.js uses top-level await
    outDir: 'dist',
    emptyOutDir: true,
    assetsInlineLimit: 0,
    chunkSizeWarningLimit: 4000,
  },
};
