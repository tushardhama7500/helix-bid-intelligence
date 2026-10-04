import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    // Bind-mounted files on Docker Desktop (Windows/macOS) do not emit fs events.
    watch: { usePolling: true },
  },
});
