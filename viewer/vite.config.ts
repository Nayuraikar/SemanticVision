import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import path from "node:path";

const nm = (p: string) => path.resolve(__dirname, "node_modules", p);

export default defineConfig({
  plugins: [react()],
  // adapters/threejs lives outside viewer/, so its bare imports must resolve here
  resolve: { alias: { three: nm("three"), react: nm("react"), "react-dom": nm("react-dom") } },
  server: { port: 5173, fs: { allow: [".."] } },
  test: { include: ["../adapters/threejs/**/*.test.ts", "src/**/*.test.ts"], environment: "node" },
});
