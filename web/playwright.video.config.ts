import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  testMatch: /demo\.spec\.ts/,
  timeout: 180_000,
  use: {
    baseURL: "http://127.0.0.1:3000",
    headless: true,
    video: { mode: "on", size: { width: 420, height: 860 } },
    viewport: { width: 420, height: 860 },
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
