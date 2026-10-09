import { defineConfig, devices } from "@playwright/test";
export default defineConfig({
  testDir: "./e2e",
  testMatch: /dbg\.spec\.ts/,
  timeout: 180_000,
  use: { baseURL: "http://127.0.0.1:3000", headless: true },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
