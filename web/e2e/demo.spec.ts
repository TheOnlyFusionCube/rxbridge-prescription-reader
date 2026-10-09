import { test } from "@playwright/test";
import path from "node:path";

test("record RxBridge demo", async ({ page }) => {
  await page.setViewportSize({ width: 420, height: 860 });
  await page.goto("/");

  await page.waitForTimeout(1200);
  await page.getByLabel("Speak in").selectOption("es");
  await page.waitForTimeout(900);

  await page.setInputFiles(
    'input[type="file"]',
    path.resolve("/Users/faye/rxbridge/fixtures_real/printed_full.jpg"),
  );
  await page.waitForTimeout(900);
  await page.getByRole("button", { name: "Read my prescription" }).click();

  await page.locator(".day-grid").waitFor({ timeout: 120_000 });
  await page.waitForTimeout(2500);

  await page.getByRole("button", { name: "Read aloud" }).click();
  await page.waitForTimeout(2000);

  await page.getByRole("group").waitFor({ state: "visible", timeout: 5000 });
  await page.waitForTimeout(1500);
});
