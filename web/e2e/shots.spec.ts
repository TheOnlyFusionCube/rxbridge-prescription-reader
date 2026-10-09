import { test } from "@playwright/test";
import path from "node:path";

const OUT = "/Users/faye/rxbridge/submission/screenshots";

test("capture submission screenshots", async ({ page }) => {
  await page.setViewportSize({ width: 430, height: 900 });
  await page.goto("/");
  await page.waitForTimeout(1400);
  await page.screenshot({ path: `${OUT}/01_upload.png` });

  await page.getByLabel("Speak in").selectOption("es");
  await page.waitForTimeout(500);

  await page.setInputFiles(
    'input[type="file"]',
    "/Users/faye/rxbridge/fixtures_real/printed_full.jpg",
  );
  await page.getByRole("button", { name: "Read my prescription" }).click();
  await page.locator(".day-grid").waitFor({ timeout: 120_000 });
  await page.waitForTimeout(2200);

  await page.screenshot({ path: `${OUT}/02_result_es.png` });

  await page.getByRole("button", { name: "Read aloud" }).click();
  await page.waitForTimeout(1200);
  await page.screenshot({ path: `${OUT}/03_read_aloud.png` });
});
