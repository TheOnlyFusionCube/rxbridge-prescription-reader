import { test, expect } from "@playwright/test";

test("upload a prescription and get a spoken pictogram schedule", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "RxBridge" })).toBeVisible();

  await page.setInputFiles(
    'input[type="file"]',
    "/Users/faye/rxbridge/fixtures_real/printed_full.jpg",
  );
  await page.getByRole("button", { name: "Read my prescription" }).click();

  await expect(page.locator(".day-grid")).toBeVisible({ timeout: 120_000 });

  const grid = page.locator(".day-grid");
  await expect(grid.getByText("Amoxicillin", { exact: false })).toBeVisible({
    timeout: 30_000,
  });
  await expect(grid.getByText("Paracetamol", { exact: false })).toBeVisible();

  const morningCell = grid.locator('.cell[data-slot="morning"]').first();
  await expect(morningCell).toContainText("500", { timeout: 30_000 });

  const numerals = grid.locator(".pill-num");
  expect(await numerals.count()).toBeGreaterThan(0);
});
