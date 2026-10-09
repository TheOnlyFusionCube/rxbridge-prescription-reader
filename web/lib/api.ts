import type { ExtractResponse } from "./schema";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export const SAMPLE: ExtractResponse = {
  status: "ok",
  language: "en",
  target_language: "es",
  explanation: null,
  overall_confidence: 0.88,
  warnings: [],
  tagger: "sample",
  drugs: [
    {
      drug: "Amoxicillin",
      dose: "500",
      dose_unit: "mg",
      route: "oral",
      frequency: "twice daily",
      times_of_day: ["morning", "night"],
      duration_days: 10,
      confidence: 0.94,
      source_text: "Amoxicillin 500 mg BID x 10 days",
      span: [0, 32],
    },
    {
      drug: "Paracetamol",
      dose: "1000",
      dose_unit: "mg",
      route: null,
      frequency: "every 8 hours",
      times_of_day: ["morning", "evening", "night"],
      duration_days: 5,
      confidence: 0.9,
      source_text: "Paracetamol 1000 mg q8h x 5d",
      span: [33, 63],
    },
  ],
  lines: [
    { text: "Amoxicillin 500 mg", confidence: 0.98 },
    { text: "Sig: Take one capsule twice daily", confidence: 0.95 },
    { text: "Paracetamol 1000 mg", confidence: 0.97 },
    { text: "Sig: Take one tablet every 8 hours", confidence: 0.93 },
  ],
};

export function apiBase(): string {
  return API_BASE;
}

export function isConfigured(): boolean {
  return Boolean(process.env.NEXT_PUBLIC_API_BASE);
}

export async function extractImage(
  file: File,
  targetLanguage: string,
): Promise<ExtractResponse> {
  if (!isConfigured()) {
    await new Promise((r) => setTimeout(r, 700));
    return { ...SAMPLE, target_language: targetLanguage };
  }

  const form = new FormData();
  form.append("image", file);
  const res = await fetch(`${API_BASE}/extract`, {
    method: "POST",
    body: form,
  });
  const body = (await res.json()) as ExtractResponse & { detail?: string };
  if (!res.ok) {
    if (body.status === "unreadable") {
      return body;
    }
    throw new Error(body.detail ?? "Request failed");
  }
  return body;
}
