export type TimesOfDay = "morning" | "noon" | "evening" | "night";

export type Status = "ok" | "unreadable";

export interface DrugEntry {
  drug: string;
  dose: string;
  dose_unit: string;
  route: string | null;
  frequency: string;
  times_of_day: TimesOfDay[];
  duration_days: number | null;
  confidence: number;
  source_text: string;
  span: [number, number] | null;
}

export interface PrescriptionSchedule {
  status: Status;
  drugs: DrugEntry[];
  language: string;
  target_language: string;
  explanation: string | null;
  overall_confidence: number | null;
  warnings: string[];
}

export interface OcrLine {
  text: string;
  confidence: number;
}

export interface ExtractResponse extends PrescriptionSchedule {
  lines?: OcrLine[];
  tagger?: string;
}

export const TIME_SLOTS: { key: TimesOfDay; label: string }[] = [
  { key: "morning", label: "Morning" },
  { key: "noon", label: "Midday" },
  { key: "evening", label: "Evening" },
  { key: "night", label: "Night" },
];

export const LANGUAGES: { code: string; label: string }[] = [
  { code: "en", label: "English" },
  { code: "es", label: "Espanol" },
  { code: "fr", label: "Francais" },
  { code: "hi", label: "Hindi" },
  { code: "ar", label: "Arabic" },
  { code: "zh", label: "Chinese" },
];
