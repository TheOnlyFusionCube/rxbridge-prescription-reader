"use client";

import { useCallback, useMemo, useRef, useState } from "react";
import DayGrid from "@/components/DayGrid";
import { extractImage } from "@/lib/api";
import { speak, speechSupported } from "@/lib/speak";
import { LANGUAGES, TIME_SLOTS, type ExtractResponse } from "@/lib/schema";

function spokenSchedule(schedule: ExtractResponse): string {
  if (schedule.status !== "ok" || schedule.drugs.length === 0) {
    return "I could not read this prescription. Please try again with a clearer photo.";
  }
  return schedule.drugs
    .map((d) => {
      const slots = TIME_SLOTS.filter((s) => d.times_of_day.includes(s.key));
      const slotText = slots.length ? ` at ${slots.map((s) => s.label).join(", ")}` : "";
      const amount = [d.dose, d.dose_unit].filter(Boolean).join(" ");
      const forText = d.duration_days ? ` for ${d.duration_days} days` : "";
      return `Take ${amount} of ${d.drug}${slotText}${forText}.`;
    })
    .join(" ");
}

export default function Page() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<ExtractResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [spoken, setSpoken] = useState(false);
  const [language, setLanguage] = useState("en");
  const inputRef = useRef<HTMLInputElement>(null);

  const canSpeak = useMemo(() => speechSupported(), []);

  const onFile = useCallback((f: File | null) => {
    setFile(f);
    setResult(null);
    setError(null);
    setSpoken(false);
    if (preview) {
      URL.revokeObjectURL(preview);
    }
    setPreview(f ? URL.createObjectURL(f) : null);
  }, [preview]);

  const onSubmit = useCallback(async () => {
    if (!file) {
      setError("Choose a photo of your prescription first.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const schedule = await extractImage(file, language);
      setResult(schedule);
      if (schedule.status === "ok" && schedule.drugs.length > 0) {
        speak(spokenSchedule(schedule), schedule.target_language);
        setSpoken(speechSupported());
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  }, [file, language]);

  const onSpeak = useCallback(() => {
    if (!result) return;
    speak(spokenSchedule(result), result.target_language);
    setSpoken(speechSupported());
  }, [result]);

  return (
    <main className="app">
      <header className="app-head">
        <h1>RxBridge</h1>
        <p className="tagline">Photograph your prescription. Hear it in your language.</p>
      </header>

      <section className="panel" aria-label="Upload a prescription">
        <label className="lang">
          <span>Speak in</span>
          <select value={language} onChange={(e) => setLanguage(e.target.value)}>
            {LANGUAGES.map((l) => (
              <option key={l.code} value={l.code}>
                {l.label}
              </option>
            ))}
          </select>
        </label>

        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          capture="environment"
          className="file-input"
          aria-label="Prescription photo"
          onChange={(e) => onFile(e.target.files?.[0] ?? null)}
        />

        <button
          type="button"
          className="btn btn-primary"
          disabled={!file || loading}
          onClick={onSubmit}
        >
          {loading ? "Reading..." : "Read my prescription"}
        </button>

        {preview && (
          <div className="preview">
            <img src={preview} alt="Your prescription photo" />
          </div>
        )}

        {error && (
          <p className="alert" role="alert">
            {error}
          </p>
        )}
      </section>

      {result && (
        <section className="panel result" aria-label="Your medicine schedule">
          {result.status === "ok" && result.drugs.length > 0 ? (
            <>
              <DayGrid drugs={result.drugs} />
              <div className="speak-row">
                <button type="button" className="btn btn-speak" onClick={onSpeak}>
                  Read aloud
                </button>
                <span className="speak-state" role="status" aria-live="polite">
                  {canSpeak
                    ? spoken
                      ? "Reading the schedule aloud"
                      : "Tap to hear it again"
                    : "Your browser cannot read text aloud"}
                </span>
              </div>
            </>
          ) : (
            <p className="alert" role="alert">
              We could not read this photo. Try again in better light, with the whole
              prescription flat in the frame.
            </p>
          )}

          {result.lines && result.lines.length > 0 && (
            <details className="what-we-read">
              <summary>What we read from the photo</summary>
              <ul>
                {result.lines.map((l, i) => (
                  <li key={`${l.text}-${i}`}>
                    <span>{l.text}</span>
                    <em>{Math.round(l.confidence * 100)}% sure</em>
                  </li>
                ))}
              </ul>
            </details>
          )}
        </section>
      )}

      <footer className="app-foot">
        <p>
          RxBridge reads prescriptions; it does not give medical advice. Always follow
          your doctor and pharmacist.
        </p>
      </footer>
    </main>
  );
}
