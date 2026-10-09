let cached: SpeechSynthesisVoice[] | null = null;

export function speak(text: string, lang: string): void {
  if (typeof window === "undefined" || !("speechSynthesis" in window)) {
    return;
  }
  try {
    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(text);
    utter.lang = lang;
    utter.rate = 0.85;
    utter.pitch = 1;
    const voice = window.speechSynthesis
      .getVoices()
      .find((v) => v.lang?.toLowerCase().startsWith(lang.slice(0, 2)));
    if (voice) {
      utter.voice = voice;
    }
    window.speechSynthesis.speak(utter);
  } catch {
    return;
  }
}

export function cancelSpeech(): void {
  if (typeof window === "undefined" || !("speechSynthesis" in window)) {
    return;
  }
  try {
    window.speechSynthesis.cancel();
  } catch {
    return;
  }
}

export function speechSupported(): boolean {
  return typeof window !== "undefined" && "speechSynthesis" in window;
}

export function warmVoices(): void {
  if (typeof window === "undefined" || !("speechSynthesis" in window)) {
    return;
  }
  if (cached) {
    return;
  }
  const voices = window.speechSynthesis.getVoices();
  if (voices.length > 0) {
    cached = voices;
  }
}
