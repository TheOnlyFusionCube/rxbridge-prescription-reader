import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RxBridge - Hear your prescription in your language",
  description:
    "Photograph a prescription and hear your medicines, doses, and times spoken in your own language.",
};

export const viewport: Viewport = {
  themeColor: "#0b3d62",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
