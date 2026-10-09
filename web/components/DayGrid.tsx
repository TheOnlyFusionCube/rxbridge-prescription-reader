"use client";

import type { DrugEntry, TimesOfDay } from "@/lib/schema";
import { TIME_SLOTS } from "@/lib/schema";

function PillIcon({ count }: { count: number }) {
  return (
    <span className="pill-count" aria-label={`${count} pills`}>
      <span className="pill-glyph" aria-hidden="true" />
      <span className="pill-num">{count}</span>
    </span>
  );
}

function SlotCell({
  slot,
  drug,
  count,
}: {
  slot: { key: TimesOfDay; label: string };
  drug: string;
  count: number;
}) {
  if (count === 0) {
    return (
      <div className="cell cell-empty" data-slot={slot.key} data-drug={drug}>
        <span className="cell-label">{slot.label}</span>
        <span className="cell-none">--</span>
      </div>
    );
  }
  return (
    <div className="cell" data-slot={slot.key} data-drug={drug}>
      <span className="cell-label">{slot.label}</span>
      <PillIcon count={count} />
    </div>
  );
}

export default function DayGrid({ drugs }: { drugs: DrugEntry[] }) {
  if (drugs.length === 0) {
    return (
      <p className="grid-empty" role="status">
        No medicines found in this prescription.
      </p>
    );
  }

  return (
    <div className="day-grid" role="table" aria-label="Medication schedule by time of day">
      <div className="grid-head" role="row">
        <span className="grid-corner" role="columnheader">
          Medicine
        </span>
        {TIME_SLOTS.map((s) => (
          <span className="grid-head-slot" role="columnheader" key={s.key}>
            {s.label}
          </span>
        ))}
      </div>
      {drugs.map((d) => {
        const doseCount = Number.parseInt(d.dose, 10);
        const perDose = Number.isFinite(doseCount) && doseCount > 0 ? doseCount : 1;
        return (
          <div className="grid-row" role="row" key={d.drug}>
            <span className="grid-drug" role="rowheader">
              <span className="drug-name">{d.drug || "Unnamed medicine"}</span>
              <span className="drug-dose">
                {[d.dose, d.dose_unit].filter(Boolean).join(" ")}
                {d.frequency ? ` - ${d.frequency}` : ""}
                {d.duration_days ? ` - ${d.duration_days} days` : ""}
              </span>
            </span>
            {TIME_SLOTS.map((s) => (
              <SlotCell
                slot={s}
                drug={d.drug}
                count={d.times_of_day.includes(s.key) ? perDose : 0}
              />
            ))}
          </div>
        );
      })}
    </div>
  );
}
