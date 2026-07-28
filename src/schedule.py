"""Patient medication schedule helpers (days + time slots)."""

from __future__ import annotations

import html
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SLOTS = ["Morning", "Noon", "Evening", "Night"]

DAY_LABELS = {
    "Mon": "Mon",
    "Tue": "Tue",
    "Wed": "Wed",
    "Thu": "Thu",
    "Fri": "Fri",
    "Sat": "Sat",
    "Sun": "Sun",
}


def week_monday(ref: Optional[date] = None) -> date:
    """Return Monday of the week containing ``ref`` (defaults to today)."""
    today = ref or date.today()
    return today - timedelta(days=today.weekday())  # Monday=0


def week_dates(ref: Optional[date] = None) -> List[Tuple[str, date]]:
    """Return [(day_key, date), ...] for Mon–Sun of the week."""
    start = week_monday(ref)
    return [(DAYS[i], start + timedelta(days=i)) for i in range(7)]


def format_week_range(ref: Optional[date] = None) -> str:
    """Human-readable week span, e.g. 'Mon 28 Jul – Sun 3 Aug 2026'."""
    days = week_dates(ref)
    start = days[0][1]
    end = days[-1][1]

    def _short(d: date, with_year: bool = False) -> str:
        base = f"{d.strftime('%a')} {d.day} {d.strftime('%b')}"
        return f"{base} {d.year}" if with_year else base

    if start.month == end.month and start.year == end.year:
        return f"{start.strftime('%a')} {start.day} – {end.strftime('%a')} {end.day} {end.strftime('%b %Y')}"
    if start.year == end.year:
        return f"{_short(start)} – {_short(end)} {end.year}"
    return f"{_short(start, True)} – {_short(end, True)}"


def _fmt_day_header(day_key: str, d: date, today: date) -> str:
    is_today = d == today
    cls = "sched-day-head is-today" if is_today else "sched-day-head"
    # %-d is POSIX; on Windows use %#d — use day without zero-pad portably
    day_num = str(d.day)
    month = d.strftime("%b")
    today_tag = '<span class="sched-today-tag">Today</span>' if is_today else ""
    return (
        f'<th scope="col" class="{cls}">'
        f'<div class="sched-day-name">{html.escape(DAY_LABELS[day_key])}</div>'
        f'<div class="sched-day-date">{html.escape(day_num)} {html.escape(month)}</div>'
        f"{today_tag}"
        f"</th>"
    )


def normalize_schedule_entry(
    medicine: str,
    days: Sequence[str],
    slots: Sequence[str],
    note: str = "",
) -> Optional[Dict[str, Any]]:
    """Build a validated schedule entry, or None if incomplete."""
    med = (medicine or "").strip()
    if not med:
        return None
    clean_days = [d for d in DAYS if d in set(days or [])]
    clean_slots = [s for s in SLOTS if s in set(slots or [])]
    if not clean_days or not clean_slots:
        return None
    return {
        "medicine": med,
        "days": clean_days,
        "slots": clean_slots,
        "note": (note or "").strip()[:120],
    }


def entry_summary(entry: Dict[str, Any]) -> str:
    days = ", ".join(entry.get("days") or [])
    slots = ", ".join(entry.get("slots") or [])
    note = entry.get("note") or ""
    base = f"{days} · {slots}"
    return f"{base} — {note}" if note else base


def schedule_count_label(schedule: List[Dict[str, Any]]) -> str:
    n = len(schedule or [])
    if n == 0:
        return "No medicines scheduled yet"
    if n == 1:
        return "1 medicine on your schedule"
    return f"{n} medicines on your schedule"


def build_weekly_grid_html(
    schedule: List[Dict[str, Any]],
    ref: Optional[date] = None,
) -> str:
    """
    Render a Mon–Sun × Morning–Night overview grid with calendar dates.

    Uses the week containing ``ref`` (defaults to today).
    """
    today = date.today()
    dated_days = week_dates(ref)
    range_label = format_week_range(ref)

    # cell[(day, slot)] -> ordered unique medicine names
    cell: Dict[tuple, List[str]] = {}
    for entry in schedule or []:
        med = html.escape(str(entry.get("medicine") or ""))
        if not med:
            continue
        for day in entry.get("days") or []:
            if day not in DAYS:
                continue
            for slot in entry.get("slots") or []:
                if slot not in SLOTS:
                    continue
                key = (day, slot)
                names = cell.setdefault(key, [])
                if med not in names:
                    names.append(med)

    head = "".join(_fmt_day_header(day_key, d, today) for day_key, d in dated_days)
    rows = []
    for slot in SLOTS:
        cells = [f'<th scope="row" class="sched-slot">{html.escape(slot)}</th>']
        for day_key, d in dated_days:
            meds = cell.get((day_key, slot), [])
            today_cls = " is-today" if d == today else ""
            if meds:
                pills = "".join(f'<span class="sched-pill">{m}</span>' for m in meds)
                cells.append(f'<td class="sched-cell{today_cls}">{pills}</td>')
            else:
                cells.append(f'<td class="sched-cell sched-empty{today_cls}"></td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return (
        '<div class="sched-wrap">'
        f'<div class="sched-week-label">Week of {html.escape(range_label)}</div>'
        '<table class="sched-grid" role="grid" aria-label="Weekly medicine schedule">'
        f'<thead><tr><th scope="col" class="sched-corner"></th>{head}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )
