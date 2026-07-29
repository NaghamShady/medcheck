"""Patient medication schedule helpers (daily routine + time anchors)."""

from __future__ import annotations

import html
import re
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Anchors tied to the patient's personal day
SLOTS = ["wake", "breakfast", "lunch", "dinner", "sleep"]

SLOT_LABELS = {
    "wake": "Wake up",
    "breakfast": "Breakfast",
    "lunch": "Lunch",
    "dinner": "Dinner",
    "sleep": "Bedtime / sleep",
}

DEFAULT_ROUTINE: Dict[str, str] = {
    "wake": "07:00",
    "breakfast": "08:00",
    "lunch": "13:00",
    "dinner": "19:00",
    "sleep": "22:30",
}

_TIME_RE = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")


def half_hour_options() -> List[str]:
    """HH:MM values every 30 minutes for a full day."""
    opts: List[str] = []
    for hour in range(24):
        for minute in (0, 30):
            opts.append(f"{hour:02d}:{minute:02d}")
    return opts


def normalize_time(value: Any) -> Optional[str]:
    """Accept HH:MM strings (or datetime.time) and return HH:MM, or None."""
    if value is None:
        return None
    if hasattr(value, "hour") and hasattr(value, "minute"):
        try:
            return f"{int(value.hour):02d}:{int(value.minute):02d}"
        except Exception:
            return None
    text = str(value).strip()
    if not text:
        return None
    m = _TIME_RE.match(text)
    if not m:
        return None
    return f"{int(m.group(1)):02d}:{m.group(2)}"


def format_time_12h(hhmm: str) -> str:
    """Format HH:MM as a short 12-hour label, e.g. 7:00 AM."""
    t = normalize_time(hhmm)
    if not t:
        return str(hhmm or "")
    hour, minute = map(int, t.split(":"))
    suffix = "AM" if hour < 12 else "PM"
    h12 = hour % 12 or 12
    return f"{h12}:{minute:02d} {suffix}"


def normalize_routine(raw: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Merge user routine with defaults; always returns all five slots."""
    base = dict(DEFAULT_ROUTINE)
    if not raw:
        return base
    for key in SLOTS:
        cleaned = normalize_time(raw.get(key))
        if cleaned:
            base[key] = cleaned
    return base


def routine_is_set(raw: Optional[Dict[str, Any]]) -> bool:
    """True when the patient has saved a full daily routine."""
    if not raw or not isinstance(raw, dict):
        return False
    return all(normalize_time(raw.get(k)) for k in SLOTS)


def slot_label(slot: str, routine: Optional[Dict[str, Any]] = None) -> str:
    """Human label with optional clock time, e.g. 'Breakfast · 8:00 AM'."""
    name = SLOT_LABELS.get(slot, slot.title())
    times = normalize_routine(routine)
    clock = times.get(slot)
    if clock:
        return f"{name} · {format_time_12h(clock)}"
    return name


def normalize_custom_times(values: Sequence[Any]) -> List[str]:
    """Deduplicate and sort valid custom HH:MM times."""
    seen = set()
    out: List[str] = []
    for value in values or []:
        cleaned = normalize_time(value)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            out.append(cleaned)
    out.sort()
    return out


def normalize_schedule_entry(
    medicine: str,
    slots: Sequence[str],
    custom_times: Optional[Sequence[Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Build a validated daily schedule entry, or None if incomplete."""
    med = (medicine or "").strip()
    if not med:
        return None
    clean_slots = [s for s in SLOTS if s in set(slots or [])]
    clean_custom = normalize_custom_times(custom_times or [])
    if not clean_slots and not clean_custom:
        return None
    return {
        "medicine": med,
        "slots": clean_slots,
        "custom_times": clean_custom,
    }


def entry_summary(
    entry: Dict[str, Any],
    routine: Optional[Dict[str, Any]] = None,
) -> str:
    parts: List[str] = []
    for s in entry.get("slots") or []:
        if s in SLOT_LABELS:
            parts.append(slot_label(s, routine))
    for t in normalize_custom_times(entry.get("custom_times") or []):
        parts.append(f"Custom · {format_time_12h(t)}")
    return " · ".join(parts) if parts else "No times set"


def schedule_count_label(schedule: List[Dict[str, Any]]) -> str:
    n = len(schedule or [])
    if n == 0:
        return "No medicines scheduled yet"
    if n == 1:
        return "1 medicine on your daily plan"
    return f"{n} medicines on your daily plan"


def _today_label_portable(ref: Optional[date] = None) -> str:
    d = ref or date.today()
    return f"{d.strftime('%A')}, {d.day} {d.strftime('%b %Y')}"


def _timeline_events(
    schedule: List[Dict[str, Any]],
    routine: Dict[str, str],
) -> List[Tuple[str, str, List[str]]]:
    """
    Build (hhmm, anchor_label, meds) rows sorted by clock time.

    Routine anchors always appear (even if empty). Custom times only appear
    when at least one medicine uses them.
    """
    # key = (hhmm, label) -> meds
    bucket: Dict[Tuple[str, str], List[str]] = {}
    order: List[Tuple[str, str]] = []

    def _add(hhmm: str, label: str, med: Optional[str] = None, force: bool = False) -> None:
        key = (hhmm, label)
        if key not in bucket:
            bucket[key] = []
            order.append(key)
        elif force and key not in order:
            order.append(key)
        if med and med not in bucket[key]:
            bucket[key].append(med)

    for slot in SLOTS:
        _add(routine[slot], SLOT_LABELS[slot], force=True)

    for entry in schedule or []:
        med = html.escape(str(entry.get("medicine") or ""))
        if not med:
            continue
        for slot in entry.get("slots") or []:
            if slot not in SLOT_LABELS:
                continue
            _add(routine[slot], SLOT_LABELS[slot], med)
        for t in normalize_custom_times(entry.get("custom_times") or []):
            _add(t, "Custom time", med)

    # Sort: by time, then routine anchors before custom at same minute
    def _sort_key(item: Tuple[str, str]) -> Tuple[str, int]:
        hhmm, label = item
        return (hhmm, 1 if label == "Custom time" else 0)

    ordered = sorted(order, key=_sort_key)
    return [(hhmm, label, bucket[(hhmm, label)]) for hhmm, label in ordered]


def build_daily_timeline_html(
    schedule: List[Dict[str, Any]],
    routine: Optional[Dict[str, Any]] = None,
    ref: Optional[date] = None,
) -> str:
    """
    Render a vertical daily timeline with routine anchors and any custom times.
    """
    times = normalize_routine(routine)
    day_label = _today_label_portable(ref)
    events = _timeline_events(schedule, times)

    rows = []
    for hhmm, label, meds in events:
        clock = html.escape(format_time_12h(hhmm))
        name = html.escape(label)
        if meds:
            pills = "".join(f'<span class="sched-pill">{m}</span>' for m in meds)
            body = f'<div class="sched-timeline-meds">{pills}</div>'
            empty_cls = ""
        else:
            body = '<div class="sched-timeline-empty">Nothing scheduled</div>'
            empty_cls = " is-empty"

        rows.append(
            f'<div class="sched-timeline-row{empty_cls}">'
            f'<div class="sched-timeline-time">'
            f'<div class="sched-timeline-clock">{clock}</div>'
            f'<div class="sched-timeline-anchor">{name}</div>'
            f"</div>"
            f'<div class="sched-timeline-dot" aria-hidden="true"></div>'
            f'<div class="sched-timeline-body">{body}</div>'
            f"</div>"
        )

    return (
        '<div class="sched-wrap">'
        f'<div class="sched-day-label">Today · {html.escape(day_label)}</div>'
        '<div class="sched-timeline" role="list" aria-label="Daily medicine schedule">'
        f"{''.join(rows)}"
        "</div>"
        "</div>"
    )
