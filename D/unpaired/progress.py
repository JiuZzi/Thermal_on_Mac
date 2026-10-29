"""Plain terminal summaries; full training numbers remain in loss.jsonl."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
import traceback


def duration(seconds):
    if seconds is None or not math.isfinite(seconds):
        return "--"
    seconds = max(0, round(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{seconds:02d}s" if minutes else f"{seconds}s"


def number(value):
    return "--" if value is None else f"{value:.4g}"


def event(tag, message):
    print(f"[{tag:<7}] {message}", flush=True)


def cuda_memory_snapshot(device=None):
    """Post-update allocator stats plus a current device-wide free-memory sample."""
    import torch
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info(device)
    scale = 1024 ** 3
    return dict(allocated_gib=torch.cuda.memory_allocated(device) / scale,
                reserved_gib=torch.cuda.memory_reserved(device) / scale,
                peak_allocated_gib=torch.cuda.max_memory_allocated(device) / scale,
                peak_reserved_gib=torch.cuda.max_memory_reserved(device) / scale,
                device_free_gib=free / scale, device_total_gib=total / scale)


class TrainingTimer:
    """Count only updates performed in this invocation, including elapsed I/O."""
    def __init__(self, start_step, total, *, clock=time.perf_counter):
        self.start_step, self.total, self.clock = start_step, total, clock
        self.started = clock()

    def snapshot(self, step):
        if not self.start_step <= step <= self.total:
            raise ValueError("Progress outside this training session")
        elapsed = max(0., self.clock() - self.started)
        updates = step - self.start_step
        average = elapsed / updates if updates else None
        remaining = self.total - step
        return dict(elapsed=elapsed, session_start_step=self.start_step, session_updates=updates,
                    progress_percent=100 * step / self.total,
                    average_update_seconds=average,
                    eta_seconds=remaining * average if average is not None else (0. if not remaining else None))


def metric_lines(row):
    """Latest logged step's losses, not an undocumented interval average."""
    step, total = row["step"], row["total_steps"]
    width = len(str(total))
    speed = row.get("average_update_seconds")
    speed = f"{speed:.2f}s/update" if speed is not None else "--s/update"
    memory = f"peak {number(row.get('peak_allocated_gib'))}GiB"
    if "reserved_gib" in row and "device_free_gib" in row:
        memory = ("VRAM peak/resv/free=" + "/".join(number(row.get(key)) for key in
                  ("peak_allocated_gib", "reserved_gib", "device_free_gib")) + "GiB")
    progress = (f"[TRAIN  ] {step:0{width}d}/{total} ({row['progress_percent']:.1f}%) | {speed}"
                f" | elapsed {duration(row['elapsed'])} | ETA ~{duration(row.get('eta_seconds'))}"
                f" | {memory}")
    main = (f"[LOSS   ] G={number(row.get('G_total'))}"
            f" | D_RGB={number(row.get('D_RGB'))} D_TIR={number(row.get('D_TIR'))}"
            f" | cycle_TIR={number(row.get('cycle_tir'))} cycle_RGB={number(row.get('cycle_rgb'))}")
    auxiliary = [f"{label}={number(row[key])}" for key, label in
                 (("edge", "edge"), ("content", "content"), ("semantic", "semantic"),
                  ("color_prior", "color"), ("chroma_smooth", "smooth")) if key in row]
    if "color_prior" in row:
        auxiliary.append(f"color_w={number(row.get('color_weight'))}")
    replay = number(row.get("rgb_replay")) if row.get("rgb_replay_active") else "off"
    auxiliary.append(f"replay={replay}")
    return [progress, main, "[AUX    ] " + " ".join(auxiliary)]


class TrainingLog:
    """Readable terminal + UTF-8 train.log; unabridged numeric JSONL."""
    def __init__(self, folder, *, details=False):
        self.folder, self.details = Path(folder), details

    def __enter__(self):
        self.handle = (self.folder / "train.log").open("a", encoding="utf-8")
        self.event("SESSION", datetime.now().astimezone().isoformat(timespec="seconds"))
        return self

    def emit(self, text):
        print(text, flush=True)
        self.handle.write(text + "\n"); self.handle.flush()

    def event(self, tag, message):
        self.emit(f"[{tag:<7}] {message}")

    def metrics(self, row):
        record = dict(row, logged_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        payload = json.dumps(record, ensure_ascii=True, allow_nan=False)
        with (self.folder / "loss.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
        self.emit("\n".join(metric_lines(record)))
        if self.details:
            self.event("DETAIL", payload)

    def __exit__(self, kind, value, tb):
        try:
            if kind is not None:
                tag = "STOP" if issubclass(kind, KeyboardInterrupt) else "ERROR"
                try:
                    self.event(tag, f"{kind.__name__}: {value}")
                    self.handle.write("".join(traceback.format_exception(kind, value, tb)))
                except OSError:
                    pass  # A full disk/broken pipe must not replace the original error.
        finally:
            try:
                self.handle.close()
            except OSError:
                if kind is None:
                    raise
        return False  # Keep the original exception/traceback visible.


class ItemProgress:
    """Sparse progress for preparation/preview, plus a 30-second heartbeat."""
    def __init__(self, total, tag, *, start=0, emit=event, clock=time.perf_counter):
        self.total, self.tag, self.start, self.emit, self.clock = total, tag, start, emit, clock
        self.started = self.last_time = clock()
        self.last_count = start
        self.every = max(1, math.ceil((total-start) / 10))

    def update(self, count, detail=""):
        now = self.clock()
        if (count != self.start + 1 and count != self.total and
            count - self.last_count < self.every and now - self.last_time < 30):
            return
        elapsed = max(0., now - self.started)
        completed = count - self.start
        eta = elapsed / completed * (self.total-count) if completed > 0 else None
        suffix = f" | {detail}" if detail else ""
        self.emit(self.tag, f"{count}/{self.total} ({100*count/self.total:.1f}%)"
                  f" | elapsed {duration(elapsed)} | ETA ~{duration(eta)}{suffix}")
        self.last_count, self.last_time = count, now
