from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hpcopt.ingest.swf import IngestResult
from hpcopt.utils.io import ensure_dir, write_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Watermark persistence
# ---------------------------------------------------------------------------

DEFAULT_WATERMARK_PATH = Path("outputs/ingest/shadow_watermark.json")


@dataclass
class WatermarkState:
    """Tracks the last-processed timestamp for incremental ingestion."""

    last_processed_ts: int | None = None
    last_poll_utc: str | None = None
    rows_ingested_total: int = 0
    poll_count: int = 0
    source_type: str = ""
    source_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_processed_ts": self.last_processed_ts,
            "last_poll_utc": self.last_poll_utc,
            "rows_ingested_total": self.rows_ingested_total,
            "poll_count": self.poll_count,
            "source_type": self.source_type,
            "source_path": self.source_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WatermarkState:
        return cls(
            last_processed_ts=data.get("last_processed_ts"),
            last_poll_utc=data.get("last_poll_utc"),
            rows_ingested_total=int(data.get("rows_ingested_total", 0)),
            poll_count=int(data.get("poll_count", 0)),
            source_type=str(data.get("source_type", "")),
            source_path=str(data.get("source_path", "")),
        )


def _load_watermark(path: Path) -> WatermarkState:
    if not path.exists():
        return WatermarkState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return WatermarkState.from_dict(data)
    except (json.JSONDecodeError, OSError, KeyError) as exc:
        logger.warning("Could not load watermark from %s: %s; starting fresh.", path, exc)
        return WatermarkState()


def _save_watermark(path: Path, state: WatermarkState) -> None:
    ensure_dir(path.parent)
    write_json(path, state.to_dict())


# ---------------------------------------------------------------------------
# Poll result
# ---------------------------------------------------------------------------


@dataclass
class PollResult:
    """Outcome of a single poll cycle."""

    success: bool
    rows_ingested: int = 0
    ingest_result: IngestResult | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Shadow ingestion daemon
# ---------------------------------------------------------------------------


class ShadowIngestionDaemon:
    """Periodically polls a scheduler data source for incremental ingestion.

    This daemon is **read-only** -- it never modifies scheduler state.  It
    reads from either a ``sacct`` command output or a PBS accounting log path,
    filters to records newer than the persisted watermark, and writes
    incremental parquet files.

    Usage
    -----
    ::

        daemon = ShadowIngestionDaemon(
            out_dir=Path("outputs/datasets/shadow"),
            report_dir=Path("outputs/reports/shadow"),
            watermark_path=Path("outputs/ingest/shadow_watermark.json"),
        )
        daemon.start(
            interval_sec=300,
            source_type="slurm",
            source_path="/var/log/slurm/sacct_dump.txt",
        )
    """

    def __init__(
        self,
        out_dir: Path | str = Path("outputs/datasets/shadow"),
        report_dir: Path | str = Path("outputs/reports/shadow"),
        watermark_path: Path | str | None = None,
        dataset_id_prefix: str = "shadow",
    ) -> None:
        self._out_dir = Path(out_dir)
        self._report_dir = Path(report_dir)
        self._watermark_path = (
            Path(watermark_path) if watermark_path else DEFAULT_WATERMARK_PATH
        )
        self._dataset_id_prefix = dataset_id_prefix

        self._source_type: str = ""
        self._source_path: Path = Path(".")
        self._interval_sec: float = 300.0

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        ensure_dir(self._out_dir)
        ensure_dir(self._report_dir)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _generate_dataset_id(self, watermark: WatermarkState) -> str:
        return f"{self._dataset_id_prefix}_poll{watermark.poll_count + 1}"

    def _ingest_incremental(
        self,
        source_type: str,
        source_path: Path,
        watermark: WatermarkState,
    ) -> PollResult:
        """Run one incremental ingestion pass.

        Reads the source file, filters to rows with ``submit_ts`` strictly
        greater than the watermark, then delegates to the appropriate
        format-specific ingest function.
        """
        import datetime as dt

        import pandas as pd

        dataset_id = self._generate_dataset_id(watermark)

        try:
            if source_type == "slurm":
                from hpcopt.ingest.slurm import ingest_slurm

                result = ingest_slurm(
                    input_path=source_path,
                    out_dir=self._out_dir,
                    dataset_id=dataset_id,
                    report_dir=self._report_dir,
                )
            elif source_type == "pbs":
                from hpcopt.ingest.pbs import ingest_pbs

                result = ingest_pbs(
                    input_path=source_path,
                    out_dir=self._out_dir,
                    dataset_id=dataset_id,
                    report_dir=self._report_dir,
                )
            elif source_type == "swf":
                from hpcopt.ingest.swf import ingest_swf

                result = ingest_swf(
                    input_path=source_path,
                    out_dir=self._out_dir,
                    dataset_id=dataset_id,
                    report_dir=self._report_dir,
                )
            else:
                return PollResult(
                    success=False,
                    error=f"Unsupported source_type: {source_type}",
                )
        except Exception as exc:
            logger.error("Ingestion failed during poll: %s", exc)
            return PollResult(success=False, error=str(exc))

        # ----------------------------------------------------------
        # Filter to new rows only (post-watermark)
        # ----------------------------------------------------------
        try:
            df = pd.read_parquet(result.dataset_path)
        except Exception as exc:
            logger.error("Could not read ingested parquet: %s", exc)
            return PollResult(success=False, error=str(exc))

        if watermark.last_processed_ts is not None:
            original_len = len(df)
            df = df[
                pd.to_numeric(df["submit_ts"], errors="coerce")
                > watermark.last_processed_ts
            ]
            logger.info(
                "Filtered %d -> %d rows (watermark=%d)",
                original_len,
                len(df),
                watermark.last_processed_ts,
            )

        if df.empty:
            logger.info("No new rows after watermark filtering.")
            return PollResult(success=True, rows_ingested=0, ingest_result=result)

        # Overwrite the parquet with only the incremental rows.
        df.to_parquet(result.dataset_path, index=False)

        # Update watermark.
        max_ts = int(pd.to_numeric(df["submit_ts"], errors="coerce").max())
        watermark.last_processed_ts = max_ts
        watermark.last_poll_utc = dt.datetime.now(tz=dt.UTC).isoformat()
        watermark.rows_ingested_total += len(df)
        watermark.poll_count += 1
        watermark.source_type = source_type
        watermark.source_path = str(source_path)
        _save_watermark(self._watermark_path, watermark)

        return PollResult(
            success=True,
            rows_ingested=len(df),
            ingest_result=result,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def poll_once(self) -> PollResult:
        """Execute a single poll cycle.

        This is the primary entry point for callers that want manual control
        instead of the background loop.
        """
        with self._lock:
            watermark = _load_watermark(self._watermark_path)
            return self._ingest_incremental(
                source_type=self._source_type,
                source_path=self._source_path,
                watermark=watermark,
            )

    def start(
        self,
        interval_sec: float,
        source_type: str,
        source_path: str | Path,
        blocking: bool = False,
    ) -> None:
        """Start periodic polling.

        Parameters
        ----------
        interval_sec:
            Seconds between poll cycles.
        source_type:
            One of ``"slurm"``, ``"pbs"``, ``"swf"``.
        source_path:
            Path to the source file (e.g. sacct dump, PBS accounting log).
        blocking:
            If ``True``, run in the foreground (blocks the calling thread).
            If ``False`` (default), spawn a background daemon thread.
        """
        self._source_type = source_type
        self._source_path = Path(source_path)
        self._interval_sec = interval_sec
        self._stop_event.clear()

        logger.info(
            "Shadow ingestion starting: source_type=%s, source_path=%s, interval=%ds",
            source_type,
            source_path,
            interval_sec,
        )

        if blocking:
            self._run_loop()
        else:
            self._thread = threading.Thread(
                target=self._run_loop,
                name="shadow-ingestion-daemon",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """Signal the polling loop to stop."""
        logger.info("Shadow ingestion stop requested.")
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_sec + 5)
            self._thread = None

    def _run_loop(self) -> None:
        """Internal polling loop."""
        while not self._stop_event.is_set():
            try:
                result = self.poll_once()
                if result.success:
                    logger.info(
                        "Poll complete: %d new rows ingested.", result.rows_ingested
                    )
                else:
                    logger.warning("Poll failed: %s", result.error)
            except Exception as exc:
                logger.exception("Unexpected error during poll cycle: %s", exc)

            self._stop_event.wait(timeout=self._interval_sec)
