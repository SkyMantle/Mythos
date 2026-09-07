from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LiveVideoMetrics(BaseModel):
    fps: float | None = None
    locked: bool | None = None
    standard: str | None = None
    line_rate: float | None = None
    lines: int | None = None
    afc_hz: float | None = None
    freq_err_hz: float | None = None
    afc_pegged: bool | None = None
    hunt_span_hz: float | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
    clip_frac: float | None = None
    overflows: int | None = None


class LiveDetection(BaseModel):
    freq_hz: float
    snr_db: float | None = None
    bandwidth_hz: float | None = None
    standard: str | None = None
    confidence: float | None = None
    channel: str | None = None
    band: str | None = None
    pic_locked: bool | None = None


class LiveSpectrum(BaseModel):
    bins: list[float] | None = None
    center_hz: float | None = None
    bw_hz: float | None = None
    span_hz: float | None = None
    cursor_hz: float | None = None
    floor_db: float | None = None
    peak_db: float | None = None
    nfft: int | None = None
    rate_hz: float | None = None
    t_mono_ms: int | None = None
    next_hz: float | None = None
    dwell_ms: float | None = None
    afc_pegged: bool | None = None
    freq_err_hz: float | None = None
    afc_hz: float | None = None
    hunt_span_hz: float | None = None


class LiveGrid(BaseModel):
    pass_index: int = 0
    pass_count: int = 0
    current_hz: float | None = None
    start_hz: float | None = None
    stop_hz: float | None = None
    step_hz: float | None = None
    progress_01: float = 0.0
    visiting_hz: list[float] = Field(default_factory=list)
    passes_done: int = 0
    t_mono_ms: int | None = None
    next_hz: float | None = None
    dwell_ms: float | None = None


class LiveContextResponse(BaseModel):
    mode: str
    frequency_hz: float | None = None
    lock_state: bool
    lock_target_hz: float | None = None
    auto: bool
    source: str
    recording: bool
    video_metrics: LiveVideoMetrics
    parameters: dict[str, Any] = Field(default_factory=dict)
    detections: list[LiveDetection] = Field(default_factory=list)
    last_frame_ref: str | None = None
    rssi: float | None = None
    snr: float | None = None
    spectrum: LiveSpectrum = Field(default_factory=LiveSpectrum)
    grid: LiveGrid = Field(default_factory=LiveGrid)
    afc_pegged: bool | None = None
    freq_err_hz: float | None = None
    afc_hz: float | None = None
    hunt_span_hz: float | None = None
