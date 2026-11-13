# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Budgeted host K/V bank with metadata and pluggable retrieval into the attention window.

Used through :class:`~flashdreams.core.attention.kvcache.LayerwiseKVOffload`:
every chunk the model writes is also stored in the bank as one entry per chunk
or per latent frame, and each forward the offload fetches the entries chosen by
a :class:`KVRetrievalPolicy` into a ``retrieve`` region of the device slot. The
attended layout per layer becomes ``[sink | retrieved | recent window]``; the
``retrieved`` region sits at the tail of the cache's sink, so ``BlockKVCache``
never rolls it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import Tensor

if TYPE_CHECKING:
    from flashdreams.core.attention.kvcache import BlockKVCache


@dataclass(kw_only=True)
class KVBankEntry:
    """One stored chunk or latent frame and its metadata."""

    chunk_idx: int
    """AR chunk that produced this entry."""

    frame: int
    """Global latent-frame index of the entry's first frame (its timestamp in the rollout)."""

    num_frames: int
    """Latent frames covered: the chunk length, or 1 at frame granularity."""

    meta: dict[str, Any] = field(default_factory=dict)
    """Caller metadata (action, camera pose, ...). Per-frame values keep a
    leading ``num_frames`` dim."""

    slot: int = -1
    """Row in the bank storage; ``-1`` for a query that is not stored."""


@dataclass(kw_only=True)
class KVRetrievalPolicy:
    """Choose which bank entries to fetch into the window. Subclass to add a policy.

    Policies are plain dataclasses so they can be set from configs.
    """

    def select(
        self,
        bank: KVBank,
        query: KVBankEntry,
        candidates: list[KVBankEntry],
        k: int,
    ) -> list[KVBankEntry]:
        """Return at most ``k`` of ``candidates``.

        Args:
            bank: The bank (``bank.entries`` lists everything stored).
            query: The chunk being generated, with its un-split metadata.
            candidates: Stored entries outside the sink and the recent window,
                oldest first.
            k: Number of entries the retrieve region holds (``> 0``).
        """
        raise NotImplementedError


@dataclass(kw_only=True)
class RecentRetrieval(KVRetrievalPolicy):
    """The ``k`` newest candidates: a longer sliding window. Needs no metadata."""

    def select(
        self,
        bank: KVBank,
        query: KVBankEntry,
        candidates: list[KVBankEntry],
        k: int,
    ) -> list[KVBankEntry]:
        return candidates[-k:]


@dataclass(kw_only=True)
class CameraPoseRetrieval(KVRetrievalPolicy):
    """Nearest camera poses: normalized translation L2 + rotation geodesic.

    Reads ``meta[key]`` camera-to-world poses ``[..., 4, 4]``, averaged per
    entry (same scoring as WorldKV's ``image2video_fast.py``).
    """

    key: str = "poses"
    """Metadata key holding absolute camera-to-world poses."""

    translation_weight: float = 0.5
    """Weight of the translation term; rotation gets ``1 - translation_weight``."""

    def select(
        self,
        bank: KVBank,
        query: KVBankEntry,
        candidates: list[KVBankEntry],
        k: int,
    ) -> list[KVBankEntry]:
        def mean_pose(entry: KVBankEntry) -> tuple[Tensor, Tensor]:
            pose = entry.meta[self.key].reshape(-1, 4, 4).float().cpu()
            return pose[:, :3, 3].mean(0), pose[:, :3, :3].mean(0)

        def normalize(d: Tensor) -> Tensor:
            return d / d.max() if d.max() > 0 else d

        query_t, query_r = mean_pose(query)
        trans, rots = zip(*(mean_pose(c) for c in candidates), strict=True)
        trans_dist = (torch.stack(trans) - query_t).square().sum(-1)
        rel = torch.stack(rots).transpose(-1, -2) @ query_r
        cos = (rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
        rot_dist = torch.acos(cos.clamp(-1.0, 1.0))
        w = self.translation_weight
        dist = w * normalize(trans_dist) + (1 - w) * normalize(rot_dist)
        idx = dist.topk(min(k, len(candidates)), largest=False).indices
        return [candidates[i] for i in idx.tolist()]


@dataclass(kw_only=True)
class KVBankConfig:
    """Host K/V bank and retrieval knobs (frames are pre-patchify, like ``window_size_t``)."""

    budget_gb: float = 8.0
    """Pinned host memory for stored entries, summed over all layers and CFG branches."""

    granularity: Literal["chunk", "frame"] = "chunk"
    """Store and retrieve whole AR chunks or single latent frames."""

    retrieve_size_t: int = 0
    """Frames fetched into the window per forward; a multiple of the entry size."""

    policy: KVRetrievalPolicy = field(default_factory=RecentRetrieval)
    """Retrieval policy; any :class:`KVRetrievalPolicy` subclass."""


class KVBank:
    """Budgeted pinned-host store of past K/V entries plus their metadata.

    One bank serves every layer and both CFG branches: entries (and their
    metadata) are shared, each registered cache has its own storage rows.
    Lifecycle per chunk: :meth:`begin_chunk` (host-synchronized), optional
    :meth:`annotate`, then per layer :meth:`fetch` / :meth:`store` on the
    offload side stream.
    """

    def __init__(
        self,
        config: KVBankConfig,
        *,
        frames_per_chunk: int,
        sink_frames: int,
        window_frames: int,
    ) -> None:
        """Args are post-patchify frames: chunk length, true sink, rolling window."""
        self.config = config
        self.entry_frames = frames_per_chunk if config.granularity == "chunk" else 1
        if config.retrieve_size_t % self.entry_frames:
            raise ValueError(
                f"retrieve_size_t ({config.retrieve_size_t}) must be a multiple of "
                f"the entry size ({self.entry_frames} frames)."
            )
        self.frames_per_chunk = frames_per_chunk
        self.sink_frames = sink_frames
        self.window_frames = window_frames
        self.k = config.retrieve_size_t // self.entry_frames
        """Entries in the retrieve region."""
        self.entries: list[KVBankEntry] = []
        """Stored entries, oldest first."""
        self._caches: list[BlockKVCache] = []
        self._store: dict[int, tuple[Tensor, Tensor]] = {}
        self._free: list[int] = []
        self._chunk = -1
        self._query = KVBankEntry(chunk_idx=-1, frame=0, num_frames=0)
        self._current: list[tuple[KVBankEntry, int]] = []
        self._plan: list[KVBankEntry] | None = None

    def add(self, cache: BlockKVCache) -> None:
        """Register a cache; storage is allocated at the first :meth:`begin_chunk`."""
        assert not self._store, "add every cache before the first chunk"
        self._caches.append(cache)

    def entry_tokens(self, cache: BlockKVCache) -> int:
        """Tokens per entry in ``cache``'s sequence dim."""
        return cache.chunk_size * self.entry_frames // self.frames_per_chunk

    def memory_start(self, cache: BlockKVCache) -> int:
        """First token of the retrieve region (end of the true sink)."""
        return cache.sink_size - self.k * self.entry_tokens(cache)

    def reset(self) -> None:
        """Drop all entries; storage stays allocated."""
        self.entries.clear()
        self._free = list(range(self.capacity)) if self._store else []
        self._chunk = -1
        self._current = []
        self._plan = None

    @property
    def capacity(self) -> int:
        """Entries the budget holds (``0`` before allocation)."""
        return next(iter(self._store.values()))[0].shape[0] if self._store else 0

    def begin_chunk(self, chunk_idx: int) -> None:
        """Allocate entries for ``chunk_idx``; repeat calls are no-ops.

        Must run with no copies in flight (the offload synchronizes first),
        since eviction hands storage rows to the new chunk.
        """
        if chunk_idx == self._chunk:
            return
        if not self._store:
            self._allocate()
        self._chunk = chunk_idx
        first = chunk_idx * self.frames_per_chunk
        self._query = KVBankEntry(
            chunk_idx=chunk_idx, frame=first, num_frames=self.frames_per_chunk
        )
        self._current = []
        self._plan = None
        for offset in range(0, self.frames_per_chunk, self.entry_frames):
            if first + offset < self.sink_frames:
                continue  # the true sink stays resident; never a candidate
            if not self._free:
                # ponytail: FIFO eviction (oldest entry first); add an eviction
                # policy hook if retrieval should keep old but relevant entries.
                self._free.append(self.entries.pop(0).slot)
            entry = KVBankEntry(
                chunk_idx=chunk_idx,
                frame=first + offset,
                num_frames=self.entry_frames,
                slot=self._free.pop(),
            )
            self.entries.append(entry)
            self._current.append((entry, offset // self.entry_frames))

    def annotate(
        self,
        chunk_idx: int,
        per_frame: Mapping[str, Any] | None = None,
        **shared: Any,
    ) -> None:
        """Attach metadata to the chunk being generated.

        Args:
            chunk_idx: Must be the chunk passed to the last :meth:`begin_chunk`.
            per_frame: Values with a leading ``frames_per_chunk`` dim (e.g. one
                pose per latent frame); frame entries get their own slice.
            shared: Values stored as-is on every entry of the chunk.
        """
        assert chunk_idx == self._chunk, (
            f"annotate({chunk_idx}) outside chunk {self._chunk}"
        )
        per_frame = dict(per_frame or {})
        for name, value in per_frame.items():
            assert len(value) == self.frames_per_chunk, (
                f"per_frame[{name!r}] has {len(value)} frames, "
                f"expected {self.frames_per_chunk}"
            )
        self._query.meta = {**shared, **per_frame}
        for entry, index in self._current:
            start = index * self.entry_frames
            entry.meta = {
                **shared,
                **{
                    name: value[start : start + self.entry_frames]
                    for name, value in per_frame.items()
                },
            }

    def plan(self) -> list[KVBankEntry]:
        """Entries to fetch for the current chunk, oldest first (computed once per chunk)."""
        if self._plan is None:
            self._plan = []
            first = self._query.frame
            if self.k and first >= self.sink_frames + self.k * self.entry_frames:
                window_start = max(
                    self.sink_frames + self.k * self.entry_frames,
                    first + self.frames_per_chunk - self.window_frames,
                )
                candidates = [
                    e for e in self.entries if e.frame + e.num_frames <= window_start
                ]
                if candidates:
                    chosen = self.config.policy.select(
                        self, self._query, candidates, self.k
                    )
                    self._plan = sorted(chosen[: self.k], key=lambda e: e.frame)
        return self._plan

    def fetch(
        self, cache: BlockKVCache, slot_k: Tensor, slot_v: Tensor
    ) -> list[tuple[Tensor, Tensor]]:
        """``(dst, src)`` copies of the planned entries into the retrieve region."""
        store_k, store_v = self._store[id(cache)]
        size = self.entry_tokens(cache)
        start = self.memory_start(cache)
        copies = []
        for j, entry in enumerate(self.plan()):
            region = cache._seq_slice(start + j * size, start + (j + 1) * size)
            copies += [
                (slot_k[region], store_k[entry.slot]),
                (slot_v[region], store_v[entry.slot]),
            ]
        return copies

    def store(self, cache: BlockKVCache) -> list[tuple[Tensor, Tensor]]:
        """``(dst, src)`` copies of the current chunk from ``cache``'s device slot into the bank."""
        store_k, store_v = self._store[id(cache)]
        size = self.entry_tokens(cache)
        write_start, _ = cache._current_write_bounds()
        copies = []
        for entry, index in self._current:
            region = cache._seq_slice(
                write_start + index * size, write_start + (index + 1) * size
            )
            copies += [
                (store_k[entry.slot], cache._k[region]),
                (store_v[entry.slot], cache._v[region]),
            ]
        return copies

    def _allocate(self) -> None:
        assert self._caches, "no caches registered"
        shapes = []
        entry_bytes = 0
        for cache in self._caches:
            assert cache.window_size >= cache.chunk_size, (
                "the bank needs window_size >= chunk_size"
            )
            size = self.entry_tokens(cache)
            assert size * self.frames_per_chunk == cache.chunk_size * self.entry_frames
            k_shape, v_shape = list(cache.k_shape), list(cache.v_shape)
            k_shape[cache.seq_dim] = v_shape[cache.seq_dim] = size
            shapes.append((k_shape, v_shape))
            entry_bytes += (
                (torch.Size(k_shape).numel() + torch.Size(v_shape).numel())
                * cache.dtype.itemsize
            )
        capacity = int(self.config.budget_gb * 1e9) // entry_bytes
        per_chunk = self.frames_per_chunk // self.entry_frames
        if capacity < per_chunk:
            raise ValueError(
                f"kv bank budget {self.config.budget_gb} GB holds {capacity} entries; "
                f"one chunk needs {per_chunk} ({entry_bytes / 1e6:.1f} MB each)."
            )
        pin = torch.cuda.is_available()
        for cache, (k_shape, v_shape) in zip(self._caches, shapes, strict=True):
            self._store[id(cache)] = (
                torch.empty((capacity, *k_shape), dtype=cache.dtype, pin_memory=pin),
                torch.empty((capacity, *v_shape), dtype=cache.dtype, pin_memory=pin),
            )
        self._free = list(range(capacity))


__all__ = [
    "CameraPoseRetrieval",
    "KVBank",
    "KVBankConfig",
    "KVBankEntry",
    "KVRetrievalPolicy",
    "RecentRetrieval",
]
