from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class NetworkLineage:
    network_id: str
    parent_id: str | None
    root_id: str
    generation: int
    blueprint_id: str | None
    role: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    promotion_count: int = 0
    failure_count: int = 0
    child_count: int = 0
    archived: bool = False

    def record_promotion(self) -> None:
        self.promotion_count += 1
        self.failure_count = 0

    def record_failure(self) -> None:
        self.failure_count += 1

    def can_spawn_child(
        self,
        *,
        min_promotions: int = 1,
        max_generation: int | None = None,
        max_children_per_network: int | None = None,
    ) -> bool:
        return (
            not self.archived
            and self.promotion_count >= min_promotions
            and (max_generation is None or max_generation <= 0 or self.generation < max_generation)
            and (max_children_per_network is None or max_children_per_network <= 0 or self.child_count < max_children_per_network)
        )

    def make_child(
        self,
        *,
        network_id: str,
        blueprint_id: str,
        role: str,
        min_promotions: int = 1,
        max_generation: int | None = None,
        max_children_per_network: int | None = None,
    ) -> "NetworkLineage":
        if not self.can_spawn_child(
            min_promotions=min_promotions,
            max_generation=max_generation,
            max_children_per_network=max_children_per_network,
        ):
            raise ValueError("lineage is not allowed to spawn child")
        if network_id == self.network_id or network_id == self.parent_id:
            raise ValueError("lineage cycle detected")
        self.child_count += 1
        return NetworkLineage(
            network_id=network_id,
            parent_id=self.network_id,
            root_id=self.root_id,
            generation=self.generation + 1,
            blueprint_id=blueprint_id,
            role=role,
        )

    def stale(self, *, max_failures: int = 5) -> bool:
        return self.failure_count >= max_failures

    def archive_if_stale(self, *, max_failures: int = 5) -> bool:
        if self.stale(max_failures=max_failures):
            self.archived = True
        return self.archived


@dataclass
class LineageRegistry:
    max_alive_networks: int | None = None
    entries: dict[str, NetworkLineage] = field(default_factory=dict)

    def add(self, lineage: NetworkLineage) -> None:
        if lineage.network_id in self.entries:
            raise ValueError(f"duplicate lineage network id: {lineage.network_id}")
        if lineage.parent_id is not None and lineage.parent_id not in self.entries:
            raise ValueError(f"missing parent lineage: {lineage.parent_id}")
        if self.max_alive_networks is not None and self.max_alive_networks > 0 and self.alive_count() >= self.max_alive_networks:
            raise ValueError("alive network budget exceeded")
        self.entries[lineage.network_id] = lineage

    def alive_count(self) -> int:
        return sum(1 for entry in self.entries.values() if not entry.archived)

    def would_create_cycle(self, network_id: str, parent_id: str | None) -> bool:
        seen = {network_id}
        current = parent_id
        while current is not None:
            if current in seen:
                return True
            seen.add(current)
            current = self.entries[current].parent_id if current in self.entries else None
        return False
