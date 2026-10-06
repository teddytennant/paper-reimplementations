"""Layer partition and epoch schedule for Algorithm 1."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RotationSchedule:
    """Partition of an ordered layer list into F, A, and rotatable groups.

    F is a permanently frozen input-side prefix. A is an always-active
    output-side anchor. R is every remaining layer, in model order, split
    into consecutive groups G_1, ..., G_K of size at most b. K is the number
    of those groups and can exceed 1 when R is longer than b.
    """

    layers: tuple
    frozen: frozenset
    anchor: frozenset
    rotatable: tuple
    groups: tuple
    group_size: int

    @property
    def K(self) -> int:
        return len(self.groups)

    @property
    def depth(self) -> int:
        return len(self.layers)


@dataclass(frozen=True)
class EpochPlan:
    """Active set for one 1-indexed epoch.

    k = 1 + ((e - 1) mod K)
    S_e = A union G_k
    l_start = min(G_k), the first layer of the active block.
    """

    epoch: int
    k: int
    group: tuple
    active: frozenset
    l_start: object


def partition_layers(layers, frozen, anchor, group_size) -> RotationSchedule:
    """Derive R and its groups from an ordered layer list, F, A, and b.

    Layers not listed in F or A form R, kept in the given order. R is then
    cut into consecutive groups of length at most group_size. The last group
    is shorter than b when len(R) is not a multiple of b.
    """

    layer_list = list(layers)
    if len(layer_list) != len(set(layer_list)):
        raise ValueError("layer ids must be unique")
    if group_size < 1:
        raise ValueError("group size b must be at least 1")

    frozen_set = frozenset(frozen)
    anchor_set = frozenset(anchor)
    known = set(layer_list)
    unknown = (frozen_set | anchor_set) - known
    if unknown:
        raise ValueError(f"F or A contains unknown layers: {sorted(unknown, key=_sort_key)}")
    overlap = frozen_set & anchor_set
    if overlap:
        raise ValueError(f"F and A overlap: {sorted(overlap, key=_sort_key)}")

    rotatable = tuple(lid for lid in layer_list if lid not in frozen_set and lid not in anchor_set)
    groups = tuple(
        tuple(rotatable[start : start + group_size])
        for start in range(0, len(rotatable), group_size)
    )
    return RotationSchedule(
        layers=tuple(layer_list),
        frozen=frozen_set,
        anchor=anchor_set,
        rotatable=rotatable,
        groups=groups,
        group_size=int(group_size),
    )


def active_set(schedule: RotationSchedule, epoch: int) -> EpochPlan:
    """Active layers for a 1-indexed epoch, following Algorithm 1."""

    if epoch < 1:
        raise ValueError("epochs are 1-indexed and must be at least 1")
    if schedule.K < 1:
        raise ValueError("rotatable set is empty, so there is no group to activate")

    k = 1 + ((epoch - 1) % schedule.K)
    group = schedule.groups[k - 1]
    return EpochPlan(
        epoch=epoch,
        k=k,
        group=group,
        active=frozenset(schedule.anchor) | frozenset(group),
        l_start=min(group),
    )


def layers_before(layers, l_start):
    """Layers that sit before l_start in forward order."""

    prefix = []
    for lid in layers:
        if lid == l_start:
            return prefix
        prefix.append(lid)
    raise ValueError(f"l_start {l_start!r} is not in the layer list")


def _sort_key(layer_id):
    if isinstance(layer_id, int):
        return (0, layer_id, "")
    return (1, 0, str(layer_id))
