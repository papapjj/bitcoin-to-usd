"""Load-time migrations for removed artifact-only implementation details.

This module deliberately contains no runtime structural-development feature.
It exists solely so checkpoints written before that feature was removed can be
read once, migrated by the active policy compatibility hook, and re-saved
without their deleted module dependency.
"""

from __future__ import annotations

import pickle
from typing import BinaryIO, Any


_REMOVED_CONTEXT_STRUCTURAL_MODULE = "self_growing_trader.phat_trien_cau_truc_context"
_REMOVED_CONTEXT_BRANCH_CLASS = (
    "self_growing_trader.he_chuyen_gia_ngu_canh",
    "BoUomNguCanh",
)


class _RemovedLegacyArtifactState:
    """Inert carrier used while unpickling removed artifact-only state.

    Old artifacts only need this object to survive deserialization long enough
    for the active compatibility hook to discard the deleted field. It must
    never be instantiated by the live learning or decision paths.
    """

    def __setstate__(self, state: object) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            # Preserve unusual old pickle state only until the facade removes
            # its containing controller attribute.  Avoid guessing an old
            # class layout or reviving a removed feature.
            self.__dict__["_legacy_pickle_state"] = state


class _ContextStructuralMigrationUnpickler(pickle.Unpickler):
    """Map removed artifact-only classes to one inert carrier."""

    def find_class(self, module: str, name: str) -> Any:
        if (
            module == _REMOVED_CONTEXT_STRUCTURAL_MODULE
            or (module, name) == _REMOVED_CONTEXT_BRANCH_CLASS
        ):
            return _RemovedLegacyArtifactState
        return super().find_class(module, name)


def load_pickle_with_removed_context_structural_migration(handle: BinaryIO) -> Any:
    """Load an artifact while accepting the deleted structural-controller type.

    The caller remains responsible for validating the artifact contract and
    invoking its compatibility migration immediately after this function.
    """

    return _ContextStructuralMigrationUnpickler(handle).load()
