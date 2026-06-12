"""Remap-stage exceptions. Fail fast -- no silent fallbacks."""


class RemapError(Exception):
    """Base class for remap-stage failures."""


class UnmappedLabelError(RemapError):
    """A source label is neither in the mapping nor in drops (mapping must be total)."""


class ClassNamesError(RemapError):
    """The class-index order of a detection source cannot be determined."""
