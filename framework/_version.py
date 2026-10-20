"""Build-time version, injected because this is a source SNAPSHOT, not a clone.

`aibuildai_version._derive_version()` tries, in order: this module, then
`git describe`, then installed package metadata, then "0.0.0+unknown". A snapshot
has no `.git`, so without this file the framework would report 0.0.0+unknown and
the published result would not be traceable to a revision.

These are the values `git describe` produced in the tree that ran the search:

    science-2026-08-04-439-gd4c0692e7-dirty

The `.dirty` suffix is honest: that tree carried four small local modifications,
listed in EXCLUDED.md under "Local modifications". None of them is on the code
path this search uses.
"""

APP_VERSION = "2026.8.4+439.gd4c0692e7.dirty"
APP_LINE = "science"
