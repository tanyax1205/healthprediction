"""Pytest bootstrap: point artifacts at a temp dir so tests never touch the
real trained artifacts (and never clobber the latest model for a task)."""

from __future__ import annotations

import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="multidx-test-")
os.environ["MULTIDX_ARTIFACTS"] = os.path.join(_TMP, "artifacts")