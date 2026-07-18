"""
EmberForge Eval Tasks — deterministic coding tasks in disposable sandbox repos.

Every task defines:
  setup(repo)  — build a tiny repo where the task is genuinely not done yet
  verify(repo) — deterministic pass/fail (usually: does pytest pass now?)
  golden(repo) — a known-correct reference fix

The golden fix exists so the harness can prove ITS OWN correctness without an
LLM: for every task, verify() must be False after setup() and True after
golden(). That property is enforced by unit tests — no flaky, unverifiable
evals allowed in this registry.
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


def run_pytest(repo: Path, timeout: int = 60) -> bool:
    """Run pytest inside the sandbox; True iff everything passes."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"],
            cwd=str(repo), capture_output=True, text=True, timeout=timeout,
        )
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        return False


@dataclass
class EvalTask:
    name:      str
    task_type: str            # debug | write | refactor | feature | docs
    prompt:    str            # what the agent is asked to do
    setup:     Callable[[Path], None]
    verify:    Callable[[Path], bool]
    golden:    Callable[[Path], None]


def _w(repo: Path, name: str, content: str) -> None:
    (repo / name).write_text(content, encoding="utf-8")


# ── 1. fix-zero-division (debug) ─────────────────────────────────────────────
def _zero_setup(repo: Path) -> None:
    _w(repo, "calc.py",
       "def divide(a, b):\n"
       "    return a / b\n")
    _w(repo, "test_calc.py",
       "import pytest\n"
       "from calc import divide\n\n"
       "def test_divide():\n"
       "    assert divide(6, 2) == 3\n\n"
       "def test_divide_by_zero_raises_value_error():\n"
       "    with pytest.raises(ValueError):\n"
       "        divide(5, 0)\n")


def _zero_golden(repo: Path) -> None:
    _w(repo, "calc.py",
       "def divide(a, b):\n"
       "    if b == 0:\n"
       "        raise ValueError('division by zero')\n"
       "    return a / b\n")


# ── 2. implement-function (write) ────────────────────────────────────────────
def _slug_setup(repo: Path) -> None:
    _w(repo, "stringx.py",
       "def slugify(text):\n"
       "    raise NotImplementedError\n")
    _w(repo, "test_stringx.py",
       "from stringx import slugify\n\n"
       "def test_basic():\n"
       "    assert slugify('Hello World!') == 'hello-world'\n\n"
       "def test_extra_spaces():\n"
       "    assert slugify('  A  B  ') == 'a-b'\n")


def _slug_golden(repo: Path) -> None:
    _w(repo, "stringx.py",
       "import re\n\n"
       "def slugify(text):\n"
       "    words = re.findall(r'[a-z0-9]+', text.lower())\n"
       "    return '-'.join(words)\n")


# ── 3. rename-across-files (refactor, multi-file) ────────────────────────────
def _rename_setup(repo: Path) -> None:
    _w(repo, "core.py",
       "def calc_total(items):\n"
       "    return sum(items)\n")
    _w(repo, "app.py",
       "from core import calc_total\n\n"
       "def get_price(items):\n"
       "    return calc_total(items) * 1.2\n")
    _w(repo, "test_rename.py",
       "from core import compute_total\n"
       "from app import get_price\n\n"
       "def test_compute_total():\n"
       "    assert compute_total([1, 2, 3]) == 6\n\n"
       "def test_get_price():\n"
       "    assert get_price([10]) == 12\n")


def _rename_golden(repo: Path) -> None:
    _w(repo, "core.py",
       "def compute_total(items):\n"
       "    return sum(items)\n")
    _w(repo, "app.py",
       "from core import compute_total\n\n"
       "def get_price(items):\n"
       "    return compute_total(items) * 1.2\n")


# ── 4. add-method (feature) ──────────────────────────────────────────────────
def _cart_setup(repo: Path) -> None:
    _w(repo, "models.py",
       "class Cart:\n"
       "    def __init__(self):\n"
       "        self.items = []\n\n"
       "    def add(self, name, price):\n"
       "        self.items.append((name, price))\n")
    _w(repo, "test_models.py",
       "from models import Cart\n\n"
       "def test_total():\n"
       "    c = Cart()\n"
       "    c.add('apple', 2.5)\n"
       "    c.add('bread', 1.5)\n"
       "    assert c.total() == 4.0\n\n"
       "def test_total_empty():\n"
       "    assert Cart().total() == 0\n")


def _cart_golden(repo: Path) -> None:
    _w(repo, "models.py",
       "class Cart:\n"
       "    def __init__(self):\n"
       "        self.items = []\n\n"
       "    def add(self, name, price):\n"
       "        self.items.append((name, price))\n\n"
       "    def total(self):\n"
       "        return sum(price for _, price in self.items)\n")


# ── 5. fix-import-crash (debug) ──────────────────────────────────────────────
def _import_setup(repo: Path) -> None:
    _w(repo, "config.py",
       "import jsonn\n\n"
       "def get_config():\n"
       "    return jsonn.loads('{\"debug\": false}')\n")
    _w(repo, "test_config.py",
       "from config import get_config\n\n"
       "def test_debug_off():\n"
       "    assert get_config()['debug'] is False\n")


def _import_golden(repo: Path) -> None:
    _w(repo, "config.py",
       "import json\n\n"
       "def get_config():\n"
       "    return json.loads('{\"debug\": false}')\n")


# ── 6. document-install (docs, non-test verification) ────────────────────────
def _docs_setup(repo: Path) -> None:
    _w(repo, "README.md",
       "# demo-pkg\n\nA demo package.\n")


def _docs_verify(repo: Path) -> bool:
    text = (repo / "README.md").read_text(encoding="utf-8", errors="ignore")
    return "## Install" in text and "pip install demo-pkg" in text


def _docs_golden(repo: Path) -> None:
    readme = repo / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8")
        + "\n## Install\n\n```bash\npip install demo-pkg\n```\n",
        encoding="utf-8",
    )


# ── Registry ──────────────────────────────────────────────────────────────────
TASKS: list[EvalTask] = [
    EvalTask(
        name="fix-zero-division", task_type="debug",
        prompt="test_calc.py is failing. Fix the bug in calc.py so all tests pass. "
               "Run the tests to verify.",
        setup=_zero_setup, verify=run_pytest, golden=_zero_golden,
    ),
    EvalTask(
        name="implement-function", task_type="write",
        prompt="Implement slugify() in stringx.py so the tests in test_stringx.py "
               "pass. Run the tests to verify.",
        setup=_slug_setup, verify=run_pytest, golden=_slug_golden,
    ),
    EvalTask(
        name="rename-across-files", task_type="refactor",
        prompt="Rename the function calc_total to compute_total everywhere it is "
               "defined and used (core.py and app.py). Run the tests to verify.",
        setup=_rename_setup, verify=run_pytest, golden=_rename_golden,
    ),
    EvalTask(
        name="add-method", task_type="feature",
        prompt="Add a total() method to the Cart class in models.py so the tests "
               "in test_models.py pass. Run the tests to verify.",
        setup=_cart_setup, verify=run_pytest, golden=_cart_golden,
    ),
    EvalTask(
        name="fix-import-crash", task_type="debug",
        prompt="Importing config.py crashes. Fix it so the tests in "
               "test_config.py pass. Run the tests to verify.",
        setup=_import_setup, verify=run_pytest, golden=_import_golden,
    ),
    EvalTask(
        name="document-install", task_type="docs",
        prompt="Add an '## Install' section to README.md documenting installation "
               "with: pip install demo-pkg",
        setup=_docs_setup, verify=_docs_verify, golden=_docs_golden,
    ),
]


def get_task(name: str) -> EvalTask | None:
    return next((t for t in TASKS if t.name == name), None)
