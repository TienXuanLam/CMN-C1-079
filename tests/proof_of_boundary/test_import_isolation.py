# PB-4: Import Isolation — CMN-C1-079 MultiLangIntentRoutingAgent
#
# Architecture contract (docs/02_design.md):
#   L0 = agenticstar-platform SDK (agenticstar / agenticstar_agentcore)  ← PROHIBITED
#   L1 = framework.*                                                       ← ALLOWED
#   L2 = shared.*                                                          ← ALLOWED

import ast
import os

import pytest

_L0_PROHIBITED = [
    "agenticstar",
    "agenticstar_agentcore",
]

_STDLIB_SAFE = {"platform"}


def _scan_imports(filepath: str) -> list[str]:
    with open(filepath) as f:
        try:
            tree = ast.parse(f.read(), filename=filepath)
        except SyntaxError:
            return []

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for prohibited in _L0_PROHIBITED:
                    if alias.name == prohibited or alias.name.startswith(f"{prohibited}."):
                        if alias.name in _STDLIB_SAFE:
                            continue
                        violations.append(f"{filepath}:{node.lineno} — import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for prohibited in _L0_PROHIBITED:
                    if node.module == prohibited or node.module.startswith(f"{prohibited}."):
                        if node.module in _STDLIB_SAFE:
                            continue
                        violations.append(f"{filepath}:{node.lineno} — from {node.module} import ...")
    return violations


def _find_python_files(directory: str, exclude_dirs: set[str] | None = None) -> list[str]:
    exclude_dirs = exclude_dirs or set()
    py_files: list[str] = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.join(root, f))
    return py_files


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class TestImportIsolation:
    """PB-4: Agent code must not import L0 (agenticstar / agenticstar_agentcore)."""

    def test_no_l0_imports_in_src(self):
        src_dir = os.path.join(_project_root(), "src")
        if not os.path.exists(src_dir):
            pytest.skip("src/ directory not found")
        violations: list[str] = []
        for filepath in _find_python_files(src_dir, exclude_dirs={"examples"}):
            violations.extend(_scan_imports(filepath))
        assert violations == [], "L0 import isolation violated in src/:\n" + "\n".join(violations)

    def test_no_l0_imports_in_tests(self):
        tests_dir = os.path.join(_project_root(), "tests")
        if not os.path.exists(tests_dir):
            pytest.skip("tests/ directory not found")
        violations: list[str] = []
        for filepath in _find_python_files(tests_dir, exclude_dirs={"proof_of_boundary", "__pycache__"}):
            violations.extend(_scan_imports(filepath))
        assert violations == [], "L0 import isolation violated in tests/:\n" + "\n".join(violations)

    def test_src_uses_l1_framework_imports(self):
        src_dir = os.path.join(_project_root(), "src")
        if not os.path.exists(src_dir):
            pytest.skip("src/ directory not found")
        l1_found = False
        for filepath in _find_python_files(src_dir, exclude_dirs={"examples"}):
            with open(filepath) as f:
                try:
                    tree = ast.parse(f.read(), filename=filepath)
                except SyntaxError:
                    continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("framework."):
                    l1_found = True
                    break
            if l1_found:
                break
        assert l1_found, "No L1 (framework.*) imports found in src/"

    def test_src_uses_l2_shared_imports(self):
        src_dir = os.path.join(_project_root(), "src")
        if not os.path.exists(src_dir):
            pytest.skip("src/ directory not found")
        l2_found = False
        for filepath in _find_python_files(src_dir, exclude_dirs={"examples"}):
            with open(filepath) as f:
                try:
                    tree = ast.parse(f.read(), filename=filepath)
                except SyntaxError:
                    continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("shared."):
                    l2_found = True
                    break
            if l2_found:
                break
        assert l2_found, "No L2 (shared.*) imports found in src/"
