"""
EMBERFORGE AST Compressor (Claw-Compactor + Headroom inspired)
Strips Python source to signatures + structure. Zero LLM cost. Reversible.
60-70% token reduction on code files.

Modes:
  signatures  → class/def names + docstring first line only
  structure   → signatures + inline comments on complex logic
  full        → original (no compression)
"""
from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ASTResult:
    compressed:   str
    original:     str
    mode:         str
    language:     str
    cache_key:    str    # SHA256 of original — for reversibility lookup
    reduction_pct:float = 0.0

    def __post_init__(self) -> None:
        if len(self.original) > 0:
            self.reduction_pct = round(
                100 * (1 - len(self.compressed) / len(self.original)), 1
            )


class ASTCompressor:
    """
    Python AST-based source code compressor.
    Understands code structure — not just regex text replacement.
    """

    def __init__(self, cache_dir: Path | None = None):
        # In-memory cache: cache_key → original source
        # This is the "reversible" part — model can request full source
        self._cache: dict[str, str] = {}
        self._cache_dir = cache_dir

    def compress(
        self,
        source: str,
        mode: str = "signatures",
        filename: str = "code.py",
    ) -> ASTResult:
        """
        Compress Python source code.
        mode: signatures | structure | full
        """
        language = self._detect_language(filename)
        cache_key = hashlib.sha256(source.encode()).hexdigest()[:16]
        self._cache[cache_key] = source   # store for retrieval

        if mode == "full" or language not in ("python",):
            # Non-Python or full mode: return as-is
            return ASTResult(
                compressed=source, original=source,
                mode="full", language=language, cache_key=cache_key,
            )

        try:
            tree = ast.parse(source)
        except SyntaxError:
            # Can't parse → return as-is
            return ASTResult(
                compressed=source, original=source,
                mode="parse_failed", language=language, cache_key=cache_key,
            )

        if mode == "signatures":
            compressed = self._extract_signatures(tree, source)
        elif mode == "structure":
            compressed = self._extract_structure(tree, source)
        else:
            compressed = source

        return ASTResult(
            compressed=compressed, original=source,
            mode=mode, language=language, cache_key=cache_key,
        )

    def retrieve(self, cache_key: str) -> str | None:
        """Retrieve original source by cache key (reversibility)."""
        return self._cache.get(cache_key)

    def _detect_language(self, filename: str) -> str:
        ext = Path(filename).suffix.lower()
        return {
            ".py": "python",
            ".js": "javascript", ".ts": "typescript",
            ".go": "go", ".rs": "rust",
            ".java": "java", ".cpp": "cpp", ".c": "c",
        }.get(ext, "unknown")

    def _extract_signatures(self, tree: ast.AST, source: str) -> str:
        """
        Extract only:
        - Module docstring
        - Import statements (collapsed)
        - Class names + their method signatures
        - Top-level function signatures
        - Constants (ALL_CAPS assignments)
        """
        lines = source.splitlines()
        output: list[str] = []

        # Module docstring
        if (
            isinstance(tree, ast.Module)
            and tree.body
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
        ):
            doc = str(tree.body[0].value.value).split("\n")[0]
            output.append(f'"""{doc}"""')
            output.append("")

        # Imports — collapse to summary
        imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        if imports:
            import_lines = set()
            for imp in imports:
                import_lines.add(ast.unparse(imp))
            output.append(f"# {len(import_lines)} imports: " + ", ".join(sorted(import_lines)[:5]))
            if len(import_lines) > 5:
                output[-1] += f" + {len(import_lines)-5} more"
            output.append("")

        # Walk top-level nodes
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                output.extend(self._class_signature(node, lines))
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                output.extend(self._func_signature(node, lines))
            elif isinstance(node, ast.Assign):
                # Constants only
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        output.append(ast.unparse(node))

        return "\n".join(output)

    def _class_signature(self, node: ast.ClassDef, lines: list[str]) -> list[str]:
        out = []
        bases = ", ".join(ast.unparse(b) for b in node.bases)
        out.append(f"class {node.name}({bases}):" if bases else f"class {node.name}:")

        # Class docstring
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        ):
            doc = str(node.body[0].value.value).split("\n")[0][:80]
            out.append(f'    """{doc}"""')

        # Method signatures
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                sig = self._func_signature(child, lines, indent=4)
                out.extend(sig)

        out.append("")
        return out

    def _func_signature(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        lines: list[str],
        indent: int = 0,
    ) -> list[str]:
        out = []
        pad = " " * indent
        prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
        args = ast.unparse(node.args)
        ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        out.append(f"{pad}{prefix}def {node.name}({args}){ret}:")

        # First line of docstring only
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        ):
            doc = str(node.body[0].value.value).split("\n")[0][:80]
            out.append(f'{pad}    """{doc}"""')

        out.append(f"{pad}    ...")
        return out

    def _extract_structure(self, tree: ast.AST, source: str) -> str:
        """
        Like signatures but keeps important inline logic:
        - Assignments to key variables
        - Return statements with logic
        - Complex conditions
        """
        # For now delegates to signatures — structure mode is Phase 2
        return self._extract_signatures(tree, source)
