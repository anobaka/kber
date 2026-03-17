"""Code repository analysis – Git operations, AST parsing, and knowledge generation.

Architecture:
    Three levels of knowledge are generated for each repository:
    1. **Block-level** – per function/class descriptions (existing)
    2. **Module-level** – per directory summaries (new)
    3. **Repo-level** – global architecture overview (new)

    Block-level progress is tracked in the ``code_block`` table with a
    ``status`` column (pending / success / failed).  Failed blocks are
    automatically retried on the next run.  The ``content_hash`` column
    detects actual code changes so unchanged blocks are skipped.

    When blocks in a directory change, the module summary for that
    directory is regenerated.  When any module summary changes, the
    repo overview is regenerated.  This is *change-driven cascading
    regeneration*.
"""

import hashlib
import logging
import os
import re
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import pathspec
from sqlalchemy import delete, select, update

from app.config import config
from app.db.models import CodeBlock, CodeRepo, KnowledgeBase, SummarizeTaskLog
from app.db.session import get_session
from app.services.cancel import CancelledError, check_cancelled
from app.services.embedding_service import embedding_service
from app.services.llm_service import llm_service
from app.services.milvus_service import milvus_service

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs",
    ".cs", ".cpp", ".c", ".h", ".rb", ".php", ".swift", ".kt",
    ".scala", ".vue", ".sql", ".proto", ".graphql",
}

BLACKLIST_DIRS = {
    "node_modules", "vendor", "dist", "build", "target", ".git",
    "__pycache__", ".tox", ".nox", ".eggs", ".mypy_cache",
    ".pytest_cache", "venv", ".venv", "env",
}

MAX_BLOCK_RETRY = 3  # Maximum number of retries for failed blocks

MAX_FILE_SIZE = 100 * 1024  # 100KB
MAX_LINE_LENGTH = 500

SENSITIVE_PATTERNS = [
    re.compile(r"""(password|passwd|pwd)\s*[=:]\s*['"][^'"]+['"]""", re.IGNORECASE),
    re.compile(r"""(api_key|apikey|api-key)\s*[=:]\s*['"][^'"]+['"]""", re.IGNORECASE),
    re.compile(r"""(secret|secret_key)\s*[=:]\s*['"][^'"]+['"]""", re.IGNORECASE),
    re.compile(r"""(token|access_token|auth_token)\s*[=:]\s*['"][^'"]+['"]""", re.IGNORECASE),
    re.compile(r"""Bearer\s+[A-Za-z0-9\-._~+/]+=*""", re.IGNORECASE),
    re.compile(r"""(AKIA|ABIA|ACCA|ASIA)[A-Z0-9]{16}"""),
]

LANG_MAP: dict[str, str] = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "tsx", ".java": "java",
    ".go": "go", ".rs": "rust", ".c": "c", ".h": "c", ".cpp": "cpp",
    ".cs": "c_sharp", ".rb": "ruby", ".php": "php",
    ".swift": "swift", ".kt": "kotlin", ".scala": "scala",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ProgressThrottle:
    """Throttle progress notifications: at most every *interval* seconds or
    every *pct_step* percent of total, whichever comes first.

    Also tracks elapsed time to estimate remaining time via :meth:`eta`.
    """

    def __init__(self, total: int, interval: float = 3.0, pct_step: int = 1) -> None:
        self._total = total
        self._interval = interval
        self._pct_step = pct_step
        self._last_time = 0.0
        self._last_pct = -pct_step
        self._start_time = time.monotonic()

    def should_notify(self, done: int) -> bool:
        if self._total <= 0:
            return False
        pct = done * 100 // self._total
        now = time.monotonic()
        if pct - self._last_pct >= self._pct_step or now - self._last_time >= self._interval:
            self._last_time = now
            self._last_pct = pct
            return True
        return False

    def eta(self, done: int) -> str:
        """Return a human-readable ETA string like '约2分30秒'."""
        if done <= 0 or self._total <= 0:
            return ""
        elapsed = time.monotonic() - self._start_time
        remaining = elapsed / done * (self._total - done)
        return _format_duration(remaining)


def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable Chinese duration string."""
    s = int(seconds)
    if s < 5:
        return "即将完成"
    if s < 60:
        return f"约{s}秒"
    m, s = divmod(s, 60)
    if s == 0:
        return f"约{m}分钟"
    return f"约{m}分{s}秒"


def _redact_sensitive(text: str) -> str:
    for pattern in SENSITIVE_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def _content_hash(code: str) -> str:
    """SHA-256 hex digest of code content."""
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _module_path(file_path: str) -> str:
    """Derive the module (directory) key from a file path.

    Uses up-to-2-level directory prefix so that ``app/services/foo.py``
    maps to ``app/services``.  Top-level files map to ``"."``.
    """
    parts = file_path.replace("\\", "/").split("/")
    if len(parts) <= 1:
        return "."
    return "/".join(parts[:2]) if len(parts) > 2 else parts[0]


# ---------------------------------------------------------------------------
# Main analyser
# ---------------------------------------------------------------------------


class RepoAnalyzer:
    """Full pipeline for analysing a code repository."""

    def __init__(self) -> None:
        self._ts_parsers: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    #  Public entry point                                                  #
    # ------------------------------------------------------------------ #

    def analyze_repo(
        self,
        repo_id: int,
        notify_chat_ids: list[str] | None = None,
    ) -> dict[str, int]:
        """Full analysis of a code repository.

        Returns stats dict ``{files_parsed, blocks_found, blocks_success,
        blocks_failed, knowledge_generated, modules_updated}``.
        """
        from app.services.debug_notifier import (
            clear_progress_msg, get_debug_chat_ids_for_repo, notify_repo,
        )

        stats: dict[str, int] = {
            "files_parsed": 0, "blocks_found": 0, "blocks_success": 0,
            "blocks_failed": 0, "blocks_permanently_failed": 0,
            "knowledge_generated": 0, "modules_updated": 0,
        }

        with get_session() as session:
            repo = session.execute(
                select(CodeRepo).where(CodeRepo.id == repo_id)
            ).scalar_one_or_none()
            if not repo:
                logger.error("Code repo %d not found", repo_id)
                return stats
            kb_id = repo.kb_id
            git_url = repo.git_url
            branch = repo.default_branch or ""
            last_commit = repo.last_commit_hash

        # Short repo label for progress messages (e.g. "fusion/crane")
        _url = git_url.split(":")[-1] if ":" in git_url and "//" not in git_url.split(":")[0] else git_url
        repo_label = "/".join(_url.rstrip("/").rsplit("/", 2)[-2:]).removesuffix(".git")

        def _check() -> None:
            if kb_id:
                check_cancelled(kb_id)

        repo_dir = os.path.join(config.REPOS_BASE_DIR, str(repo_id))
        task_log_id = self._start_task_log(kb_id, "code")

        # Unified notification helper.
        # progress=True → edit the previous progress card in-place.
        # done=True → final update for a phase (update card, then clear tracking).
        def _notify(msg: str, *, progress: bool = False, done: bool = False) -> None:
            msg = f"【{repo_label}】{msg}"
            if not progress:
                # Non-progress message: clear tracked card so next progress
                # notification creates a fresh card.
                clear_progress_msg("repo", repo_id)
            notify_repo(repo_id, msg, progress=progress)
            if notify_chat_ids:
                from app.services.debug_notifier import _send_fn, _send_or_update
                if _send_fn:
                    debug_ids = set(get_debug_chat_ids_for_repo(repo_id))
                    for cid in notify_chat_ids:
                        if cid not in debug_ids:
                            if progress:
                                _send_or_update(cid, msg, "repo", repo_id)
                            else:
                                _send_fn(cid, msg)
            if done:
                # Phase finished: clear tracking so next phase gets a fresh card.
                clear_progress_msg("repo", repo_id)

        total_files = 0
        try:
            # ----------------------------------------------------------
            # Step 1: Git clone / pull
            # ----------------------------------------------------------
            _check()
            _notify("📦 正在同步代码仓库...", progress=True)

            is_new = not os.path.exists(os.path.join(repo_dir, ".git"))
            if is_new:
                self._git_clone(git_url, repo_dir, branch)
            else:
                self._git_pull(repo_dir, branch)

            current_commit = self._get_current_commit(repo_dir)

            # File-level diff (commit hash is only for *file-level* scope)
            if last_commit and not is_new:
                changed_files = self._get_changed_files(repo_dir, last_commit, f"origin/{branch}")
            else:
                changed_files = None  # all files

            _notify("✅ 代码仓库同步完成", progress=True, done=True)

            # ----------------------------------------------------------
            # Step 2: Scan & filter files
            # ----------------------------------------------------------
            _check()
            _notify("🔍 正在扫描文件...", progress=True)

            all_files = self._scan_files(repo_dir)
            if changed_files is not None:
                changed_set = {c["path"] for c in changed_files if c["status"] in ("A", "M")}
                files_to_process = [f for f in all_files if self._relative_path(f, repo_dir) in changed_set]
                deleted_paths = [c["path"] for c in changed_files if c["status"] == "D"]
                self._delete_file_knowledge(kb_id, repo_id, deleted_paths)
            else:
                files_to_process = all_files

            total_files = len(files_to_process)
            _notify("✅ 文件扫描完成", progress=True, done=True)

            # ----------------------------------------------------------
            # Step 3: AST parse → code blocks
            # ----------------------------------------------------------
            _check()
            new_blocks: list[dict[str, Any]] = []
            parse_throttle = _ProgressThrottle(total_files)
            for i, fpath in enumerate(files_to_process):
                if parse_throttle.should_notify(i + 1):
                    _check()
                    pct = (i + 1) * 100 // total_files
                    eta = parse_throttle.eta(i + 1)
                    eta_part = f"，预计{eta}" if eta else ""
                    _notify(f"🔍 正在解析代码结构（{pct}%{eta_part}）...", progress=True)
                blocks = self._parse_file(fpath, repo_dir, repo_id)
                new_blocks.extend(blocks)
                stats["files_parsed"] += 1

            if total_files:
                _notify("✅ 代码结构解析完成", progress=True, done=True)

            # ----------------------------------------------------------
            # Step 4: Upsert code_block records, detect what needs LLM
            # ----------------------------------------------------------
            _check()
            blocks_to_generate = self._sync_code_blocks(
                repo_id, new_blocks, current_commit, changed_files,
            )

            # Repair orphaned blocks (success in DB but missing from Milvus)
            repaired = self._repair_orphaned_blocks(repo_id, kb_id)
            if repaired:
                _notify(f"🔧 发现 {repaired} 个孤儿代码块（之前中断导致），已重置为待处理", progress=True)

            # Also pick up failed / orphaned-pending blocks from previous runs
            existing_cb_ids = {b["_cb_id"] for b in blocks_to_generate if b.get("_cb_id")}
            retry_blocks = self._load_failed_blocks(repo_id, exclude_ids=existing_cb_ids)
            blocks_to_generate.extend(retry_blocks)

            # Count permanently failed blocks (exceeded max retry)
            with get_session() as session:
                perm_failed = session.execute(
                    select(CodeBlock).where(
                        CodeBlock.repo_id == repo_id,
                        CodeBlock.status == "failed",
                        CodeBlock.retry_count >= MAX_BLOCK_RETRY,
                    )
                ).scalars().all()
                stats["blocks_permanently_failed"] = len(perm_failed)
                if perm_failed:
                    for pf in perm_failed:
                        logger.warning(
                            "Block permanently failed (retry_count=%d): %s:%s (lines %s-%s) — %s",
                            pf.retry_count, pf.file_path, pf.block_name,
                            pf.start_line, pf.end_line, pf.error_message,
                        )

            stats["blocks_found"] = len(new_blocks) + len(retry_blocks)

            # ----------------------------------------------------------
            # Step 5: LLM knowledge generation (block level)
            # ----------------------------------------------------------
            _check()
            repo_map = self._generate_repo_map(repo_dir, all_files)

            # Load blocks that already have LLM descriptions from a previous
            # interrupted run (status='generated') — skip LLM, go straight
            # to embedding + Milvus write.
            cached_entries = self._load_generated_blocks(repo_id, current_commit)
            if cached_entries:
                _notify(f"♻️ 恢复 {len(cached_entries)} 个已缓存的代码块描述，跳过 LLM 生成", progress=True)

            if blocks_to_generate:
                success_entries, failed_count = self._generate_block_knowledge(
                    blocks_to_generate, repo_map, kb_id, repo_id, current_commit,
                    notify_fn=_notify, check_cancelled_fn=_check,
                )
                stats["blocks_failed"] = failed_count
            else:
                success_entries = []

            # Merge newly generated entries with cached ones
            all_entries = cached_entries + success_entries
            stats["blocks_success"] = len(all_entries)
            stats["knowledge_generated"] = len(all_entries)

            # Write all entries to Milvus (batch embedding)
            if all_entries:
                _notify("💾 正在写入代码块知识...", progress=True)
                self._store_block_knowledge(kb_id, all_entries)
                _notify("✅ 代码块知识写入完成", progress=True, done=True)

            # Always advance commit hash (file-level checkpoint)
            self._update_repo_commit(repo_id, current_commit)

            # ----------------------------------------------------------
            # Step 6: Module summaries (cascading)
            # ----------------------------------------------------------
            _check()
            affected_dirs = self._get_affected_modules(
                repo_id, blocks_to_generate, changed_files,
            )
            if affected_dirs:
                _notify(f"📝 正在更新 {len(affected_dirs)} 个模块摘要...", progress=True)
                updated = self._regenerate_module_summaries(
                    kb_id, repo_id, affected_dirs, repo_map, _notify,
                    check_cancelled_fn=_check,
                )
                stats["modules_updated"] = updated

            # ----------------------------------------------------------
            # Step 7: Repo overview (regenerate if anything changed)
            # ----------------------------------------------------------
            _check()
            if blocks_to_generate or affected_dirs:
                _notify("📋 正在更新仓库概览...", progress=True)
                try:
                    self._regenerate_repo_overview(kb_id, repo_id, repo_map)
                    _notify("✅ 仓库概览更新完成", progress=True, done=True)
                except Exception:
                    logger.exception("Failed to regenerate repo overview for repo %d", repo_id)
                    _notify("⚠️ 仓库概览更新失败", progress=True, done=True)

            self._finish_task_log(task_log_id, "success", stats)

            summary_parts = [
                f"共解析 {stats['files_parsed']} 个文件",
                f"生成 {stats['knowledge_generated']} 条代码知识",
            ]
            if stats["blocks_failed"]:
                summary_parts.append(
                    f"{stats['blocks_failed']} 个代码块失败"
                    f"（将在下次重试，最多 {MAX_BLOCK_RETRY} 次）"
                )
            if stats["blocks_permanently_failed"]:
                summary_parts.append(
                    f"{stats['blocks_permanently_failed']} 个代码块已永久失败（超过最大重试次数）"
                )
            if stats["modules_updated"]:
                summary_parts.append(f"更新 {stats['modules_updated']} 个模块摘要")
            clear_progress_msg("repo", repo_id)
            _notify(f"✅ 代码库分析完成！{'，'.join(summary_parts)}。")

        except CancelledError:
            logger.info("Repo analysis cancelled for repo_id=%d", repo_id)
            self._finish_task_log(task_log_id, "failed", stats, "用户取消")
            clear_progress_msg("repo", repo_id)
            _notify(
                f"🛑 代码库分析已停止。"
                f"已完成：{stats['files_parsed']} 个文件解析，{stats['knowledge_generated']} 条知识。"
            )

        except Exception as e:
            logger.exception("Repo analysis failed for repo_id=%d", repo_id)
            self._finish_task_log(task_log_id, "failed", stats, str(e))
            clear_progress_msg("repo", repo_id)
            _notify(
                f"⚠️ 代码库分析失败：{str(e)[:100]}，"
                f"已成功处理 {stats['files_parsed']}/{total_files} 个文件。"
            )

        return stats

    def check_and_update(self, repo_id: int) -> dict[str, int]:
        return self.analyze_repo(repo_id)

    # ================================================================== #
    #  Git operations                                                      #
    # ================================================================== #

    @staticmethod
    def _build_clone_url(repo_path: str) -> str:
        url = repo_path
        if not url.startswith("http://") and not url.startswith("https://"):
            base = config.GIT_BASE_URL.rstrip("/")
            url = f"{base}/{url.strip('/')}.git"
        pat = config.GIT_PAT
        if not pat:
            return url
        if url.startswith("https://"):
            return url.replace("https://", f"https://oauth2:{pat}@", 1)
        if url.startswith("http://"):
            return url.replace("http://", f"http://oauth2:{pat}@", 1)
        return url

    def _git_clone(self, url: str, dest: str, branch: str) -> None:
        os.makedirs(dest, exist_ok=True)
        auth_url = self._build_clone_url(url)
        cmd = ["git", "clone", auth_url, dest]
        if branch:
            cmd = ["git", "clone", "-b", branch, auth_url, dest]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            stderr = result.stderr.replace(config.GIT_PAT, "***") if config.GIT_PAT else result.stderr
            raise RuntimeError(f"git clone failed (exit {result.returncode}): {stderr}")
        logger.info("Cloned %s to %s", url, dest)

    def _git_pull(self, repo_dir: str, branch: str) -> None:
        subprocess.run(
            ["git", "fetch", "origin"],
            cwd=repo_dir, check=True, capture_output=True, timeout=300,
        )
        subprocess.run(
            ["git", "pull", "origin", branch],
            cwd=repo_dir, check=True, capture_output=True, timeout=300,
        )

    def _get_current_commit(self, repo_dir: str) -> str:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir, capture_output=True, text=True,
        )
        return result.stdout.strip()

    def _get_changed_files(self, repo_dir: str, from_commit: str, to_ref: str) -> list[dict[str, str]]:
        result = subprocess.run(
            ["git", "diff", f"{from_commit}..{to_ref}", "--name-status"],
            cwd=repo_dir, capture_output=True, text=True,
        )
        changes: list[dict[str, str]] = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                changes.append({"status": parts[0][0], "path": parts[-1]})
        return changes

    # ================================================================== #
    #  File scanning                                                       #
    # ================================================================== #

    def _scan_files(self, repo_dir: str) -> list[str]:
        gitignore_path = os.path.join(repo_dir, ".gitignore")
        spec = None
        if os.path.exists(gitignore_path):
            with open(gitignore_path) as f:
                spec = pathspec.PathSpec.from_lines("gitwildmatch", f)

        result: list[str] = []
        for root, dirs, files in os.walk(repo_dir):
            dirs[:] = [d for d in dirs if d not in BLACKLIST_DIRS]
            for fname in files:
                fpath = os.path.join(root, fname)
                rel_path = os.path.relpath(fpath, repo_dir)
                ext = os.path.splitext(fname)[1].lower()
                if ext not in ALLOWED_EXTENSIONS:
                    continue
                if spec and spec.match_file(rel_path):
                    continue
                try:
                    if os.path.getsize(fpath) > MAX_FILE_SIZE:
                        continue
                except OSError:
                    continue
                try:
                    with open(fpath, "r", errors="ignore") as f:
                        first_line = f.readline()
                        if len(first_line) > MAX_LINE_LENGTH:
                            continue
                except Exception:
                    continue
                result.append(fpath)
        return result

    def _relative_path(self, fpath: str, repo_dir: str) -> str:
        return os.path.relpath(fpath, repo_dir)

    # ================================================================== #
    #  AST parsing                                                         #
    # ================================================================== #

    def _parse_file(self, fpath: str, repo_dir: str, repo_id: int) -> list[dict[str, Any]]:
        ext = os.path.splitext(fpath)[1].lower()
        lang = LANG_MAP.get(ext)
        rel_path = self._relative_path(fpath, repo_dir)
        try:
            with open(fpath, "r", errors="ignore") as f:
                source = f.read()
        except Exception:
            return []
        source = _redact_sensitive(source)
        if not source.strip():
            logger.debug("Skipping empty file: %s", rel_path)
            return []
        if lang:
            blocks = self._parse_with_treesitter(source, lang, rel_path, repo_id)
            if blocks:
                return blocks
        return self._fallback_parse(source, rel_path, repo_id, ext)

    def _parse_with_treesitter(self, source: str, lang: str, rel_path: str, repo_id: int) -> list[dict[str, Any]]:
        try:
            import tree_sitter
            if lang not in self._ts_parsers:
                ts_lang = self._load_ts_language(lang)
                if ts_lang is None:
                    return []
                self._ts_parsers[lang] = tree_sitter.Parser(ts_lang)
            parser = self._ts_parsers[lang]
            tree = parser.parse(source.encode("utf-8"))
            blocks: list[dict[str, Any]] = []
            self._extract_blocks(tree.root_node, source, rel_path, repo_id, lang, blocks, parent_class=None)
            return blocks
        except Exception as e:
            logger.debug("Tree-sitter parse failed for %s: %s", rel_path, e)
            return []

    def _load_ts_language(self, lang: str) -> Any:
        try:
            if lang == "python":
                import tree_sitter_python; return tree_sitter_python.language()
            elif lang == "javascript":
                import tree_sitter_javascript; return tree_sitter_javascript.language()
            elif lang == "typescript":
                import tree_sitter_typescript; return tree_sitter_typescript.language_typescript()
            elif lang == "tsx":
                import tree_sitter_typescript; return tree_sitter_typescript.language_tsx()
            elif lang == "java":
                import tree_sitter_java; return tree_sitter_java.language()
            elif lang == "go":
                import tree_sitter_go; return tree_sitter_go.language()
            elif lang == "rust":
                import tree_sitter_rust; return tree_sitter_rust.language()
            elif lang == "c":
                import tree_sitter_c; return tree_sitter_c.language()
            elif lang == "cpp":
                import tree_sitter_cpp; return tree_sitter_cpp.language()
            return None
        except ImportError:
            logger.debug("tree-sitter language %s not installed", lang)
            return None

    def _extract_blocks(self, node: Any, source: str, rel_path: str, repo_id: int, lang: str, blocks: list[dict[str, Any]], parent_class: str | None) -> None:
        class_types = {"class_definition", "class_declaration", "class_specifier", "struct_item", "struct_declaration", "interface_declaration", "type_declaration", "enum_declaration"}
        func_types = {"function_definition", "function_declaration", "method_definition", "method_declaration", "function_item", "arrow_function"}

        node_type = node.type
        if node_type in class_types:
            name = self._get_node_name(node)
            code = source[node.start_byte:node.end_byte]
            blocks.append({
                "file_path": rel_path, "language": lang, "block_type": "class",
                "block_name": name, "parent_class": parent_class,
                "start_line": node.start_point[0] + 1, "end_line": node.end_point[0] + 1,
                "signature": self._extract_signature(code), "code": code, "repo_id": repo_id,
            })
            for child in node.children:
                self._extract_blocks(child, source, rel_path, repo_id, lang, blocks, parent_class=name)
            return

        if node_type in func_types:
            name = self._get_node_name(node)
            code = source[node.start_byte:node.end_byte]
            blocks.append({
                "file_path": rel_path, "language": lang,
                "block_type": "method" if parent_class else "function",
                "block_name": name, "parent_class": parent_class,
                "start_line": node.start_point[0] + 1, "end_line": node.end_point[0] + 1,
                "signature": self._extract_signature(code), "code": code, "repo_id": repo_id,
            })
            return

        for child in node.children:
            self._extract_blocks(child, source, rel_path, repo_id, lang, blocks, parent_class=parent_class)

    def _get_node_name(self, node: Any) -> str:
        for child in node.children:
            if child.type in ("identifier", "name", "type_identifier", "property_identifier"):
                return child.text.decode("utf-8") if isinstance(child.text, bytes) else child.text
        return "anonymous"

    def _extract_signature(self, code: str) -> str:
        for line in code.split("\n"):
            s = line.strip()
            if s and not s.startswith("#") and not s.startswith("//"):
                return s[:500]
        return ""

    def _fallback_parse(self, source: str, rel_path: str, repo_id: int, ext: str) -> list[dict[str, Any]]:
        lang = LANG_MAP.get(ext, ext.lstrip("."))
        lines = source.split("\n")
        if len(lines) <= 100:
            return [{
                "file_path": rel_path, "language": lang, "block_type": "file",
                "block_name": os.path.basename(rel_path), "parent_class": None,
                "start_line": 1, "end_line": len(lines), "signature": "",
                "code": source, "repo_id": repo_id,
            }]
        blocks: list[dict[str, Any]] = []
        chunk_size = 80
        for i in range(0, len(lines), chunk_size):
            chunk = "\n".join(lines[i:i + chunk_size])
            blocks.append({
                "file_path": rel_path, "language": lang, "block_type": "chunk",
                "block_name": f"{os.path.basename(rel_path)}:{i + 1}", "parent_class": None,
                "start_line": i + 1, "end_line": min(i + chunk_size, len(lines)),
                "signature": "", "code": chunk, "repo_id": repo_id,
            })
        return blocks

    # ================================================================== #
    #  Block-level tracking (code_block table)                             #
    # ================================================================== #

    def _sync_code_blocks(
        self,
        repo_id: int,
        new_blocks: list[dict[str, Any]],
        commit_hash: str,
        changed_files: list[dict[str, str]] | None,
    ) -> list[dict[str, Any]]:
        """Sync parsed blocks with ``code_block`` table.

        For each parsed block, compute a ``content_hash``.  If a matching
        record already exists with the same hash and ``status=success``,
        skip it (no change).  Otherwise upsert as ``pending`` so it will
        be sent to LLM.

        Returns the list of block dicts that need LLM generation.
        """
        blocks_needing_llm: list[dict[str, Any]] = []

        with get_session() as session:
            # Load existing blocks for this repo (keyed by file:name:start)
            existing_rows = session.execute(
                select(CodeBlock).where(CodeBlock.repo_id == repo_id)
            ).scalars().all()
            existing_map: dict[str, CodeBlock] = {}
            for row in existing_rows:
                key = f"{row.file_path}:{row.block_name}:{row.start_line}"
                existing_map[key] = row

            # If files were deleted, clean up their code_block records
            if changed_files:
                deleted_paths = {c["path"] for c in changed_files if c["status"] == "D"}
                if deleted_paths:
                    session.execute(
                        delete(CodeBlock).where(
                            CodeBlock.repo_id == repo_id,
                            CodeBlock.file_path.in_(deleted_paths),
                        )
                    )

            for block in new_blocks:
                key = f"{block['file_path']}:{block.get('block_name', '')}:{block.get('start_line', 0)}"
                chash = _content_hash(block.get("code", ""))

                existing = existing_map.get(key)
                if existing and existing.content_hash == chash and existing.status in ("success", "generated"):
                    # Code hasn't changed and was successfully processed (or LLM
                    # already generated the description) → skip LLM generation.
                    continue

                if existing:
                    # Code changed or previous attempt failed → reset to pending
                    existing.content_hash = chash
                    existing.commit_hash = commit_hash
                    existing.start_line = block.get("start_line")
                    existing.end_line = block.get("end_line")
                    existing.signature = block.get("signature")
                    existing.status = "pending"
                    existing.error_message = None
                    block["_cb_id"] = existing.id
                else:
                    # New block
                    cb = CodeBlock(
                        repo_id=repo_id,
                        file_path=block["file_path"],
                        block_type=block["block_type"],
                        block_name=block.get("block_name"),
                        parent_class=block.get("parent_class"),
                        start_line=block.get("start_line"),
                        end_line=block.get("end_line"),
                        signature=block.get("signature"),
                        content_hash=chash,
                        commit_hash=commit_hash,
                        status="pending",
                    )
                    session.add(cb)
                    session.flush()
                    block["_cb_id"] = cb.id

                blocks_needing_llm.append(block)

        return blocks_needing_llm

    def _load_failed_blocks(
        self, repo_id: int, exclude_ids: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Load previously failed or orphaned pending blocks for retry.

        Blocks with ``status='failed'`` are always loaded (up to MAX_BLOCK_RETRY).
        Blocks with ``status='pending'`` are also loaded — these represent blocks
        from interrupted previous runs that were never completed.

        ``exclude_ids`` should contain ``_cb_id`` values of blocks already queued
        for generation by ``_sync_code_blocks`` in this run, to avoid duplicates.
        """
        with get_session() as session:
            query = select(CodeBlock).where(
                CodeBlock.repo_id == repo_id,
                CodeBlock.status.in_(["failed", "pending"]),
                CodeBlock.retry_count < MAX_BLOCK_RETRY,
            )
            if exclude_ids:
                query = query.where(CodeBlock.id.notin_(exclude_ids))

            rows = session.execute(query).scalars().all()

            skipped = 0
            blocks: list[dict[str, Any]] = []
            for row in rows:
                if row.retry_count >= MAX_BLOCK_RETRY:
                    skipped += 1
                    continue
                blocks.append({
                    "file_path": row.file_path,
                    "language": LANG_MAP.get(os.path.splitext(row.file_path)[1].lower(), ""),
                    "block_type": row.block_type,
                    "block_name": row.block_name,
                    "parent_class": row.parent_class,
                    "start_line": row.start_line,
                    "end_line": row.end_line,
                    "signature": row.signature or "",
                    "code": "",  # Code needs to be re-read from file
                    "repo_id": repo_id,
                    "_cb_id": row.id,
                    "_is_retry": True,
                    "_retry_count": row.retry_count,
                })

            if skipped:
                logger.info(
                    "Skipped %d blocks that exceeded max retry count (%d) for repo %d",
                    skipped, MAX_BLOCK_RETRY, repo_id,
                )
            if blocks:
                logger.info(
                    "Loaded %d blocks for retry (failed=%d, pending=%d) for repo %d",
                    len(blocks),
                    sum(1 for r in rows if r.status == "failed" and r.retry_count < MAX_BLOCK_RETRY),
                    sum(1 for r in rows if r.status == "pending" and r.retry_count < MAX_BLOCK_RETRY),
                    repo_id,
                )

            return blocks

    def _mark_block_generated(self, cb_id: int, description: str) -> None:
        """Mark a block as 'generated' and persist its LLM description.

        This is an intermediate checkpoint: the LLM call succeeded, but the
        description hasn't been embedded/written to Milvus yet.  If the
        process is interrupted before Milvus write, the description can be
        recovered on the next run without re-calling the LLM.
        """
        with get_session() as session:
            session.execute(
                update(CodeBlock).where(CodeBlock.id == cb_id).values(
                    status="generated",
                    description=description,
                    error_message=None,
                )
            )

    def _load_generated_blocks(
        self, repo_id: int, commit_hash: str,
    ) -> list[dict[str, Any]]:
        """Load blocks with ``status='generated'`` — LLM description cached in DB.

        These blocks had their LLM generation completed in a previous run but
        the process was interrupted before embedding + Milvus write.  We can
        skip the LLM call and go straight to the storage step.
        """
        with get_session() as session:
            rows = session.execute(
                select(CodeBlock).where(
                    CodeBlock.repo_id == repo_id,
                    CodeBlock.status == "generated",
                )
            ).scalars().all()

            entries: list[dict[str, Any]] = []
            for row in rows:
                if not row.description:
                    # Edge case: generated but description is empty — reset to pending
                    row.status = "pending"
                    continue
                entries.append({
                    "block": {
                        "file_path": row.file_path,
                        "block_type": row.block_type,
                        "block_name": row.block_name,
                        "parent_class": row.parent_class,
                        "start_line": row.start_line,
                        "end_line": row.end_line,
                        "language": LANG_MAP.get(
                            os.path.splitext(row.file_path)[1].lower(), "",
                        ),
                        "_cb_id": row.id,
                    },
                    "description": row.description,
                    "commit_hash": commit_hash,
                })

            if entries:
                logger.info(
                    "Loaded %d generated blocks (LLM cached) for repo %d",
                    len(entries), repo_id,
                )
            return entries

    def _mark_block_success(self, cb_id: int, milvus_id: str | None = None) -> None:
        with get_session() as session:
            session.execute(
                update(CodeBlock).where(CodeBlock.id == cb_id).values(
                    status="success", error_message=None, milvus_id=milvus_id,
                )
            )

    def _mark_block_failed(self, cb_id: int, error: str) -> None:
        with get_session() as session:
            session.execute(
                update(CodeBlock).where(CodeBlock.id == cb_id).values(
                    status="failed",
                    error_message=error[:500],
                    retry_count=CodeBlock.retry_count + 1,
                )
            )

    def _repair_orphaned_blocks(self, repo_id: int, kb_id: int) -> int:
        """Detect and reset 'success' blocks that have no data in Milvus.

        This can happen when a previous run was interrupted after LLM
        generation but before Milvus write completed.  Such blocks are
        marked ``success`` in MySQL but have no corresponding Milvus
        entry.

        If the block still has a ``description`` cached, it is reset to
        ``generated`` (so the LLM call can be skipped on the next run).
        Otherwise it is reset to ``pending``.

        Returns the number of blocks repaired.
        """
        with get_session() as session:
            success_blocks = session.execute(
                select(CodeBlock).where(
                    CodeBlock.repo_id == repo_id,
                    CodeBlock.status == "success",
                )
            ).scalars().all()

            if not success_blocks:
                return 0

            # Get topics present in Milvus for this KB
            milvus_topics = milvus_service.get_existing_topics(kb_id)

            to_generated: list[int] = []
            to_pending: list[int] = []
            for block in success_blocks:
                topic = f"{block.file_path}:{block.block_name or ''}"
                if topic not in milvus_topics:
                    if block.description:
                        to_generated.append(block.id)
                    else:
                        to_pending.append(block.id)

            repaired = len(to_generated) + len(to_pending)
            if to_generated:
                session.execute(
                    update(CodeBlock)
                    .where(CodeBlock.id.in_(to_generated))
                    .values(status="generated", error_message=None)
                )
            if to_pending:
                session.execute(
                    update(CodeBlock)
                    .where(CodeBlock.id.in_(to_pending))
                    .values(status="pending", retry_count=0, error_message=None)
                )

            if repaired:
                logger.info(
                    "Repaired %d orphaned blocks for repo %d "
                    "(reset %d to generated, %d to pending)",
                    repaired, repo_id, len(to_generated), len(to_pending),
                )
            else:
                logger.debug(
                    "No orphaned blocks found for repo %d (%d success blocks all in Milvus)",
                    repo_id, len(success_blocks),
                )

            return repaired

    # ================================================================== #
    #  Block-level knowledge generation                                    #
    # ================================================================== #

    def _generate_repo_map(self, repo_dir: str, files: list[str]) -> str:
        tree: dict[str, list[str]] = {}
        for fpath in files[:200]:
            rel = self._relative_path(fpath, repo_dir)
            parts = rel.split(os.sep)
            dir_path = "/".join(parts[:-1]) or "."
            tree.setdefault(dir_path, []).append(parts[-1])

        lines = ["项目结构："]
        for dir_path in sorted(tree.keys()):
            lines.append(f"  {dir_path}/")
            for fname in sorted(tree[dir_path])[:20]:
                lines.append(f"    {fname}")
            if len(tree[dir_path]) > 20:
                lines.append(f"    ... 还有 {len(tree[dir_path]) - 20} 个文件")
        return "\n".join(lines[:100])

    def _generate_block_knowledge(
        self,
        blocks: list[dict[str, Any]],
        repo_map: str,
        kb_id: int,
        repo_id: int,
        commit_hash: str,
        notify_fn: Any = None,
        check_cancelled_fn: Any = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Generate knowledge for blocks via LLM.

        Returns ``(success_entries, failed_count)``.
        """
        entries: list[dict[str, Any]] = []
        failed = 0
        total = len(blocks)
        repo_dir = os.path.join(config.REPOS_BASE_DIR, str(repo_id))

        def process_block(block: dict[str, Any]) -> dict[str, Any] | None:
            cb_id = block.get("_cb_id")
            block_label = f"{block.get('file_path')}:{block.get('block_name', '?')} (lines {block.get('start_line')}-{block.get('end_line')})"
            is_retry = block.get("_is_retry", False)
            try:
                # For retry blocks, re-read code from file
                code = block.get("code", "")
                if not code and is_retry:
                    fpath = os.path.join(repo_dir, block["file_path"])
                    if os.path.exists(fpath):
                        with open(fpath, "r", errors="ignore") as f:
                            source = f.read()
                        source = _redact_sensitive(source)
                        start = (block.get("start_line") or 1) - 1
                        end = block.get("end_line") or len(source.split("\n"))
                        code = "\n".join(source.split("\n")[start:end])
                    else:
                        logger.warning(
                            "Block retry failed - file not found: %s (block: %s)",
                            fpath, block_label,
                        )

                if not code or not code.strip():
                    # Empty code block — no point retrying, mark as permanently failed
                    logger.info(
                        "Skipping empty code block [%s] (file_exists=%s, is_retry=%s)",
                        block_label,
                        os.path.exists(os.path.join(repo_dir, block["file_path"])),
                        is_retry,
                    )
                    if cb_id:
                        with get_session() as session:
                            session.execute(
                                update(CodeBlock).where(CodeBlock.id == cb_id).values(
                                    status="failed",
                                    error_message="Empty code block — skipped permanently",
                                    retry_count=MAX_BLOCK_RETRY,  # prevent further retries
                                )
                            )
                    return None

                description = llm_service.generate_code_knowledge(
                    repo_map=repo_map,
                    file_path=block["file_path"],
                    block_type=block["block_type"],
                    block_name=block.get("block_name") or "unknown",
                    language=block.get("language", ""),
                    code=code[:8000],
                )

                # Persist description to DB immediately so it survives
                # process interruption.  Final "success" is set after
                # Milvus write in _store_block_knowledge.
                if cb_id:
                    self._mark_block_generated(cb_id, description)

                return {
                    "block": block,
                    "description": description,
                    "commit_hash": commit_hash,
                }
            except CancelledError:
                raise
            except Exception as e:
                retry_info = f" (retry #{block.get('_retry_count', 0) + 1}/{MAX_BLOCK_RETRY})" if is_retry else ""
                logger.warning(
                    "Failed to generate knowledge for block [%s]%s: %s",
                    block_label, retry_info, e,
                    exc_info=True,
                )
                if cb_id:
                    self._mark_block_failed(cb_id, str(e))
                return None

        throttle = _ProgressThrottle(total)
        with ThreadPoolExecutor(max_workers=config.LLM_CONCURRENCY_BLOCK) as executor:
            futures = {executor.submit(process_block, b): b for b in blocks}
            done_count = 0
            for future in as_completed(futures):
                # Check cancellation before collecting each result
                if check_cancelled_fn:
                    check_cancelled_fn()
                result = future.result()
                if result:
                    entries.append(result)
                else:
                    failed += 1
                done_count += 1
                if notify_fn and throttle.should_notify(done_count):
                    pct = done_count * 100 // total
                    eta = throttle.eta(done_count)
                    eta_part = f"，预计{eta}" if eta else ""
                    notify_fn(f"🤖 正在生成知识摘要（{pct}%{eta_part}）...", progress=True)

        if notify_fn and total:
            notify_fn("✅ 知识摘要生成完成", progress=True, done=True)

        return entries, failed

    def _store_block_knowledge(self, kb_id: int, entries: list[dict[str, Any]]) -> None:
        """Embed and store block-level knowledge in Milvus.

        Before inserting, delete old Milvus entries for the same blocks
        (identified by file_path) to avoid duplicates.
        """
        # Collect file paths that need cleaning
        file_paths = {e["block"]["file_path"] for e in entries}
        for fpath in file_paths:
            try:
                milvus_service.delete_by_expr(
                    kb_id, f'file_path == "{fpath}" and block_type != "module_summary" and block_type != "repo_summary"',
                )
            except Exception as e:
                logger.warning("Failed to clean old block knowledge for %s: %s", fpath, e)

        # Split long descriptions into segments via LLM, then embed all at once.
        expanded: list[tuple[dict[str, Any], str]] = []  # (entry, segment)
        for entry in entries:
            desc = entry["description"]
            if len(desc) > 4500:
                segments = llm_service.split_long_content(desc, max_length=4500)
                for seg in segments:
                    expanded.append((entry, seg))
            else:
                expanded.append((entry, desc))

        texts = [seg for _, seg in expanded]
        vectors = embedding_service.embed_batch(texts)

        milvus_entries: list[dict[str, Any]] = []
        for vec, (entry, seg) in zip(vectors, expanded):
            block = entry["block"]
            milvus_entries.append({
                "vector": vec,
                "topic": f"{block['file_path']}:{block.get('block_name', '')}",
                "content": seg,
                "source": "code",
                "source_detail": f"{block['file_path']}:{block.get('start_line', 0)}-{block.get('end_line', 0)}",
                "certainty": "confirmed",
                "kb_id": kb_id,
                "last_updated_at": int(time.time()),
                "last_referenced_at": int(time.time()),
                "file_path": block["file_path"],
                "language": block.get("language", ""),
                "block_type": block["block_type"],
                "block_name": block.get("block_name") or "",
                "commit_hash": entry["commit_hash"],
            })

        milvus_service.insert_knowledge_dicts(kb_id, milvus_entries)

        # Mark blocks as "success" AFTER Milvus write completes.
        # This prevents orphaned blocks (marked success but not in Milvus)
        # when the process is interrupted between LLM generation and Milvus write.
        success_cb_ids = [e["block"]["_cb_id"] for e in entries if e["block"].get("_cb_id")]
        if success_cb_ids:
            with get_session() as session:
                session.execute(
                    update(CodeBlock)
                    .where(CodeBlock.id.in_(success_cb_ids))
                    .values(status="success", error_message=None)
                )
            logger.info("Marked %d blocks as success after Milvus write", len(success_cb_ids))

    # ================================================================== #
    #  Module-level summaries                                              #
    # ================================================================== #

    def _get_affected_modules(
        self,
        repo_id: int,
        blocks_processed: list[dict[str, Any]],
        changed_files: list[dict[str, str]] | None,
    ) -> set[str]:
        """Determine which module directories need summary regeneration."""
        affected: set[str] = set()

        # Modules affected by newly generated blocks
        for block in blocks_processed:
            affected.add(_module_path(block["file_path"]))

        # Modules affected by deleted files
        if changed_files:
            for cf in changed_files:
                if cf["status"] == "D":
                    affected.add(_module_path(cf["path"]))

        return affected

    def _regenerate_module_summaries(
        self,
        kb_id: int,
        repo_id: int,
        affected_dirs: set[str],
        repo_map: str,
        notify_fn: Any = None,
        check_cancelled_fn: Any = None,
    ) -> int:
        """Regenerate module summaries for affected directories (concurrent).

        Returns count of successfully updated modules.
        """
        updated = 0
        dirs_list = sorted(affected_dirs)

        def _process_module(module_dir: str) -> bool:
            """Process a single module. Returns True on success."""
            try:
                with get_session() as session:
                    blocks = session.execute(
                        select(CodeBlock).where(
                            CodeBlock.repo_id == repo_id,
                            CodeBlock.status == "success",
                            CodeBlock.file_path.like(f"{module_dir}/%") if module_dir != "." else CodeBlock.file_path.not_like("%/%"),
                        )
                    ).scalars().all()

                if not blocks:
                    try:
                        milvus_service.delete_by_expr(
                            kb_id, f'block_type == "module_summary" and file_path == "{module_dir}"',
                        )
                    except Exception:
                        pass
                    return False

                block_summaries = self._collect_block_descriptions(kb_id, module_dir, blocks)

                summary = llm_service.generate_module_summary(
                    repo_map=repo_map,
                    module_path=module_dir,
                    block_summaries=block_summaries,
                )

                try:
                    milvus_service.delete_by_expr(
                        kb_id, f'block_type == "module_summary" and file_path == "{module_dir}"',
                    )
                except Exception:
                    pass

                segments = llm_service.split_long_content(summary, max_length=4500)
                seg_texts = segments
                seg_vecs = embedding_service.embed_batch(seg_texts) if len(seg_texts) > 1 else [embedding_service.embed(seg_texts[0])]
                seg_entries = []
                for sv, st in zip(seg_vecs, seg_texts):
                    seg_entries.append({
                    "vector": sv,
                    "topic": f"模块摘要：{module_dir}",
                    "content": st,
                    "source": "code",
                    "source_detail": f"module:{module_dir}",
                    "certainty": "confirmed",
                    "kb_id": kb_id,
                    "last_updated_at": int(time.time()),
                    "last_referenced_at": int(time.time()),
                    "file_path": module_dir,
                    "language": "",
                    "block_type": "module_summary",
                    "block_name": module_dir,
                    "commit_hash": "",
                    })
                milvus_service.insert_knowledge_dicts(kb_id, seg_entries)
                return True

            except Exception as e:
                logger.warning("Failed to generate module summary for %s: %s", module_dir, e)
                return False

        with ThreadPoolExecutor(max_workers=config.LLM_CONCURRENCY_MODULE) as executor:
            futures = {executor.submit(_process_module, d): d for d in dirs_list}
            done_count = 0
            throttle = _ProgressThrottle(len(dirs_list))
            for future in as_completed(futures):
                if check_cancelled_fn:
                    check_cancelled_fn()
                if future.result():
                    updated += 1
                done_count += 1
                if notify_fn and throttle.should_notify(done_count):
                    pct = done_count * 100 // len(dirs_list)
                    eta = throttle.eta(done_count)
                    eta_part = f"，预计{eta}" if eta else ""
                    notify_fn(f"📝 正在更新模块摘要（{pct}%{eta_part}）...", progress=True)

        if notify_fn and dirs_list:
            notify_fn("✅ 模块摘要更新完成", progress=True, done=True)

        return updated

    def _collect_block_descriptions(
        self, kb_id: int, module_dir: str, blocks: list[Any],
    ) -> str:
        """Collect existing block-level descriptions from Milvus for a module."""
        parts: list[str] = []
        for block in blocks[:50]:  # Limit to avoid huge prompts
            parts.append(
                f"- [{block.block_type}] {block.file_path}:{block.block_name}"
                f" (L{block.start_line}-{block.end_line})"
            )

        # Also try to fetch actual content from Milvus
        try:
            results = milvus_service.get_all_entries(
                kb_id, limit=50,
                output_fields=["topic", "content", "file_path", "block_type"],
            )
            for r in results:
                fp = r.get("file_path", "")
                bt = r.get("block_type", "")
                if bt in ("module_summary", "repo_summary"):
                    continue
                if module_dir == ".":
                    if "/" in fp:
                        continue
                elif not fp.startswith(module_dir + "/"):
                    continue
                content = r.get("content", "")
                if content:
                    parts.append(f"\n### {r.get('topic', fp)}\n{content[:500]}")
        except Exception:
            pass

        return "\n".join(parts) if parts else "（无已生成的代码块描述）"

    # ================================================================== #
    #  Repo-level overview                                                 #
    # ================================================================== #

    def _regenerate_repo_overview(self, kb_id: int, repo_id: int, repo_map: str) -> None:
        """Regenerate the repository-level overview."""
        try:
            # Collect all module summaries from Milvus
            results = milvus_service.get_all_entries(
                kb_id, limit=200,
                output_fields=["topic", "content", "block_type", "file_path"],
            )
            module_parts: list[str] = []
            for r in results:
                if r.get("block_type") == "module_summary":
                    module_parts.append(f"### {r.get('file_path', '')}\n{r.get('content', '')}")

            module_summaries = "\n\n".join(module_parts) if module_parts else "（暂无模块摘要）"

            overview = llm_service.generate_repo_overview(
                repo_map=repo_map,
                module_summaries=module_summaries,
            )

            # Delete old overview, insert new
            try:
                milvus_service.delete_by_expr(kb_id, 'block_type == "repo_summary"')
            except Exception:
                pass

            segments = llm_service.split_long_content(overview, max_length=4500)
            seg_vecs = embedding_service.embed_batch(segments) if len(segments) > 1 else [embedding_service.embed(segments[0])]
            overview_entries = []
            for sv, st in zip(seg_vecs, segments):
                overview_entries.append({
                    "vector": sv,
                    "topic": "仓库全局概览",
                    "content": st,
                    "source": "code",
                    "source_detail": "repo:overview",
                    "certainty": "confirmed",
                    "kb_id": kb_id,
                    "last_updated_at": int(time.time()),
                    "last_referenced_at": int(time.time()),
                    "file_path": ".",
                    "language": "",
                    "block_type": "repo_summary",
                    "block_name": "overview",
                    "commit_hash": "",
                })
            milvus_service.insert_knowledge_dicts(kb_id, overview_entries)

        except Exception as e:
            logger.warning("Failed to generate repo overview: %s", e)

    # ================================================================== #
    #  Cleanup helpers                                                     #
    # ================================================================== #

    def _delete_file_knowledge(self, kb_id: int, repo_id: int, file_paths: list[str]) -> None:
        for fpath in file_paths:
            try:
                milvus_service.delete_by_expr(kb_id, f'file_path == "{fpath}"')
            except Exception as e:
                logger.warning("Failed to delete knowledge for %s: %s", fpath, e)
        if file_paths:
            with get_session() as session:
                session.execute(
                    delete(CodeBlock).where(
                        CodeBlock.repo_id == repo_id,
                        CodeBlock.file_path.in_(file_paths),
                    )
                )

    # ================================================================== #
    #  Task log helpers                                                    #
    # ================================================================== #

    def _update_repo_commit(self, repo_id: int, commit_hash: str) -> None:
        with get_session() as session:
            session.execute(
                update(CodeRepo).where(CodeRepo.id == repo_id).values(
                    last_commit_hash=commit_hash, last_analyzed_at=datetime.utcnow(),
                )
            )

    def _start_task_log(self, kb_id: int | None, task_type: str) -> int:
        if not kb_id:
            return 0
        with get_session() as session:
            log = SummarizeTaskLog(
                kb_id=kb_id, task_type=task_type,
                status="running", started_at=datetime.utcnow(),
            )
            session.add(log)
            session.flush()
            return log.id

    def _finish_task_log(self, log_id: int, status: str, stats: dict[str, int], error_msg: str | None = None) -> None:
        if not log_id:
            return
        with get_session() as session:
            session.execute(
                update(SummarizeTaskLog).where(SummarizeTaskLog.id == log_id).values(
                    status=status,
                    message_count=stats.get("files_parsed", 0),
                    new_knowledge_count=stats.get("knowledge_generated", 0),
                    error_message=error_msg,
                    finished_at=datetime.utcnow(),
                )
            )


repo_analyzer = RepoAnalyzer()
