"""Code repository analysis – Git operations, AST parsing, and knowledge generation."""

import logging
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import pathspec
from sqlalchemy import delete, select, update

from app.config import config
from app.db.models import CodeBlock, CodeRepo, KnowledgeBase, SummarizeTaskLog
from app.db.session import get_session
from app.services.embedding_service import embedding_service
from app.services.llm_service import llm_service
from app.services.milvus_service import milvus_service

logger = logging.getLogger(__name__)

# File extensions to process
ALLOWED_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs",
    ".cs", ".cpp", ".c", ".h", ".rb", ".php", ".swift", ".kt",
    ".scala", ".vue", ".sql", ".proto", ".graphql",
}

# Directories to always skip
BLACKLIST_DIRS = {
    "node_modules", "vendor", "dist", "build", "target", ".git",
    "__pycache__", ".tox", ".nox", ".eggs", ".mypy_cache",
    ".pytest_cache", "venv", ".venv", "env",
}

MAX_FILE_SIZE = 100 * 1024  # 100KB
MAX_LINE_LENGTH = 500  # Skip minified files

# Sensitive info patterns
SENSITIVE_PATTERNS = [
    re.compile(r"""(password|passwd|pwd)\s*[=:]\s*['"][^'"]+['"]""", re.IGNORECASE),
    re.compile(r"""(api_key|apikey|api-key)\s*[=:]\s*['"][^'"]+['"]""", re.IGNORECASE),
    re.compile(r"""(secret|secret_key)\s*[=:]\s*['"][^'"]+['"]""", re.IGNORECASE),
    re.compile(r"""(token|access_token|auth_token)\s*[=:]\s*['"][^'"]+['"]""", re.IGNORECASE),
    re.compile(r"""Bearer\s+[A-Za-z0-9\-._~+/]+=*""", re.IGNORECASE),
    re.compile(r"""(AKIA|ABIA|ACCA|ASIA)[A-Z0-9]{16}"""),  # AWS key pattern
]

# Language map for tree-sitter
LANG_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cs": "c_sharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
}


def _redact_sensitive(text: str) -> str:
    """Replace sensitive information with [REDACTED]."""
    for pattern in SENSITIVE_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


class RepoAnalyzer:
    """Full pipeline for analyzing a code repository."""

    def __init__(self) -> None:
        self._ts_parsers: dict[str, Any] = {}

    def analyze_repo(
        self,
        repo_id: int,
        chat_id: str | None = None,
        progress_callback: Any = None,
    ) -> dict[str, int]:
        """Full analysis of a code repository.

        Args:
            repo_id: Database ID of the code_repo record.
            chat_id: For progress reporting.
            progress_callback: Callable(chat_id, message) for progress updates.

        Returns:
            Stats dict {files_parsed, blocks_found, knowledge_generated}.
        """
        stats = {"files_parsed": 0, "blocks_found": 0, "knowledge_generated": 0}

        with get_session() as session:
            repo = session.execute(
                select(CodeRepo).where(CodeRepo.id == repo_id)
            ).scalar_one_or_none()
            if not repo:
                logger.error("Code repo %d not found", repo_id)
                return stats

            kb_id = repo.kb_id
            git_url = repo.git_url
            branch = repo.default_branch or "main"
            last_commit = repo.last_commit_hash

        repo_dir = os.path.join(config.REPOS_BASE_DIR, str(repo_id))
        task_log_id = self._start_task_log(kb_id, "code")

        try:
            # Step 1: Clone or pull
            if progress_callback and chat_id:
                progress_callback(chat_id, "📦 正在克隆仓库...")

            is_new = not os.path.exists(os.path.join(repo_dir, ".git"))
            if is_new:
                self._git_clone(git_url, repo_dir, branch)
            else:
                self._git_pull(repo_dir, branch)

            current_commit = self._get_current_commit(repo_dir)

            # Determine changed files (or all files if first run)
            if last_commit and not is_new:
                changed_files = self._get_changed_files(repo_dir, last_commit, f"origin/{branch}")
            else:
                changed_files = None  # Will process all files

            # Step 2: Scan and filter files
            if progress_callback and chat_id:
                progress_callback(chat_id, "🔍 正在解析代码结构...")

            all_files = self._scan_files(repo_dir)
            if changed_files is not None:
                # Filter to only changed/new files
                files_to_process = [f for f in all_files if self._relative_path(f, repo_dir) in {c["path"] for c in changed_files if c["status"] in ("A", "M")}]
                # Handle deleted files
                deleted_paths = [c["path"] for c in changed_files if c["status"] == "D"]
                self._delete_file_knowledge(kb_id, repo_id, deleted_paths)
            else:
                files_to_process = all_files

            total_files = len(files_to_process)

            # Step 3: Parse files and extract code blocks
            all_blocks: list[dict[str, Any]] = []
            for i, fpath in enumerate(files_to_process):
                if progress_callback and chat_id and (i + 1) % 10 == 0:
                    progress_callback(chat_id, f"🔍 正在解析代码结构（已解析 {i + 1}/{total_files} 个文件）...")

                blocks = self._parse_file(fpath, repo_dir, repo_id)
                all_blocks.extend(blocks)
                stats["files_parsed"] += 1

            stats["blocks_found"] = len(all_blocks)

            if not all_blocks:
                self._update_repo_commit(repo_id, current_commit)
                self._finish_task_log(task_log_id, "success", stats)
                return stats

            # Step 4: Generate repo map
            repo_map = self._generate_repo_map(repo_dir, all_files)

            # Step 5: LLM knowledge generation
            if progress_callback and chat_id:
                progress_callback(chat_id, f"🤖 正在生成知识摘要（共 {len(all_blocks)} 个代码块）...")

            knowledge_entries = self._generate_knowledge(
                all_blocks, repo_map, kb_id, repo_id, current_commit,
                chat_id=chat_id,
                progress_callback=progress_callback,
            )
            stats["knowledge_generated"] = len(knowledge_entries)

            # Step 6: Store in Milvus
            if progress_callback and chat_id:
                progress_callback(chat_id, "💾 正在写入向量数据库...")

            if knowledge_entries:
                # Delete old knowledge for modified files
                if changed_files is not None:
                    modified_paths = [c["path"] for c in changed_files if c["status"] == "M"]
                    self._delete_file_knowledge(kb_id, repo_id, modified_paths)

                self._store_knowledge(kb_id, knowledge_entries)

            # Update repo commit hash
            self._update_repo_commit(repo_id, current_commit)
            self._finish_task_log(task_log_id, "success", stats)

            if progress_callback and chat_id:
                progress_callback(
                    chat_id,
                    f"✅ 代码库分析完成！共解析 {stats['files_parsed']} 个文件，"
                    f"生成 {stats['knowledge_generated']} 条知识。",
                )

        except Exception as e:
            logger.exception("Repo analysis failed for repo_id=%d", repo_id)
            self._finish_task_log(task_log_id, "failed", stats, str(e))
            if progress_callback and chat_id:
                progress_callback(
                    chat_id,
                    f"⚠️ 代码库分析部分失败：{str(e)[:100]}，"
                    f"已成功处理 {stats['files_parsed']}/{total_files} 个文件。",
                )

        return stats

    def check_and_update(self, repo_id: int) -> dict[str, int]:
        """Check for remote updates and run incremental analysis."""
        return self.analyze_repo(repo_id)

    # ------------------------------------------------------------------
    # Git operations
    # ------------------------------------------------------------------

    @staticmethod
    def _build_clone_url(repo_path: str) -> str:
        """Build a full authenticated clone URL.

        repo_path is either a full URL or a short path like ``org/repo``.
        When GIT_BASE_URL is configured, short paths are expanded to
        ``{GIT_BASE_URL}/{repo_path}.git``.  The PAT is injected into
        the HTTPS URL for authentication.
        """
        url = repo_path
        # Expand short path → full URL
        if not url.startswith("http://") and not url.startswith("https://"):
            base = config.GIT_BASE_URL.rstrip("/")
            url = f"{base}/{url.strip('/')}.git"

        pat = config.GIT_PAT
        if not pat:
            return url
        if url.startswith("https://"):
            return url.replace("https://", f"https://{pat}@", 1)
        if url.startswith("http://"):
            return url.replace("http://", f"http://{pat}@", 1)
        return url

    def _git_clone(self, url: str, dest: str, branch: str) -> None:
        os.makedirs(dest, exist_ok=True)
        auth_url = self._build_clone_url(url)
        subprocess.run(
            ["git", "clone", "-b", branch, auth_url, dest],
            check=True,
            capture_output=True,
            timeout=600,
        )
        logger.info("Cloned %s to %s", url, dest)

    def _git_pull(self, repo_dir: str, branch: str) -> None:
        subprocess.run(
            ["git", "fetch", "origin"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            timeout=300,
        )
        subprocess.run(
            ["git", "pull", "origin", branch],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            timeout=300,
        )

    def _get_current_commit(self, repo_dir: str) -> str:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def _get_changed_files(self, repo_dir: str, from_commit: str, to_ref: str) -> list[dict[str, str]]:
        result = subprocess.run(
            ["git", "diff", f"{from_commit}..{to_ref}", "--name-status"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
        )
        changes: list[dict[str, str]] = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                changes.append({"status": parts[0][0], "path": parts[-1]})
        return changes

    # ------------------------------------------------------------------
    # File scanning and filtering
    # ------------------------------------------------------------------

    def _scan_files(self, repo_dir: str) -> list[str]:
        """Scan repository for processable files."""
        # Load .gitignore
        gitignore_path = os.path.join(repo_dir, ".gitignore")
        spec = None
        if os.path.exists(gitignore_path):
            with open(gitignore_path) as f:
                spec = pathspec.PathSpec.from_lines("gitwildmatch", f)

        result: list[str] = []
        for root, dirs, files in os.walk(repo_dir):
            # Skip blacklisted directories
            dirs[:] = [d for d in dirs if d not in BLACKLIST_DIRS]

            for fname in files:
                fpath = os.path.join(root, fname)
                rel_path = os.path.relpath(fpath, repo_dir)

                # Extension check
                ext = os.path.splitext(fname)[1].lower()
                if ext not in ALLOWED_EXTENSIONS:
                    continue

                # Gitignore check
                if spec and spec.match_file(rel_path):
                    continue

                # Size check
                try:
                    if os.path.getsize(fpath) > MAX_FILE_SIZE:
                        continue
                except OSError:
                    continue

                # Minified check
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

    # ------------------------------------------------------------------
    # AST parsing
    # ------------------------------------------------------------------

    def _parse_file(self, fpath: str, repo_dir: str, repo_id: int) -> list[dict[str, Any]]:
        """Parse a file into code blocks using tree-sitter."""
        ext = os.path.splitext(fpath)[1].lower()
        lang = LANG_MAP.get(ext)
        rel_path = self._relative_path(fpath, repo_dir)

        try:
            with open(fpath, "r", errors="ignore") as f:
                source = f.read()
        except Exception:
            return []

        source = _redact_sensitive(source)

        if lang:
            blocks = self._parse_with_treesitter(source, lang, rel_path, repo_id)
            if blocks:
                return blocks

        # Fallback: treat the whole file as one block
        return self._fallback_parse(source, rel_path, repo_id, ext)

    def _parse_with_treesitter(
        self, source: str, lang: str, rel_path: str, repo_id: int,
    ) -> list[dict[str, Any]]:
        """Extract code blocks using tree-sitter AST."""
        try:
            import tree_sitter

            parser_key = lang
            if parser_key not in self._ts_parsers:
                ts_lang = self._load_ts_language(lang)
                if ts_lang is None:
                    return []
                parser = tree_sitter.Parser(ts_lang)
                self._ts_parsers[parser_key] = parser
            else:
                parser = self._ts_parsers[parser_key]

            tree = parser.parse(source.encode("utf-8"))
            blocks: list[dict[str, Any]] = []

            self._extract_blocks(tree.root_node, source, rel_path, repo_id, lang, blocks, parent_class=None)
            return blocks

        except Exception as e:
            logger.debug("Tree-sitter parse failed for %s: %s", rel_path, e)
            return []

    def _load_ts_language(self, lang: str) -> Any:
        """Load tree-sitter language module."""
        try:
            if lang == "python":
                import tree_sitter_python
                return tree_sitter_python.language()
            elif lang == "javascript":
                import tree_sitter_javascript
                return tree_sitter_javascript.language()
            elif lang == "typescript":
                import tree_sitter_typescript
                return tree_sitter_typescript.language_typescript()
            elif lang == "java":
                import tree_sitter_java
                return tree_sitter_java.language()
            elif lang == "go":
                import tree_sitter_go
                return tree_sitter_go.language()
            elif lang == "rust":
                import tree_sitter_rust
                return tree_sitter_rust.language()
            elif lang == "c":
                import tree_sitter_c
                return tree_sitter_c.language()
            elif lang == "cpp":
                import tree_sitter_cpp
                return tree_sitter_cpp.language()
            else:
                return None
        except ImportError:
            logger.debug("tree-sitter language %s not installed", lang)
            return None

    def _extract_blocks(
        self,
        node: Any,
        source: str,
        rel_path: str,
        repo_id: int,
        lang: str,
        blocks: list[dict[str, Any]],
        parent_class: str | None,
    ) -> None:
        """Recursively extract code blocks from AST."""
        # Node types that represent meaningful code blocks
        class_types = {
            "class_definition", "class_declaration", "class_specifier",
            "struct_item", "struct_declaration", "interface_declaration",
            "type_declaration", "enum_declaration",
        }
        func_types = {
            "function_definition", "function_declaration", "method_definition",
            "method_declaration", "function_item", "arrow_function",
        }
        const_types = {
            "const_declaration", "variable_declaration", "assignment",
            "const_item", "static_item",
        }

        node_type = node.type

        if node_type in class_types:
            name = self._get_node_name(node)
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            code = source[node.start_byte:node.end_byte]
            signature = self._extract_signature(code)

            blocks.append({
                "file_path": rel_path,
                "language": lang,
                "block_type": "class",
                "block_name": name,
                "parent_class": parent_class,
                "start_line": start_line,
                "end_line": end_line,
                "signature": signature,
                "code": code,
                "repo_id": repo_id,
            })

            # Recurse into class body for methods
            for child in node.children:
                self._extract_blocks(child, source, rel_path, repo_id, lang, blocks, parent_class=name)
            return

        if node_type in func_types:
            name = self._get_node_name(node)
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            code = source[node.start_byte:node.end_byte]
            signature = self._extract_signature(code)

            block_type = "method" if parent_class else "function"
            blocks.append({
                "file_path": rel_path,
                "language": lang,
                "block_type": block_type,
                "block_name": name,
                "parent_class": parent_class,
                "start_line": start_line,
                "end_line": end_line,
                "signature": signature,
                "code": code,
                "repo_id": repo_id,
            })
            return

        # Recurse into children
        for child in node.children:
            self._extract_blocks(child, source, rel_path, repo_id, lang, blocks, parent_class=parent_class)

    def _get_node_name(self, node: Any) -> str:
        """Extract the name identifier from an AST node."""
        for child in node.children:
            if child.type in ("identifier", "name", "type_identifier", "property_identifier"):
                return child.text.decode("utf-8") if isinstance(child.text, bytes) else child.text
        return "anonymous"

    def _extract_signature(self, code: str) -> str:
        """Extract the first line (signature) of a code block."""
        lines = code.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and not stripped.startswith("//"):
                return stripped[:500]
        return ""

    def _fallback_parse(self, source: str, rel_path: str, repo_id: int, ext: str) -> list[dict[str, Any]]:
        """Fallback: split file into chunks if tree-sitter isn't available."""
        lang = LANG_MAP.get(ext, ext.lstrip("."))
        lines = source.split("\n")
        if len(lines) <= 100:
            return [{
                "file_path": rel_path,
                "language": lang,
                "block_type": "file",
                "block_name": os.path.basename(rel_path),
                "parent_class": None,
                "start_line": 1,
                "end_line": len(lines),
                "signature": "",
                "code": source,
                "repo_id": repo_id,
            }]

        # Split into ~80-line chunks
        blocks: list[dict[str, Any]] = []
        chunk_size = 80
        for i in range(0, len(lines), chunk_size):
            chunk = "\n".join(lines[i:i + chunk_size])
            blocks.append({
                "file_path": rel_path,
                "language": lang,
                "block_type": "chunk",
                "block_name": f"{os.path.basename(rel_path)}:{i + 1}",
                "parent_class": None,
                "start_line": i + 1,
                "end_line": min(i + chunk_size, len(lines)),
                "signature": "",
                "code": chunk,
                "repo_id": repo_id,
            })
        return blocks

    # ------------------------------------------------------------------
    # Knowledge generation
    # ------------------------------------------------------------------

    def _generate_repo_map(self, repo_dir: str, files: list[str]) -> str:
        """Generate a lightweight repo structure overview."""
        tree: dict[str, list[str]] = {}
        for fpath in files[:200]:  # Limit to avoid huge maps
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

    def _generate_knowledge(
        self,
        blocks: list[dict[str, Any]],
        repo_map: str,
        kb_id: int,
        repo_id: int,
        commit_hash: str,
        chat_id: str | None = None,
        progress_callback: Any = None,
    ) -> list[dict[str, Any]]:
        """Generate knowledge descriptions for code blocks using LLM."""
        entries: list[dict[str, Any]] = []
        total = len(blocks)

        # Process in batches with limited concurrency
        def process_block(idx: int, block: dict[str, Any]) -> dict[str, Any] | None:
            try:
                description = llm_service.generate_code_knowledge(
                    repo_map=repo_map,
                    file_path=block["file_path"],
                    block_type=block["block_type"],
                    block_name=block["block_name"] or "unknown",
                    language=block["language"],
                    code=block["code"][:8000],  # Limit code length
                )

                # Save code block record
                self._save_code_block(block, commit_hash)

                return {
                    "block": block,
                    "description": description,
                    "commit_hash": commit_hash,
                }
            except Exception as e:
                logger.warning("Failed to generate knowledge for %s:%s: %s",
                               block["file_path"], block["block_name"], e)
                return None

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(process_block, i, b): i
                for i, b in enumerate(blocks)
            }
            done_count = 0
            for future in as_completed(futures):
                result = future.result()
                if result:
                    entries.append(result)
                done_count += 1
                if progress_callback and chat_id and done_count % 20 == 0:
                    progress_callback(
                        chat_id,
                        f"🤖 正在生成知识摘要（已完成 {done_count}/{total} 个代码块）...",
                    )

        return entries

    def _store_knowledge(self, kb_id: int, entries: list[dict[str, Any]]) -> None:
        """Embed and store knowledge entries in Milvus."""
        texts = [e["description"] for e in entries]
        vectors = embedding_service.embed_batch(texts)

        milvus_entries: list[dict[str, Any]] = []
        for vec, entry in zip(vectors, entries):
            block = entry["block"]
            milvus_entries.append({
                "vector": vec,
                "topic": f"{block['file_path']}:{block['block_name']}",
                "content": entry["description"][:5000],
                "source": "code",
                "source_detail": f"{block['file_path']}:{block['start_line']}-{block['end_line']}",
                "certainty": "confirmed",
                "kb_id": kb_id,
                "last_updated_at": int(time.time()),
                "last_referenced_at": int(time.time()),
                # Dynamic fields for code knowledge
                "file_path": block["file_path"],
                "language": block["language"],
                "block_type": block["block_type"],
                "block_name": block["block_name"] or "",
                "commit_hash": entry["commit_hash"],
            })

        milvus_service.insert_knowledge_dicts(kb_id, milvus_entries)

    def _delete_file_knowledge(self, kb_id: int, repo_id: int, file_paths: list[str]) -> None:
        """Delete knowledge entries for specific files from Milvus."""
        for fpath in file_paths:
            try:
                milvus_service.delete_by_expr(kb_id, f'file_path == "{fpath}"')
            except Exception as e:
                logger.warning("Failed to delete knowledge for %s: %s", fpath, e)

        # Also clean up code_block records
        if file_paths:
            with get_session() as session:
                session.execute(
                    delete(CodeBlock).where(
                        CodeBlock.repo_id == repo_id,
                        CodeBlock.file_path.in_(file_paths),
                    )
                )

    def _save_code_block(self, block: dict[str, Any], commit_hash: str) -> None:
        """Save code block parsing record to MySQL."""
        try:
            with get_session() as session:
                cb = CodeBlock(
                    repo_id=block["repo_id"],
                    file_path=block["file_path"],
                    block_type=block["block_type"],
                    block_name=block.get("block_name"),
                    parent_class=block.get("parent_class"),
                    start_line=block.get("start_line"),
                    end_line=block.get("end_line"),
                    signature=block.get("signature"),
                    commit_hash=commit_hash,
                )
                session.add(cb)
        except Exception:
            pass  # Non-critical

    def _update_repo_commit(self, repo_id: int, commit_hash: str) -> None:
        with get_session() as session:
            session.execute(
                update(CodeRepo)
                .where(CodeRepo.id == repo_id)
                .values(last_commit_hash=commit_hash, last_analyzed_at=datetime.utcnow())
            )

    def _start_task_log(self, kb_id: int | None, task_type: str) -> int:
        if not kb_id:
            return 0
        with get_session() as session:
            log = SummarizeTaskLog(
                kb_id=kb_id,
                task_type=task_type,
                status="running",
                started_at=datetime.utcnow(),
            )
            session.add(log)
            session.flush()
            return log.id

    def _finish_task_log(self, log_id: int, status: str, stats: dict[str, int], error_msg: str | None = None) -> None:
        if not log_id:
            return
        with get_session() as session:
            session.execute(
                update(SummarizeTaskLog)
                .where(SummarizeTaskLog.id == log_id)
                .values(
                    status=status,
                    message_count=stats.get("files_parsed", 0),
                    new_knowledge_count=stats.get("knowledge_generated", 0),
                    error_message=error_msg,
                    finished_at=datetime.utcnow(),
                )
            )


repo_analyzer = RepoAnalyzer()
