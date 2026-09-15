"""Document RAG utilities for agents_v5.

This service owns deterministic document ingestion and retrieval. The RAG
agent decides when to use these operations; the service keeps conversion,
chunking, embedding, and FAISS persistence reproducible and testable.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from configs.settings import DEFAULT_ENV_FILE, get_setting, load_settings
from m3os.agents_v5.core.config import PathsConfig


SUPPORTED_DOCUMENT_SUFFIXES = {".pdf", ".md", ".markdown", ".txt", ".docx", ".doc"}


@dataclass
class DocumentIngestionResult:
    """Result of preparing user documents for retrieval."""

    kb_id: Optional[str]
    markdown_paths: List[str]
    metadata: Dict[str, Any]
    failures: List[Dict[str, str]]


class DocumentRAGService:
    """Build and query per-document-set FAISS vector stores."""

    def __init__(self, paths_config: PathsConfig):
        self.paths_config = paths_config
        self.data_dir = Path(paths_config.rag_data_dir)
        if not self.data_dir.is_absolute():
            self.data_dir = Path.cwd() / self.data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def collect_document_files(
        self,
        document_files: Optional[Sequence[str]] = None,
        document_dir: Optional[str] = None,
    ) -> List[str]:
        """Return normalized supported document paths."""
        paths: List[Path] = []
        for item in document_files or []:
            path = Path(item).expanduser()
            if path.is_file() and path.suffix.lower() in SUPPORTED_DOCUMENT_SUFFIXES:
                paths.append(path)

        if document_dir:
            root = Path(document_dir).expanduser()
            if root.exists() and root.is_dir():
                for child in sorted(root.iterdir()):
                    if child.is_file() and child.suffix.lower() in SUPPORTED_DOCUMENT_SUFFIXES:
                        paths.append(child)

        seen = set()
        normalized = []
        for path in paths:
            resolved = str(path.resolve())
            if resolved not in seen:
                normalized.append(resolved)
                seen.add(resolved)
        return normalized

    def prepare_documents(
        self,
        document_files: Optional[Sequence[str]] = None,
        document_dir: Optional[str] = None,
        force_rebuild: bool = False,
    ) -> DocumentIngestionResult:
        """Convert documents to Markdown and build/load their vector store."""
        files = self.collect_document_files(document_files, document_dir)
        if not files:
            return DocumentIngestionResult(None, [], {}, [])

        kb_id = self.compute_kb_id(files)
        kb_dir = self.data_dir / kb_id
        metadata_path = kb_dir / "metadata.json"
        if metadata_path.exists() and not force_rebuild:
            metadata = self.load_metadata(kb_id)
            return DocumentIngestionResult(
                kb_id=kb_id,
                markdown_paths=metadata.get("markdown_files", []),
                metadata=metadata,
                failures=metadata.get("failures", []),
            )

        markdown_dir = kb_dir / "markdown"
        kb_dir.mkdir(parents=True, exist_ok=True)
        markdown_dir.mkdir(parents=True, exist_ok=True)

        markdown_paths: List[str] = []
        failures: List[Dict[str, str]] = []
        for source in files:
            try:
                markdown_paths.extend(self.convert_to_markdown(source, markdown_dir))
            except Exception as exc:
                failures.append({"file": source, "error": str(exc)})

        metadata = self.build_vector_store(
            kb_id=kb_id,
            markdown_paths=markdown_paths,
            source_files=files,
            failures=failures,
            force_rebuild=True,
        )
        return DocumentIngestionResult(kb_id, markdown_paths, metadata, failures)

    def prepare_documents_fail_fast(
        self,
        document_files: Optional[Sequence[str]] = None,
        document_dir: Optional[str] = None,
        force_rebuild: bool = False,
    ) -> DocumentIngestionResult:
        """Prepare user documents before chat startup and fail on unusable input."""
        self._validate_document_inputs(document_files=document_files, document_dir=document_dir)
        files = self.collect_document_files(document_files, document_dir)
        if not files:
            raise ValueError(
                "Document inputs were provided, but no supported files were found. "
                "Supported suffixes: .pdf, .md, .markdown, .txt, .docx, .doc."
            )

        result = self.prepare_documents(
            document_files=document_files,
            document_dir=document_dir,
            force_rebuild=force_rebuild,
        )
        if (
            self._has_pdf_inputs(files)
            and result.metadata.get("conversion_backend") != "mineru-open-api"
            and not force_rebuild
        ):
            result = self.prepare_documents(
                document_files=document_files,
                document_dir=document_dir,
                force_rebuild=True,
            )
        if result.failures and not force_rebuild:
            result = self.prepare_documents(
                document_files=document_files,
                document_dir=document_dir,
                force_rebuild=True,
            )
        if result.failures:
            details = "; ".join(
                f"{item.get('file', '<unknown>')}: {item.get('error', '<unknown error>')}"
                for item in result.failures
            )
            raise RuntimeError(f"Document preprocessing failed: {details}")
        doc_count = int(result.metadata.get("doc_count", 0) or 0)
        if not result.kb_id or doc_count <= 0:
            raise RuntimeError(
                "Document preprocessing produced no retrievable chunks. "
                "Check that the input documents contain extractable text."
            )
        return result

    def prepare_documents_full_text_fail_fast(
        self,
        document_files: Optional[Sequence[str]] = None,
        document_dir: Optional[str] = None,
        force_rebuild: bool = False,
    ) -> DocumentIngestionResult:
        """Convert user documents to Markdown without chunking or vector storage."""
        self._validate_document_inputs(document_files=document_files, document_dir=document_dir)
        files = self.collect_document_files(document_files, document_dir)
        if not files:
            raise ValueError(
                "Document inputs were provided, but no supported files were found. "
                "Supported suffixes: .pdf, .md, .markdown, .txt, .docx, .doc."
            )

        result = self.prepare_documents_full_text(
            document_files=document_files,
            document_dir=document_dir,
            force_rebuild=force_rebuild,
        )
        if result.failures and not force_rebuild:
            result = self.prepare_documents_full_text(
                document_files=document_files,
                document_dir=document_dir,
                force_rebuild=True,
            )
        if result.failures:
            details = "; ".join(
                f"{item.get('file', '<unknown>')}: {item.get('error', '<unknown error>')}"
                for item in result.failures
            )
            raise RuntimeError(f"Document preprocessing failed: {details}")
        if not result.kb_id or not result.markdown_paths:
            raise RuntimeError(
                "Document preprocessing produced no converted Markdown. "
                "Check that the input documents contain extractable text."
            )
        return result

    def prepare_documents_full_text(
        self,
        document_files: Optional[Sequence[str]] = None,
        document_dir: Optional[str] = None,
        force_rebuild: bool = False,
    ) -> DocumentIngestionResult:
        """Convert documents to Markdown and save metadata only."""
        files = self.collect_document_files(document_files, document_dir)
        if not files:
            return DocumentIngestionResult(None, [], {}, [])

        kb_id = self.compute_kb_id(files)
        kb_dir = self.data_dir / kb_id
        metadata_path = kb_dir / "metadata.json"
        if metadata_path.exists() and not force_rebuild:
            metadata = self.load_metadata(kb_id)
            markdown_paths = metadata.get("markdown_files", [])
            if markdown_paths and metadata.get("ingestion_mode") == "full_text_system_prompt":
                return DocumentIngestionResult(
                    kb_id=kb_id,
                    markdown_paths=markdown_paths,
                    metadata=metadata,
                    failures=metadata.get("failures", []),
                )

        markdown_dir = kb_dir / "markdown"
        kb_dir.mkdir(parents=True, exist_ok=True)
        markdown_dir.mkdir(parents=True, exist_ok=True)

        markdown_paths: List[str] = []
        failures: List[Dict[str, str]] = []
        for source in files:
            try:
                markdown_paths.extend(self.convert_to_markdown(source, markdown_dir))
            except Exception as exc:
                failures.append({"file": source, "error": str(exc)})

        metadata = self.build_full_text_metadata(
            kb_id=kb_id,
            markdown_paths=markdown_paths,
            source_files=files,
            failures=failures,
            force_rebuild=True,
        )
        return DocumentIngestionResult(kb_id, markdown_paths, metadata, failures)

    def _validate_document_inputs(
        self,
        document_files: Optional[Sequence[str]] = None,
        document_dir: Optional[str] = None,
    ) -> None:
        for item in document_files or []:
            path = Path(item).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Document file does not exist: {item}")
            if not path.is_file():
                raise ValueError(f"Document path is not a file: {item}")
        if document_dir:
            root = Path(document_dir).expanduser()
            if not root.exists():
                raise FileNotFoundError(f"Document directory does not exist: {document_dir}")
            if not root.is_dir():
                raise ValueError(f"Document path is not a directory: {document_dir}")

    def _has_pdf_inputs(self, files: Sequence[str]) -> bool:
        return any(Path(file_path).suffix.lower() == ".pdf" for file_path in files)

    def compute_kb_id(self, files: Sequence[str]) -> str:
        """Create a stable cache key for a document set and RAG parameters."""
        parts: List[Dict[str, Any]] = []
        for file_path in sorted(files):
            path = Path(file_path)
            stat = path.stat()
            parts.append(
                {
                    "path": str(path.resolve()),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        payload = {
            "files": parts,
            "chunk_size": self.paths_config.rag_chunk_size,
            "chunk_overlap": self.paths_config.rag_chunk_overlap,
            "embedding_model": self._embedding_model_name(),
        }
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return digest[:24]

    def convert_to_markdown(self, source_file: str, markdown_dir: Path) -> List[str]:
        """Convert one supported source file into Markdown paths."""
        source = Path(source_file)
        suffix = source.suffix.lower()
        if suffix in {".md", ".markdown"}:
            target = markdown_dir / f"{source.stem}.md"
            if source.resolve() != target.resolve():
                shutil.copyfile(source, target)
            return [str(target)]
        if suffix == ".txt":
            return [str(self._convert_text_to_markdown(source, markdown_dir))]
        if suffix == ".docx":
            return [str(self._convert_docx_to_markdown(source, markdown_dir))]
        if suffix == ".doc":
            docx_path = self._convert_doc_to_docx(source, markdown_dir)
            return [str(self._convert_docx_to_markdown(docx_path, markdown_dir, source_name=source.name))]

        if suffix != ".pdf":
            raise ValueError(f"Unsupported document type: {source_file}")

        if self._mineru_token():
            try:
                return [str(path) for path in self._run_mineru_extract(source, markdown_dir)]
            except Exception as exc:
                raise RuntimeError(
                    f"PDF conversion with MinerU extract failed: {self._format_conversion_error(exc)}"
                ) from exc

        try:
            return [str(path) for path in self._run_mineru_flash_extract(source, markdown_dir)]
        except subprocess.CalledProcessError as exc:
            if self._is_mineru_page_limit_error(exc):
                try:
                    return [
                        str(path)
                        for path in self._run_mineru_flash_extract_by_page_ranges(
                            source,
                            markdown_dir,
                        )
                    ]
                except Exception as range_exc:
                    raise RuntimeError(
                        "PDF conversion with MinerU page ranges failed: "
                        f"{self._format_conversion_error(range_exc)}"
                    ) from range_exc
            raise RuntimeError(
                f"PDF conversion with MinerU failed: {self._format_conversion_error(exc)}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"PDF conversion with MinerU failed: {self._format_conversion_error(exc)}"
            ) from exc

    def _convert_text_to_markdown(self, source: Path, markdown_dir: Path) -> Path:
        markdown_dir.mkdir(parents=True, exist_ok=True)
        target = markdown_dir / f"{source.stem}.md"
        text = source.read_text(encoding="utf-8", errors="replace")
        target.write_text(
            f"# {source.name}\n\n{text.strip()}\n",
            encoding="utf-8",
        )
        return target

    def _convert_docx_to_markdown(
        self,
        source: Path,
        markdown_dir: Path,
        *,
        source_name: str | None = None,
    ) -> Path:
        markdown_dir.mkdir(parents=True, exist_ok=True)
        target_stem = Path(source_name or source.name).stem
        target = markdown_dir / f"{target_stem}.md"
        markdown = self._extract_docx_markdown(source, source_name=source_name or source.name)
        target.write_text(markdown, encoding="utf-8")
        return target

    def _extract_docx_markdown(self, source: Path, *, source_name: str) -> str:
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError(
                "python-docx is required to read .docx files. "
                "Install it in the agents_v5 Python environment."
            ) from exc

        document = Document(str(source))
        sections: List[str] = [f"# {source_name}"]
        paragraph_text = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
        if paragraph_text:
            sections.append("\n\n".join(paragraph_text))

        for table_index, table in enumerate(document.tables, start=1):
            rows = [
                [self._escape_markdown_cell(cell.text.strip()) for cell in row.cells]
                for row in table.rows
            ]
            rows = [row for row in rows if any(cell for cell in row)]
            if not rows:
                continue
            width = max(len(row) for row in rows)
            normalized_rows = [row + [""] * (width - len(row)) for row in rows]
            header = normalized_rows[0]
            body = normalized_rows[1:]
            table_lines = [
                f"## Table {table_index}",
                "| " + " | ".join(header) + " |",
                "| " + " | ".join(["---"] * width) + " |",
            ]
            table_lines.extend("| " + " | ".join(row) + " |" for row in body)
            sections.append("\n".join(table_lines))

        content = "\n\n".join(section for section in sections if section.strip()).strip()
        return f"{content}\n"

    def _convert_doc_to_docx(self, source: Path, markdown_dir: Path) -> Path:
        converter = self._office_converter_command()
        if not converter:
            raise FileNotFoundError(
                "LibreOffice/soffice was not found; it is required to convert legacy .doc files."
            )
        output_dir = markdown_dir / f"{source.stem}_doc_conversion"
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            converter,
            "--headless",
            "--convert-to",
            "docx",
            "--outdir",
            str(output_dir),
            str(source),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Legacy .doc conversion failed: {self._format_conversion_error(exc)}"
            ) from exc

        expected = output_dir / f"{source.stem}.docx"
        if expected.exists():
            return expected
        candidates = sorted(output_dir.glob("*.docx"))
        if candidates:
            return candidates[0]
        raise RuntimeError(f"Legacy .doc conversion did not produce a .docx file for {source}")

    @staticmethod
    def _office_converter_command() -> str | None:
        for candidate in (
            os.getenv("LIBREOFFICE_CLI"),
            shutil.which("libreoffice"),
            shutil.which("soffice"),
        ):
            if candidate and Path(candidate).exists():
                return str(candidate)
        return None

    @staticmethod
    def _escape_markdown_cell(value: str) -> str:
        return " ".join(value.replace("|", "\\|").split())

    def _run_mineru_extract(self, source: Path, output_dir: Path) -> List[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        expected = output_dir / f"{source.stem}.md"
        expected.unlink(missing_ok=True)
        before = set(output_dir.rglob("*.md"))
        subprocess.run(
            [
                self._mineru_command(),
                "extract",
                str(source),
                "-o",
                str(output_dir),
                "-f",
                "md",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=self._mineru_env(),
        )
        after = set(output_dir.rglob("*.md"))
        created = sorted(after - before)
        if created:
            return created
        if expected.exists():
            return [expected]
        raise RuntimeError(f"MinerU extract did not produce a Markdown file for {source}")

    def _run_mineru_flash_extract(
        self,
        source: Path,
        output_dir: Path,
        pages: Optional[str] = None,
    ) -> List[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        expected = output_dir / f"{source.stem}.md"
        expected.unlink(missing_ok=True)
        before = set(output_dir.rglob("*.md"))
        command = [
            self._mineru_command(),
            "flash-extract",
            str(source),
            "-o",
            str(output_dir),
        ]
        if pages:
            command.extend(["--pages", pages])
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=self._mineru_env(),
        )
        after = set(output_dir.rglob("*.md"))
        created = sorted(after - before)
        if created:
            return created
        if expected.exists():
            return [expected]
        page_note = f" for pages {pages}" if pages else ""
        raise RuntimeError(f"MinerU did not produce a Markdown file for {source}{page_note}")

    def _run_mineru_flash_extract_by_page_ranges(
        self,
        source: Path,
        markdown_dir: Path,
    ) -> List[Path]:
        page_count = self._pdf_page_count(source)
        if page_count <= 0:
            raise RuntimeError(f"Could not determine page count for {source}")

        markdown_paths: List[Path] = []
        for start in range(1, page_count + 1, 20):
            end = min(start + 19, page_count)
            pages = f"{start}-{end}"
            range_dir = markdown_dir / f"{source.stem}_pages_{start}_{end}"
            created = self._run_mineru_flash_extract(source, range_dir, pages=pages)
            markdown_paths.extend(
                self._normalize_page_range_markdown(created, markdown_dir, source.stem, start, end)
            )
        return markdown_paths

    def _normalize_page_range_markdown(
        self,
        markdown_paths: Sequence[Path],
        markdown_dir: Path,
        source_stem: str,
        start: int,
        end: int,
    ) -> List[Path]:
        normalized: List[Path] = []
        for index, path in enumerate(markdown_paths, start=1):
            suffix = f"pages_{start}_{end}"
            if len(markdown_paths) > 1:
                suffix = f"{suffix}_{index}"
            target = markdown_dir / f"{source_stem}_{suffix}.md"
            if path.resolve() != target.resolve():
                shutil.copyfile(path, target)
            normalized.append(target)
        return normalized

    def _pdf_page_count(self, source: Path) -> int:
        command = shutil.which("pdfinfo")
        if not command:
            raise FileNotFoundError(
                "pdfinfo command was not found in PATH; it is required to split "
                "MinerU flash-extract requests for PDFs over 20 pages."
            )
        result = subprocess.run(
            [command, str(source)],
            check=True,
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            if line.lower().startswith("pages:"):
                return int(line.split(":", 1)[1].strip())
        raise RuntimeError(f"pdfinfo output did not include page count for {source}")

    def _is_mineru_page_limit_error(self, exc: subprocess.CalledProcessError) -> bool:
        message = "\n".join(part for part in [exc.stderr, exc.stdout] if part)
        return "[-30003]" in message or "page count exceeds API limit" in message

    def _format_conversion_error(self, exc: Optional[Exception]) -> str:
        if exc is None:
            return "none"
        if isinstance(exc, subprocess.CalledProcessError):
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            parts = [f"exit status {exc.returncode}"]
            if stderr:
                parts.append(f"stderr: {stderr}")
            if stdout:
                parts.append(f"stdout: {stdout}")
            return "; ".join(parts)
        return str(exc)

    def _mineru_command(self) -> str:
        """Return the MinerU CLI executable path or raise an actionable error."""
        configured = os.getenv("MINERU_OPEN_API_CLI") or os.getenv("MINERU_CLI_PATH")
        if configured:
            return configured

        command = shutil.which("mineru-open-api")
        if command:
            return command

        raise FileNotFoundError(
            "mineru-open-api command was not found in PATH. "
            "MINERU_API_KEY is loaded, but it is only the API token, not the CLI executable. "
            "Install the CLI with `uv tool install mineru-open-api`, or set "
            "MINERU_OPEN_API_CLI=/absolute/path/to/mineru-open-api in your env file."
        )

    def _mineru_env(self) -> Dict[str, str]:
        """Build an environment that passes the configured MinerU token to the CLI."""
        env = os.environ.copy()
        api_key = self._mineru_token(env)
        if api_key:
            # Different MinerU CLI versions have used different token names.
            # Keep MINERU_API_KEY and mirror it to common aliases.
            env.setdefault("MINERU_TOKEN", api_key)
            env.setdefault("MINERU_OPEN_API_KEY", api_key)
            env.setdefault("MINERU_OPEN_API_TOKEN", api_key)
        return env

    def _mineru_token(self, env: Optional[Dict[str, str]] = None) -> Optional[str]:
        env = env or os.environ
        return (
            env.get("MINERU_TOKEN")
            or env.get("MINERU_API_KEY")
            or env.get("MINERU_OPEN_API_KEY")
            or env.get("MINERU_OPEN_API_TOKEN")
        )

    def build_vector_store(
        self,
        kb_id: str,
        markdown_paths: Sequence[str],
        source_files: Optional[Sequence[str]] = None,
        failures: Optional[List[Dict[str, str]]] = None,
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        """Split Markdown documents, embed chunks, save FAISS and metadata."""
        kb_dir = self.data_dir / kb_id
        metadata_path = kb_dir / "metadata.json"
        if metadata_path.exists() and not force_rebuild:
            return self.load_metadata(kb_id)

        documents = self._split_markdown_documents(markdown_paths)
        topics = self._extract_topics(documents)
        if documents:
            embeddings = self._create_embeddings()
            vector_store = self._faiss_class().from_documents(
                documents,
                embedding=embeddings,
            )
            vector_store.save_local(str(kb_dir))

        metadata = {
            "kb_id": kb_id,
            "source_files": list(source_files or []),
            "markdown_files": list(markdown_paths),
            "topics": topics,
            "doc_count": len(documents),
            "chunk_size": self.paths_config.rag_chunk_size,
            "chunk_overlap": self.paths_config.rag_chunk_overlap,
            "embedding_model": self._embedding_model_name(),
            "conversion_backend": "mineru-open-api",
            "failures": failures or [],
        }
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
        return metadata

    def build_full_text_metadata(
        self,
        kb_id: str,
        markdown_paths: Sequence[str],
        source_files: Optional[Sequence[str]] = None,
        failures: Optional[List[Dict[str, str]]] = None,
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        """Save converted-document metadata without FAISS or embeddings."""
        kb_dir = self.data_dir / kb_id
        metadata_path = kb_dir / "metadata.json"
        if metadata_path.exists() and not force_rebuild:
            return self.load_metadata(kb_id)

        topics = self._extract_topics_from_markdown(markdown_paths)
        metadata = {
            "kb_id": kb_id,
            "source_files": list(source_files or []),
            "markdown_files": list(markdown_paths),
            "topics": topics,
            "doc_count": len(markdown_paths),
            "chunk_count": 0,
            "chunk_size": None,
            "chunk_overlap": None,
            "embedding_model": None,
            "conversion_backend": "mineru-open-api",
            "ingestion_mode": "full_text_system_prompt",
            "vector_store": False,
            "failures": failures or [],
        }
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
        return metadata

    def _extract_topics_from_markdown(self, markdown_paths: Sequence[str]) -> List[str]:
        topics: List[str] = []
        seen: set[str] = set()
        for markdown_file in markdown_paths:
            path = Path(markdown_file)
            if not path.exists():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for line in text.splitlines():
                heading = line.strip()
                if not heading.startswith("#"):
                    continue
                topic = heading.lstrip("#").strip()
                if topic and topic not in seen:
                    topics.append(topic)
                    seen.add(topic)
                if len(topics) >= 50:
                    return topics
        return topics

    def retrieve_user_documents(
        self,
        query: str,
        kb_id: Optional[str],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve source chunks from a user-document knowledge base."""
        if not kb_id:
            return []
        kb_dir = self.data_dir / kb_id
        if not kb_dir.exists():
            return []
        if not (kb_dir / "index.faiss").exists():
            return []

        vector_store = self._faiss_class().load_local(
            folder_path=str(kb_dir),
            embeddings=self._create_embeddings(),
            allow_dangerous_deserialization=True,
        )
        docs_and_scores = vector_store.similarity_search_with_score(
            query,
            k=top_k or self.paths_config.rag_top_k,
        )
        results = []
        for doc, score in docs_and_scores:
            results.append(
                {
                    "source_type": "user_document",
                    "content": doc.page_content,
                    "metadata": dict(doc.metadata),
                    "score": float(score),
                }
            )
        return results

    def get_uploaded_file_info(self, kb_id: Optional[str]) -> Dict[str, Any]:
        """Return session document metadata and full-text feasibility."""
        if not kb_id:
            return {
                "kb_id": None,
                "has_documents": False,
                "document_count": 0,
                "chunk_count": 0,
                "source_files": [],
                "markdown_files": [],
                "documents": [],
                "topics": [],
                "total_chars": 0,
                "full_text_max_chars": self.paths_config.rag_full_text_max_chars,
                "can_load_full_text": False,
            }

        metadata = self.load_metadata(kb_id)
        markdown_files = list(metadata.get("markdown_files", []))
        source_files = list(metadata.get("source_files", []))
        documents = []
        total_chars = 0
        for index, markdown_file in enumerate(markdown_files):
            markdown_path = Path(markdown_file)
            char_count = self._safe_text_char_count(markdown_path)
            total_chars += char_count
            source_file = source_files[index] if index < len(source_files) else None
            documents.append(
                {
                    "source_file": source_file,
                    "source_name": Path(source_file).name if source_file else markdown_path.name,
                    "markdown_file": str(markdown_path),
                    "markdown_name": markdown_path.name,
                    "char_count": char_count,
                    "exists": markdown_path.exists(),
                }
            )

        max_chars = self.paths_config.rag_full_text_max_chars
        return {
            "kb_id": kb_id,
            "has_documents": bool(markdown_files),
            "document_count": len(markdown_files),
            "chunk_count": int(metadata.get("chunk_count", metadata.get("doc_count", 0)) or 0),
            "source_files": source_files,
            "markdown_files": markdown_files,
            "documents": documents,
            "topics": metadata.get("topics", []),
            "total_chars": total_chars,
            "full_text_max_chars": max_chars,
            "can_load_full_text": bool(markdown_files) and total_chars <= max_chars,
        }

    def load_uploaded_documents(
        self,
        kb_id: Optional[str],
        document_names: Optional[Sequence[str]] = None,
        max_chars: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Load Markdown full text for documents listed in kb metadata only."""
        info = self.get_uploaded_file_info(kb_id)
        if not kb_id or not info.get("has_documents"):
            return {
                **info,
                "mode": "full",
                "content": "",
                "loaded_documents": [],
                "truncated": False,
                "error": "No uploaded documents are available in this session.",
            }

        selected = self._filter_document_info(info["documents"], document_names)
        limit = max_chars if max_chars is not None else self.paths_config.rag_full_text_max_chars
        sections = []
        loaded_documents = []
        total_chars = 0
        truncated = False
        for item in selected:
            path = Path(item["markdown_file"])
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            header = (
                f"[DOCUMENT: {item['source_name']}]\n"
                f"Markdown file: {item['markdown_name']}\n"
            )
            section = f"{header}\n{text.strip()}\n"
            if limit is not None and total_chars + len(section) > limit:
                remaining = max(limit - total_chars, 0)
                if remaining > len(header):
                    section = section[:remaining]
                    sections.append(section)
                    total_chars += len(section)
                truncated = True
                break
            sections.append(section)
            total_chars += len(section)
            loaded_documents.append(item)

        return {
            **info,
            "mode": "full",
            "content": "\n\n".join(sections).strip(),
            "loaded_documents": loaded_documents,
            "loaded_document_count": len(loaded_documents),
            "loaded_chars": total_chars,
            "truncated": truncated,
        }

    def classify_document_intent(self, query: str) -> str:
        """Lightweight route for document questions."""
        normalized = " ".join((query or "").lower().split())
        if not normalized:
            return "fact_lookup"

        comparative_terms = [
            "比较",
            "对比",
            "异同",
            "共同点",
            "差异",
            "compare",
            "comparison",
            "contrast",
            "across",
        ]
        summary_terms = [
            "总结",
            "概括",
            "综述",
            "总览",
            "归纳",
            "提炼",
            "传入的文档",
            "上传的文档",
            "全部文档",
            "所有文档",
            "这些文档",
            "这几篇",
            "四篇文章",
            "summarize",
            "summary",
            "overview",
            "review",
            "outline",
            "key points",
            "main points",
        ]
        fact_terms = [
            "多少",
            "哪一个",
            "是什么",
            "具体",
            "数值",
            "性质",
            "机制",
            "compound",
            "fxr",
            "jak",
            "prmt5",
            "what is",
            "which",
            "specific",
            "property",
            "mechanism",
        ]

        has_summary = any(term in normalized for term in summary_terms)
        has_comparative = any(term in normalized for term in comparative_terms)
        has_fact = any(term in normalized for term in fact_terms)

        if has_comparative:
            return "comparative_summary"
        if has_summary and has_fact:
            return "mixed"
        if has_summary:
            return "global_summary"
        return "fact_lookup"

    def load_metadata(self, kb_id: str) -> Dict[str, Any]:
        path = self.data_dir / kb_id / "metadata.json"
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _safe_text_char_count(self, path: Path) -> int:
        if not path.exists() or not path.is_file():
            return 0
        try:
            return len(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return 0

    def _filter_document_info(
        self,
        documents: Sequence[Dict[str, Any]],
        document_names: Optional[Sequence[str]],
    ) -> List[Dict[str, Any]]:
        if not document_names:
            return list(documents)

        needles = [name.strip().lower() for name in document_names if name and name.strip()]
        if not needles:
            return list(documents)

        selected = []
        for item in documents:
            haystack = " ".join(
                str(item.get(key, "")).lower()
                for key in ("source_file", "source_name", "markdown_file", "markdown_name")
            )
            if any(needle in haystack for needle in needles):
                selected.append(item)
        return selected

    def _split_markdown_documents(self, markdown_paths: Sequence[str]):
        splitter_cls = self._markdown_header_splitter_class()
        recursive_cls = self._recursive_splitter_class()
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        header_splitter = splitter_cls(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )
        recursive_splitter = recursive_cls(
            chunk_size=self.paths_config.rag_chunk_size,
            chunk_overlap=self.paths_config.rag_chunk_overlap,
        )

        all_docs = []
        for markdown_path in markdown_paths:
            path = Path(markdown_path)
            text = path.read_text(encoding="utf-8")
            header_docs = header_splitter.split_text(text)
            for doc in header_docs:
                doc.metadata = {
                    **dict(doc.metadata),
                    "source_file": str(path),
                    "source_name": path.name,
                    "source_type": "user_document",
                }
            split_docs = recursive_splitter.split_documents(header_docs)
            for index, doc in enumerate(split_docs):
                doc.metadata["chunk_index"] = index
            all_docs.extend(split_docs)
        return all_docs

    def _extract_topics(self, documents: Iterable[Any]) -> List[str]:
        topics = []
        seen = set()
        for doc in documents:
            for key in ("Header 1", "Header 2", "Header 3"):
                value = doc.metadata.get(key)
                if value and value not in seen:
                    topics.append(value)
                    seen.add(value)
                if len(topics) >= 30:
                    return topics
        return topics

    def _create_embeddings(self):
        from langchain_openai import OpenAIEmbeddings

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")

        return OpenAIEmbeddings(
            api_key=api_key,
            base_url=base_url,
            model=self._embedding_model_name(),
        )

    def _embedding_model_name(self) -> str:
        model = os.getenv("EMBEDDING_MODEL")
        if model:
            return model
        settings = load_settings(
            os.getenv("M3OS_ENV_FILE") or DEFAULT_ENV_FILE,
            use_default_config=True,
            dotenv_override=False,
        )
        return str(get_setting(settings, "EMBEDDING_MODEL", "llm.embedding_model", "text-embedding-3-large"))

    @staticmethod
    def _faiss_class():
        from langchain_community.vectorstores import FAISS

        return FAISS

    @staticmethod
    def _markdown_header_splitter_class():
        from langchain_text_splitters import MarkdownHeaderTextSplitter

        return MarkdownHeaderTextSplitter

    @staticmethod
    def _recursive_splitter_class():
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        return RecursiveCharacterTextSplitter
