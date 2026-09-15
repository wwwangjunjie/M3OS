"""M3OS public-database retrieval helpers for MCP tools."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup

from configs.settings import DEFAULT_ENV_FILE, get_setting, load_settings, resolve_intermediate_path


_SETTINGS = load_settings(
    os.getenv("M3OS_ENV_FILE") or DEFAULT_ENV_FILE,
    use_default_config=True,
)
EMBEDDING_MODEL = str(get_setting(_SETTINGS, "EMBEDDING_MODEL", "llm.embedding_model", "text-embedding-3-large"))
MAX_PAPERS = int(get_setting(_SETTINGS, "MAX_PAPERS", "retrieval.max_papers", 10))
PAPER_DIR = resolve_intermediate_path(_SETTINGS, "PAPER_DIR", "retrieval.paper_dir", "papers")
UNIPROT_NUM_IDS = int(get_setting(_SETTINGS, "UNIPROT_NUM_IDS", "retrieval.uniprot_num_ids", 1))


def _configured_url(env_name: str, config_path: str, *, strip_trailing_slash: bool = False) -> str:
    value = str(get_setting(_SETTINGS, env_name, config_path, "") or "").strip()
    if not value:
        raise RuntimeError(f"Missing retrieval URL configuration: {env_name} / {config_path}")
    if strip_trailing_slash:
        return value.rstrip("/")
    return value


UNIPROT_BASE_URL = _configured_url(
    "UNIPROT_BASE_URL",
    "retrieval.urls.uniprot_base_url",
    strip_trailing_slash=True,
)
CHEMBL_BASE_URL = _configured_url(
    "CHEMBL_BASE_URL",
    "retrieval.urls.chembl_base_url",
    strip_trailing_slash=True,
)
SEMANTIC_SCHOLAR_SEARCH_URL = _configured_url(
    "SEMANTIC_SCHOLAR_SEARCH_URL",
    "retrieval.urls.semantic_scholar_search_url",
)
PUBMED_SEARCH_URL = _configured_url(
    "PUBMED_SEARCH_URL",
    "retrieval.urls.pubmed_search_url",
)
REQUEST_TIMEOUT_SECONDS = 20


def download_pdf(url: str, title: str, paper_dir: str) -> str | None:
    """Download a PDF from a URL and save it under a filename based on title."""
    try:
        response = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT_SECONDS)
        if response.status_code != 200:
            print(f"Failed to download: {title} ({url})")
            return None

        filename = "".join(char if char.isalnum() else "_" for char in title)[:100] + ".pdf"
        output_dir = Path(paper_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / filename

        with file_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    handle.write(chunk)
        return str(file_path)
    except Exception as exc:
        print(f"Error downloading {title}: {exc}")
        return None


def _paper_markdown_root() -> Path:
    configured = os.getenv("PAPER_MARKDOWN_DIR")
    if configured and configured.strip():
        return Path(configured).expanduser()
    return Path(PAPER_DIR).expanduser().resolve().parent / "paper_markdown"


def _ensure_mineru_cli_from_current_python() -> None:
    if os.getenv("MINERU_OPEN_API_CLI") or os.getenv("MINERU_CLI_PATH"):
        return
    candidates = [
        Path(sys.prefix) / "bin" / "mineru-open-api",
        Path(sys.executable).parent / "mineru-open-api",
        Path(__file__).resolve().parents[3] / ".venv" / "bin" / "mineru-open-api",
    ]
    for candidate in candidates:
        if candidate.exists():
            os.environ["MINERU_OPEN_API_CLI"] = str(candidate)
            return


def _paper_document_rag_service():
    _ensure_mineru_cli_from_current_python()
    from m3os.agents_v5.core.config import PathsConfig
    from m3os.agents_v5.services.document_rag import DocumentRAGService

    rag_data_dir = os.getenv("PAPER_RAG_DATA_DIR") or str(
        Path(PAPER_DIR).expanduser().resolve().parent / "paper_rag"
    )
    paths_config = PathsConfig(
        rag_data_dir=rag_data_dir,
        rag_chunk_size=int(os.getenv("PAPER_RAG_CHUNK_SIZE", os.getenv("RAG_CHUNK_SIZE", "1024"))),
        rag_chunk_overlap=int(os.getenv("PAPER_RAG_CHUNK_OVERLAP", os.getenv("RAG_CHUNK_OVERLAP", "200"))),
        rag_top_k=int(os.getenv("PAPER_RAG_TOP_K", os.getenv("RAG_TOP_K", "8"))),
    )
    return DocumentRAGService(paths_config)


def _paper_markdown_cache_dir(pdf_path: Path, markdown_root: Path) -> Path:
    stat = pdf_path.stat()
    digest_payload = {
        "path": str(pdf_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    digest = hashlib.sha256(
        json.dumps(digest_payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    safe_stem = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in pdf_path.stem)[:80]
    return markdown_root / f"{safe_stem}_{digest}"


def _mineru_markdown_paths_for_pdf(service, pdf_path: Path, markdown_root: Path) -> List[str]:
    cache_dir = _paper_markdown_cache_dir(pdf_path, markdown_root)
    metadata_path = cache_dir / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            markdown_paths = [str(Path(path)) for path in metadata.get("markdown_paths", [])]
            if markdown_paths and all(Path(path).exists() for path in markdown_paths):
                return markdown_paths
        except Exception:
            pass

    cache_dir.mkdir(parents=True, exist_ok=True)
    markdown_paths = service.convert_to_markdown(str(pdf_path), cache_dir)
    metadata_path.write_text(
        json.dumps(
            {
                "source_pdf": str(pdf_path),
                "markdown_paths": markdown_paths,
                "conversion_backend": "mineru-open-api",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return markdown_paths


def _search_fallback_message(query: str) -> str:
    return (
        f"Search results for '{query}':\n\n"
        "Due to network restrictions, direct search is unavailable. However, based on the query "
        f"about {query}, you should look for:\n"
        "1. Known drug molecules targeting the specified protein\n"
        "2. Clinical trials and research papers\n"
        "3. Binding affinity data and mechanism of action\n"
        "4. SMILES structures of existing compounds\n\n"
        "Recommend using the download_relevant_papers tool for academic literature and "
        "get_drug_smiles tool for specific compound structures."
    )


def search(query):
    """M3OS web search with DuckDuckGo and PubMed fallback."""
    search_query = str(query or "").strip()
    if not search_query:
        return "Search query is required."

    try:
        from ddgs import DDGS

        results = DDGS().text(search_query, max_results=5)
        if results:
            return results
    except Exception:
        pass

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }
        response = requests.get(
            PUBMED_SEARCH_URL,
            params={"term": search_query, "size": 10},
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.find_all("article", class_="full-docsum")

        results = []
        for article in articles[:5]:
            title_elem = article.find("a", class_="docsum-title")
            abstract_elem = article.find("div", class_="full-view-snippet")

            if title_elem:
                title = title_elem.get_text().strip()
                abstract = abstract_elem.get_text().strip() if abstract_elem else "No abstract available"
                results.append(f"Title: {title}\nAbstract: {abstract}\n")

        if results:
            return "\n".join(results)
    except Exception:
        pass

    return _search_fallback_message(search_query)


def _get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def _protein_name(record: Dict[str, Any]) -> str:
    description = record.get("proteinDescription") or {}
    recommended = description.get("recommendedName") or {}
    full_name = recommended.get("fullName") or {}
    if full_name.get("value"):
        return full_name["value"]

    for key in ("submissionNames", "alternativeNames"):
        names = description.get(key) or []
        if names:
            name = (names[0].get("fullName") or {}).get("value")
            if name:
                return name

    return ""


DEFAULT_UNIPROT_ORGANISM_ID = "9606"


def _uniprot_query(
    search_term: str,
    *,
    reviewed: Optional[bool],
    organism_id: Optional[str],
) -> str:
    filters = [search_term]
    if organism_id:
        filters.append(f"organism_id:{organism_id}")
    if reviewed is True:
        filters.append("reviewed:true")
    return " AND ".join(filters)


def _uniprot_query_attempts(
    *,
    organism_id: Optional[str],
    reviewed: Optional[bool],
) -> List[Dict[str, Any]]:
    organism_text = str(organism_id or "").strip()
    organism_options: List[Optional[str]]
    if organism_text:
        organism_options = [organism_text]
    else:
        organism_options = [DEFAULT_UNIPROT_ORGANISM_ID, None]

    if reviewed is True:
        reviewed_options: List[Optional[bool]] = [True]
    elif reviewed is False:
        reviewed_options = [None]
    else:
        reviewed_options = [True, None]

    attempts: List[Dict[str, Any]] = []
    seen: set[tuple[Optional[str], Optional[bool]]] = set()
    for current_organism_id in organism_options:
        for current_reviewed in reviewed_options:
            key = (current_organism_id, current_reviewed)
            if key in seen:
                continue
            seen.add(key)
            attempts.append(
                {
                    "organism_id": current_organism_id,
                    "reviewed": current_reviewed,
                }
            )
    return attempts


def _uniprot_results(payload: Dict[str, Any]):
    return [
        (record["primaryAccession"], _protein_name(record))
        for record in payload.get("results") or []
        if "primaryAccession" in record
    ]


def get_uniprot_ids(
    protein_name: str,
    size: Optional[int] = None,
    reviewed: Optional[bool] = None,
    organism_id: Optional[str] = None,
):
    """Search UniProtKB and return M3OS-style ID/name tuples.

    By default, human reviewed entries are preferred, but the search falls back
    to broader UniProtKB queries so non-human targets are not silently blocked.
    Pass organism_id to restrict the search to a specific NCBI taxonomy id.
    Pass reviewed=True to require Swiss-Prot/reviewed entries only.
    """
    query = str(protein_name or "").strip()
    if not query:
        return "No UniProt IDs found for ''."

    search_terms = [f"gene_exact:{query}", query]
    result_size = max(1, min(int(size if size is not None else UNIPROT_NUM_IDS), 25))

    try:
        for attempt in _uniprot_query_attempts(
            organism_id=organism_id,
            reviewed=reviewed,
        ):
            for search_term in search_terms:
                payload = _get_json(
                    f"{UNIPROT_BASE_URL}/search",
                    params={
                        "query": _uniprot_query(
                            search_term,
                            reviewed=attempt["reviewed"],
                            organism_id=attempt["organism_id"],
                        ),
                        "format": "json",
                        "size": result_size,
                    },
                )
                results = _uniprot_results(payload)
                if results:
                    return results
    except requests.RequestException as exc:
        return f"API request failed: {exc}"
    except ValueError as exc:
        return f"API request failed: UniProt returned invalid JSON: {exc}"

    return f"No UniProt IDs found for '{protein_name}'."


def fetch_uniprot_fasta(uniprot_id: str):
    """Fetch a FASTA sequence from UniProtKB by accession or entry id."""
    accession = str(uniprot_id or "").strip()
    if not accession:
        return None

    response = requests.get(
        f"{UNIPROT_BASE_URL}/{accession}.fasta",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if response.status_code == 200:
        return response.text

    print(f"Failed to fetch FASTA for {accession}. Status Code: {response.status_code}")
    return None


def get_drug_smiles(drug_name: str):
    """Fetch a drug molecule canonical SMILES from ChEMBL by exact preferred name."""
    query = str(drug_name or "").strip()
    if not query:
        return None, None

    params = {"pref_name__iexact": query, "limit": 25}
    try:
        payload = _get_json(f"{CHEMBL_BASE_URL}/molecule.json", params=params)
    except (requests.RequestException, ValueError) as exc:
        return f"Error: Unable to connect to ChEMBL database. Please check network connection. Details: {exc}"

    for record in payload.get("molecules") or []:
        structures = record.get("molecule_structures") or {}
        canonical_smiles = structures.get("canonical_smiles")
        if canonical_smiles:
            return canonical_smiles

    print(f"No SMILES found for {drug_name}.")
    return None, None


def _openai_runtime_config() -> tuple[str, Optional[str], str, str]:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL") or "gpt-4"
    embedding_model = os.getenv("EMBEDDING_MODEL") or EMBEDDING_MODEL

    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY for answer_research_paper_question.")

    return api_key, base_url, model, embedding_model


def _retrieval_qa_bypass_token_limit(
    vector_store,
    query: str,
    llm,
    k: int = 10,
    fetch_k: int = 50,
    min_k: int = 2,
    chain_type: str = "stuff",
):
    """Run RetrievalQA, reducing k if the selected context exceeds the model limit."""
    from langchain_classic.chains import RetrievalQA
    from langchain_classic.memory import ConversationBufferMemory
    from openai import OpenAIError

    while k >= min_k:
        try:
            retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "fetch_k": fetch_k},
            )
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type=chain_type,
                retriever=retriever,
                memory=ConversationBufferMemory(),
            )
            result = qa_chain.invoke({"query": query})
            if isinstance(result, dict) and "result" in result:
                return result["result"]
            return result
        except OpenAIError as exc:
            if "maximum context length" in str(exc):
                print(f"\nk={k} results hitting the token limit. Reducing k and retrying...\n")
                k -= 1
                continue
            raise

    return "Unable to answer because the retrieved context exceeds the model token limit."


def question_answering(query: str):
    """
    Answer questions using downloaded research papers through a temporary RAG index.

    The implementation mirrors the M3OS tool: it reads PDFs from PAPER_DIR,
    chunks them, builds an in-memory FAISS vector store, then runs RetrievalQA.
    """
    search_query = str(query or "").strip()
    if not search_query:
        return "Question query is required."

    if not os.path.exists(PAPER_DIR):
        return "No papers found in the directory. Use generic design guidelines instead."

    from langchain_community.vectorstores import FAISS
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    service = _paper_document_rag_service()
    markdown_root = _paper_markdown_root()
    markdown_paths: List[str] = []

    for paper_file in os.listdir(PAPER_DIR):
        if not paper_file.endswith(".pdf"):
            continue

        paper_file_path = Path(PAPER_DIR) / paper_file
        with open(paper_file_path, "rb") as file_handle:
            header = file_handle.read(10)
            if not header.startswith(b"%PDF"):
                print(f"Skipping invalid PDF file: {paper_file}")
                continue

        try:
            markdown_paths.extend(_mineru_markdown_paths_for_pdf(service, paper_file_path, markdown_root))
        except Exception as exc:
            print(f"Skipping PDF file that MinerU could not convert {paper_file}: {exc}")

    documents = service._split_markdown_documents(markdown_paths)

    if not documents:
        return "No valid MinerU markdown documents found to construct vector storage."

    api_key, base_url, model, embedding_model = _openai_runtime_config()
    embeddings = OpenAIEmbeddings(
        api_key=api_key,
        base_url=base_url,
        model=embedding_model,
    )
    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=0.3,
    )
    faiss_vectorstore = FAISS.from_documents(documents, embeddings)
    return _retrieval_qa_bypass_token_limit(faiss_vectorstore, search_query, llm)


def _normalize_max_papers(max_papers: Optional[int]) -> int:
    """Return a small, bounded paper download count for interactive agent use."""
    if max_papers is None:
        return MAX_PAPERS
    try:
        value = int(max_papers)
    except (TypeError, ValueError):
        value = 3
    return max(1, min(value, 10))


def download_relevant_papers(query: str, max_papers: Optional[int] = None) -> List[str]:
    """
    Search Semantic Scholar and download open-access PDFs related to a research query.

    Returns a list of local PDF paths downloaded into PAPER_DIR.
    """
    search_query = str(query or "").strip()
    if not search_query:
        return []
    result_limit = _normalize_max_papers(max_papers)

    print(f"Searching for papers on: {search_query} (max_papers={result_limit})")

    headers = {}
    semantic_scholar_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if semantic_scholar_api_key:
        headers["x-api-key"] = semantic_scholar_api_key

    params = {
        "query": search_query,
        "fields": "title,url,abstract,isOpenAccess,openAccessPdf",
        "limit": result_limit,
    }

    try:
        response = requests.get(
            SEMANTIC_SCHOLAR_SEARCH_URL,
            headers=headers or None,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        results = response.json().get("data", [])
    except requests.RequestException as exc:
        print(f"Error fetching papers from Semantic Scholar: {exc}")
        return []
    except ValueError as exc:
        print(f"Error parsing Semantic Scholar response: {exc}")
        return []

    papers_downloaded = []
    for paper in results:
        title = paper.get("title") or "Untitled"
        open_access_pdf = paper.get("openAccessPdf")
        if not isinstance(open_access_pdf, dict):
            continue

        pdf_url = open_access_pdf.get("url")
        if not pdf_url:
            continue

        file_path = download_pdf(pdf_url, title, PAPER_DIR)
        if file_path:
            papers_downloaded.append(file_path)
        time.sleep(2)

    return papers_downloaded
