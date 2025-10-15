"""
LangGraph-based RAG Pipeline with user approvals and detailed progress tracking.
"""
import os
import sys
import time
from pathlib import Path
from typing import TypedDict, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest import chunk_articles
from src.build_index import build_chroma
from src.query import answer_question, load_questions_csv
from src.utils import read_json, save_json
from src.rate_limiter import RateLimiter
from src.config import (
    INPUT_DATA, CHUNKS_DATA, QUERIES, OUTPUT_DATA, CHROMA_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, RERANK_MODEL,
    GENERATION_MODEL, TOP_K_RETRIEVAL, USE_RERANK
)

load_dotenv()
console = Console()

class PipelineState(TypedDict):
    """State that flows through the pipeline"""
    stage: str
    chunks_path: str
    questions_path: str
    results_path: str
    chroma_path: str
    chunks_created: int
    questions_processed: int
    total_questions: int
    start_time: float
    stage_times: dict
    total_tokens: int
    approved: bool
    error: str

def display_env_config(stage: str) -> None:
    """Display relevant environment variables for a stage"""
    table = Table(title=f"Configuration for {stage}", show_header=True)
    table.add_column("Variable", style="cyan")
    table.add_column("Value", style="green")
    
    if stage == "Chunking":
        table.add_row("INPUT_DATA", str(INPUT_DATA))
        table.add_row("CHUNKS_DATA", str(CHUNKS_DATA))
        table.add_row("CHUNK_SIZE", str(CHUNK_SIZE))
        table.add_row("CHUNK_OVERLAP", str(CHUNK_OVERLAP))
    
    elif stage == "Embedding":
        table.add_row("EMBEDDING_MODEL", EMBEDDING_MODEL)
        table.add_row("EMBED_BATCH_SIZE", os.getenv("EMBED_BATCH_SIZE", "64"))
        table.add_row("CHROMA_PERSIST_DIRECTORY", str(CHROMA_DIR))
    
    elif stage == "Query":
        table.add_row("QUERIES", str(QUERIES))
        table.add_row("OUTPUT_DATA", str(OUTPUT_DATA))
        table.add_row("RERANK_MODEL", RERANK_MODEL)
        table.add_row("GENERATION_MODEL", GENERATION_MODEL)
        table.add_row("TOP_K_RETRIEVAL", str(TOP_K_RETRIEVAL))
        table.add_row("USE_RERANK", str(USE_RERANK))
    
    console.print(table)

def get_user_approval(stage: str) -> bool:
    """Get user approval to proceed with a stage"""
    console.print(Panel.fit(
        f"[bold yellow]Ready to execute: {stage}[/bold yellow]\n"
        "Review the configuration above.",
        title="⚠️  Approval Required"
    ))
    
    response = console.input("[bold cyan]Proceed? (yes/no): [/bold cyan]").strip().lower()
    return response in ['yes', 'y']

# ============================================
# Pipeline Nodes
# ============================================

def start_node(state: PipelineState) -> PipelineState:
    """Initialize the pipeline"""
    console.print("\n" + "="*70)
    console.print("[bold magenta]🚀 RAG Pipeline - LangGraph Orchestrator[/bold magenta]")
    console.print("="*70 + "\n")
    
    state["start_time"] = time.time()
    state["stage_times"] = {}
    state["total_tokens"] = 0
    
    # Display overall configuration
    table = Table(title="Pipeline Overview", show_header=True)
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    
    chunks_path = Path(CHUNKS_DATA)
    chroma_path = Path(CHROMA_DIR) / "chroma.sqlite3"
    
    table.add_row("Input Articles", "✓ Found" if INPUT_DATA.exists() else "✗ Missing")
    table.add_row("Chunks", "✓ Exists" if chunks_path.exists() else "⚠ Will Create")
    table.add_row("ChromaDB Index", "✓ Exists" if chroma_path.exists() else "⚠ Will Create")
    
    console.print(table)
    console.print()
    
    state["chunks_path"] = str(chunks_path)
    state["chroma_path"] = str(chroma_path)
    state["questions_path"] = str(QUERIES)
    state["results_path"] = str(OUTPUT_DATA)
    
    return state

def check_chunks_node(state: PipelineState) -> PipelineState:
    """Check if chunks exist"""
    chunks_path = Path(state["chunks_path"])
    
    if chunks_path.exists():
        chunks = read_json(str(chunks_path))
        state["chunks_created"] = len(chunks)
        console.print(f"[green]✓ Found existing chunks: {len(chunks)} chunks[/green]\n")
        state["stage"] = "check_index"
    else:
        console.print("[yellow]⚠ Chunks not found. Need to create them.[/yellow]\n")
        state["stage"] = "approve_chunking"
    
    return state

def approve_chunking_node(state: PipelineState) -> PipelineState:
    """Get user approval for chunking"""
    display_env_config("Chunking")
    state["approved"] = get_user_approval("Document Chunking")
    
    if state["approved"]:
        state["stage"] = "chunking"
    else:
        state["error"] = "User cancelled chunking"
        state["stage"] = "end"
    
    return state

def chunking_node(state: PipelineState) -> PipelineState:
    """Perform document chunking"""
    console.print("\n[bold blue]📄 Stage 1: Document Chunking[/bold blue]")
    stage_start = time.time()
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Chunking documents...", total=100)
            
            progress.update(task, advance=30)
            
            chunks = chunk_articles(
                str(INPUT_DATA),
                state["chunks_path"],
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            
            progress.update(task, advance=70)
            state["chunks_created"] = len(chunks)
        
        elapsed = time.time() - stage_start
        state["stage_times"]["chunking"] = elapsed
        
        console.print(f"[green]✓ Created {state['chunks_created']} chunks in {elapsed:.2f}s[/green]\n")
        state["stage"] = "check_index"
        
    except Exception as e:
        state["error"] = f"Chunking failed: {str(e)}"
        state["stage"] = "end"
    
    return state

def check_index_node(state: PipelineState) -> PipelineState:
    """Check if ChromaDB index exists"""
    chroma_path = Path(state["chroma_path"])
    
    if chroma_path.exists():
        console.print(f"[green]✓ Found existing ChromaDB index[/green]\n")
        state["stage"] = "approve_query"
    else:
        console.print("[yellow]⚠ ChromaDB index not found. Need to build it.[/yellow]\n")
        state["stage"] = "approve_indexing"
    
    return state

def approve_indexing_node(state: PipelineState) -> PipelineState:
    """Get user approval for indexing"""
    display_env_config("Embedding")
    state["approved"] = get_user_approval("Vector Index Building")
    
    if state["approved"]:
        state["stage"] = "indexing"
    else:
        state["error"] = "User cancelled indexing"
        state["stage"] = "end"
    
    return state

def indexing_node(state: PipelineState) -> PipelineState:
    """Build ChromaDB index"""
    console.print("\n[bold blue]🔍 Stage 2: Building Vector Index[/bold blue]")
    stage_start = time.time()
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Building embeddings and index...", total=None)
            
            # Display model info
            console.print(f"[dim]Using: {EMBEDDING_MODEL}[/dim]")
            console.print(f"[dim]Batch size: {os.getenv('EMBED_BATCH_SIZE', '64')}[/dim]\n")
            
            db = build_chroma(state["chunks_path"], persist_dir=str(CHROMA_DIR))
            
            progress.update(task, completed=True)
        
        elapsed = time.time() - stage_start
        state["stage_times"]["indexing"] = elapsed
        
        console.print(f"[green]✓ Index built successfully in {elapsed:.2f}s[/green]\n")
        state["stage"] = "approve_query"
        
    except Exception as e:
        state["error"] = f"Indexing failed: {str(e)}"
        state["stage"] = "end"
    
    return state

def approve_query_node(state: PipelineState) -> PipelineState:
    """Get user approval for querying"""
    display_env_config("Query")
    
    # Load and display question count
    questions = load_questions_csv(state["questions_path"])
    state["total_questions"] = len(questions)
    
    console.print(f"\n[cyan]Found {len(questions)} questions to process[/cyan]")
    
    state["approved"] = get_user_approval("Query Processing")
    
    if state["approved"]:
        state["stage"] = "querying"
    else:
        state["error"] = "User cancelled querying"
        state["stage"] = "end"
    
    return state

def querying_node(state: PipelineState) -> PipelineState:
    """Process all queries"""
    console.print("\n[bold blue]💬 Stage 3: Processing Queries[/bold blue]")
    stage_start = time.time()
    
    try:
        questions = load_questions_csv(state["questions_path"])
        results = []
        
        # Initialize rate limiter
        rate_limit_tpm = int(os.getenv("RATE_LIMIT_TPM", "30000"))
        rate_limit_buffer = float(os.getenv("RATE_LIMIT_BUFFER", "0.75"))
        enable_rate_limiting = os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true"
        
        rate_limiter = RateLimiter(
            tokens_per_minute=rate_limit_tpm,
            buffer=rate_limit_buffer
        ) if enable_rate_limiting else None
        
        # Display model info
        console.print(f"[dim]Rerank Model: {RERANK_MODEL}[/dim]")
        console.print(f"[dim]Generation Model: {GENERATION_MODEL}[/dim]")
        if rate_limiter:
            console.print(f"[dim]🚦 Rate Limiter: {rate_limiter.max_tokens:,} tokens/min[/dim]")
        console.print()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            console=console
        ) as progress:
            task = progress.add_task(
                "[cyan]Processing questions...",
                total=len(questions)
            )
            
            for i, q in enumerate(questions, 1):
                try:
                    # Show rate limit status
                    if rate_limiter and i % 5 == 0:
                        current, percentage = rate_limiter.get_current_usage()
                        console.print(f"  📊 Token usage: {current:,}/{rate_limiter.max_tokens:,} ({percentage}%)")
                    
                    resp = answer_question(
                        q, 
                        top_k=TOP_K_RETRIEVAL, 
                        use_rerank=USE_RERANK,
                        rate_limiter=rate_limiter
                    )
                    results.append({"question": q, "response": resp})
                    
                    # Estimate tokens (rough)
                    tokens_used = len(q.split()) + len(resp.get("answer", "").split()) * 2
                    state["total_tokens"] += tokens_used
                    
                except Exception as e:
                    error_str = str(e)
                    console.print(f"[red]✗ Error on Q{i}: {error_str[:100]}[/red]")
                    
                    # Handle rate limit errors
                    if "rate_limit" in error_str.lower() or "429" in error_str:
                        console.print(f"[yellow]⏸️  Rate limit hit. Pausing 60s...[/yellow]")
                        time.sleep(60)
                        # Retry
                        try:
                            resp = answer_question(
                                q,
                                top_k=TOP_K_RETRIEVAL,
                                use_rerank=USE_RERANK,
                                rate_limiter=rate_limiter
                            )
                            results.append({"question": q, "response": resp})
                            console.print(f"[green]✓ Retry successful[/green]")
                        except Exception as e2:
                            results.append({"question": q, "error": str(e2)})
                    else:
                        results.append({"question": q, "error": error_str})
                
                progress.update(task, advance=1)
                state["questions_processed"] = i
                
                # Save progress periodically
                save_every = int(os.getenv("SAVE_PROGRESS_EVERY", "5"))
                if i % save_every == 0:
                    temp_path = Path(state["results_path"]).parent / f"results_progress_{i}.json"
                    save_json(str(temp_path), results)
                    console.print(f"  💾 Progress saved ({i}/{len(questions)})")
        
        # Save results
        save_json(state["results_path"], results)
        
        elapsed = time.time() - stage_start
        state["stage_times"]["querying"] = elapsed
        
        console.print(f"\n[green]✓ Processed {len(results)} questions in {elapsed:.2f}s[/green]")
        console.print(f"[green]✓ Results saved to: {state['results_path']}[/green]\n")
        
        state["stage"] = "end"
        
    except Exception as e:
        state["error"] = f"Query processing failed: {str(e)}"
        state["stage"] = "end"
    
    return state

def end_node(state: PipelineState) -> PipelineState:
    """Display final summary"""
    console.print("\n" + "="*70)
    console.print("[bold magenta]📊 Pipeline Summary[/bold magenta]")
    console.print("="*70 + "\n")
    
    if state.get("error"):
        console.print(f"[red]❌ Pipeline stopped: {state['error']}[/red]\n")
    else:
        # Summary table
        table = Table(title="Execution Summary", show_header=True)
        table.add_column("Stage", style="cyan")
        table.add_column("Time (s)", style="green", justify="right")
        
        for stage, duration in state.get("stage_times", {}).items():
            table.add_row(stage.capitalize(), f"{duration:.2f}")
        
        total_time = time.time() - state["start_time"]
        table.add_row("[bold]TOTAL", f"[bold]{total_time:.2f}")
        
        console.print(table)
        
        # Statistics
        stats_table = Table(show_header=False)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Chunks Created", str(state.get("chunks_created", 0)))
        stats_table.add_row("Questions Processed", str(state.get("questions_processed", 0)))
        stats_table.add_row("Est. Tokens Used", str(state.get("total_tokens", 0)))
        
        console.print("\n", stats_table)
        
        console.print(f"\n[bold green]✅ Pipeline completed successfully![/bold green]\n")
    
    return state

# ============================================
# Graph Construction
# ============================================

def should_continue_from_check_chunks(state: PipelineState) -> Literal["approve_chunking", "check_index"]:
    """Route based on chunks existence"""
    return state["stage"]

def should_continue_from_approve_chunking(state: PipelineState) -> Literal["chunking", "end"]:
    """Route based on user approval"""
    return state["stage"]

def should_continue_from_check_index(state: PipelineState) -> Literal["approve_indexing", "approve_query"]:
    """Route based on index existence"""
    return state["stage"]

def should_continue_from_approve_indexing(state: PipelineState) -> Literal["indexing", "end"]:
    """Route based on user approval"""
    return state["stage"]

def should_continue_from_approve_query(state: PipelineState) -> Literal["querying", "end"]:
    """Route based on user approval"""
    return state["stage"]

def build_graph():
    """Build the LangGraph workflow"""
    workflow = StateGraph(PipelineState)
    
    # Add nodes
    workflow.add_node("start", start_node)
    workflow.add_node("check_chunks", check_chunks_node)
    workflow.add_node("approve_chunking", approve_chunking_node)
    workflow.add_node("chunking", chunking_node)
    workflow.add_node("check_index", check_index_node)
    workflow.add_node("approve_indexing", approve_indexing_node)
    workflow.add_node("indexing", indexing_node)
    workflow.add_node("approve_query", approve_query_node)
    workflow.add_node("querying", querying_node)
    workflow.add_node("end", end_node)
    
    # Add edges
    workflow.set_entry_point("start")
    workflow.add_edge("start", "check_chunks")
    
    workflow.add_conditional_edges(
        "check_chunks",
        should_continue_from_check_chunks,
        {
            "approve_chunking": "approve_chunking",
            "check_index": "check_index"
        }
    )
    
    workflow.add_conditional_edges(
        "approve_chunking",
        should_continue_from_approve_chunking,
        {
            "chunking": "chunking",
            "end": "end"
        }
    )
    
    workflow.add_edge("chunking", "check_index")
    
    workflow.add_conditional_edges(
        "check_index",
        should_continue_from_check_index,
        {
            "approve_indexing": "approve_indexing",
            "approve_query": "approve_query"
        }
    )
    
    workflow.add_conditional_edges(
        "approve_indexing",
        should_continue_from_approve_indexing,
        {
            "indexing": "indexing",
            "end": "end"
        }
    )
    
    workflow.add_edge("indexing", "approve_query")
    
    workflow.add_conditional_edges(
        "approve_query",
        should_continue_from_approve_query,
        {
            "querying": "querying",
            "end": "end"
        }
    )
    
    workflow.add_edge("querying", "end")
    workflow.add_edge("end", END)
    
    return workflow.compile()

def main():
    """Run the pipeline"""
    try:
        # Initialize state
        initial_state: PipelineState = {
            "stage": "start",
            "chunks_path": "",
            "questions_path": "",
            "results_path": "",
            "chroma_path": "",
            "chunks_created": 0,
            "questions_processed": 0,
            "total_questions": 0,
            "start_time": 0.0,
            "stage_times": {},
            "total_tokens": 0,
            "approved": False,
            "error": ""
        }
        
        # Build and run graph
        app = build_graph()
        final_state = app.invoke(initial_state)
        
        return final_state
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Pipeline interrupted by user[/yellow]\n")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]❌ Pipeline error: {str(e)}[/red]\n")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
