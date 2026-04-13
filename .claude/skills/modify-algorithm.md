# Modifying the MemoryAgent Algorithm

This skill guides you through modifying the core recurrent memory algorithm in ReMemR1.

## Architecture Overview

The MemoryAgent processes long documents **chunk-by-chunk** in a recurrent loop:

```
for each chunk in document:
    1. Format prompt (TEMPLATE) with: problem, chunk, current_memory, recalled_memory
    2. Generate LLM response
    3. Parse <update>...</update> → new memory state
    4. Parse <recall>query</recall> → TF-IDF retrieval query
    5. Add memory string to history_memory set
    6. Retrieve from history_memory using TF-IDF
After all chunks:
    7. Format final prompt (TEMPLATE_FINAL_BOXED) with: problem, memory, recalled_memory
    8. Generate final answer in \boxed{}
```

## Key Files

| File | Purpose |
|---|---|
| `recurrent/impls/memory_revisit.py` | **Core agent**: MemoryAgent, MemoryConfig, MemoryDataset, templates, parsing, registration |
| `recurrent/impls/tf_idf_retriever.py` | TF-IDF retrieval using sklearn + LLM tokenizer |
| `recurrent/interface.py` | Abstract base classes: RAgent, RConfig, RDataset, RRegister, AsyncRAgent |
| `recurrent/impls/async_memory.py` | Async variant of MemoryAgent for ChatCompletionProxy rollout |
| `recurrent/utils.py` | TokenTemplate, chat_template helper, unpad utility |

## RAgent Lifecycle (interface.py)

The `RAgent` abstract class defines this lifecycle, called by the generation loop:

```
__init__(tokenizer, config)  → one-time setup (templates, retriever, constants)
start(gen_batch, timing_raw)  → per-batch init (history_memory, memory arrays, step=0)
  loop:
    action()   → build input prompts for LLM (returns List[Tensor], meta_info)
    [LLM generates responses]
    update(gen_output) → parse responses, update memory, run retrieval
    done()     → True when all chunks consumed (is_final after final turn)
end()          → cleanup, return (final_mask, sample_index)
```

Key state managed across the loop:
- `self.memory`: np.array of token tensors — current memory per sample
- `self.history_memory`: list of sets — all past memory strings per sample (retrieval corpus)
- `self.recall_memories`: np.array — last recalled memory per sample
- `self.active_mask`: bool tensor — which samples still have chunks to process
- `self.is_final`: bool — True on the last turn (final answer generation)
- `self.step`: int — current chunk index

## MemoryConfig Parameters

Defined in `memory_revisit.py` as a dataclass extending `RConfig`:

```python
@dataclass
class MemoryConfig(RConfig):
    context_key: str              # dataset column containing the long context
    max_prompt_length: int        # max tokens for the problem/question prompt
    chunk_size: int               # tokens per context chunk
    max_memorization_length: int  # max tokens for memory + LLM response per turn
    max_chunks: int               # max number of chunks to process
    max_final_response_length: int  # max tokens for the final boxed answer
```

Derived properties:
- `max_raw_input_length` = max_prompt_length + chunk_size + 2 * max_memorization_length
- `gen_max_tokens_memorization` = max_memorization_length (generation budget per chunk turn)
- `gen_max_tokens_final_response` = max_final_response_length (generation budget for final)

---

## Modification Guide

### 1. Changing the Retrieval Method

**Current**: TF-IDF with LLM tokenizer (cosine similarity), defined in `recurrent/impls/tf_idf_retriever.py`.

**Files to modify**:
- `recurrent/impls/tf_idf_retriever.py` — replace or extend `TfidfRetriever`
- `recurrent/impls/memory_revisit.py` line ~178 — `self.retriever = TfidfRetriever(tokenizer)` in `__init__`

**Steps**:
1. Create your new retriever class (e.g., `BM25Retriever`, `DenseRetriever`) with the same interface:
   - `__init__(self, tokenizer)` — initialize with the LLM's tokenizer
   - `retrieve(self, query, corpus, top_k=3)` → `List[Tuple[str|None, float]]`
   - `top1_retrieve(self, query, corpus)` → `str|None`
2. Import it in `memory_revisit.py`
3. Replace `self.retriever = TfidfRetriever(tokenizer)` with your new retriever
4. The retriever is called in `update()` at line ~271:
   ```python
   recalled_memories = [self.retriever.top1_retrieve(query, self.history_memory[idx]) ...]
   ```
   If your retriever returns differently, update this call accordingly.

**Example — BM25 retriever**:
```python
# recurrent/impls/bm25_retriever.py
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def _tokenize(self, text):
        return self.tokenizer.tokenize(text.lower())
    
    def retrieve(self, query, corpus, top_k=3):
        if not query or not corpus:
            return [(None, 0.0)] * top_k
        corpus_list = list(corpus)
        tokenized = [self._tokenize(doc) for doc in corpus_list]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(self._tokenize(query))
        top_ids = scores.argsort()[::-1][:top_k]
        return [(corpus_list[i], scores[i]) for i in top_ids]
    
    def top1_retrieve(self, query, corpus):
        return self.retrieve(query, corpus, top_k=1)[0][0]
```

### 2. Modifying the Prompt Templates

**Current**: Two templates in `memory_revisit.py`:
- `TEMPLATE` (lines 116-137) — per-chunk processing template with `{prompt}`, `{recalled_memory}`, `{memory}`, `{chunk}` placeholders
- `TEMPLATE_FINAL_BOXED` (lines 140-155) — final answer template with `{prompt}`, `{recalled_memory}`, `{memory}` placeholders

**Files to modify**:
- `recurrent/impls/memory_revisit.py` — the template strings directly

**Steps**:
1. Edit `TEMPLATE` and/or `TEMPLATE_FINAL_BOXED` string constants
2. Keep the format placeholders (`{prompt}`, `{memory}`, `{recalled_memory}`, `{chunk}`) — they are filled in `action()`
3. If you add new placeholders, update `action()` where `.format()` is called (~line 215-247)
4. The templates are wrapped in `TokenTemplate` in `__init__` for token-level manipulation:
   ```python
   self.token_message_template = TokenTemplate(self.chat_template.format(message=TEMPLATE), tokenizer)
   ```
5. If template length changes significantly, verify `self.max_input_length` calculation still has headroom

**Common modifications**:
- Add Chain-of-Thought instructions (already has `<thinking>` in TEMPLATE)
- Change the `<update>` / `<recall>` tag format
- Add structured output instructions (JSON, XML)
- Modify the final answer format (e.g., remove `\boxed{}` requirement)

**Warning**: If you change `<update>` or `<recall>` tag names, you MUST also update the parsing functions:
- `_parse_recall_query()` — regex: `r'<recall>(.+)</recall>'`
- `_parse_update_memory()` — regex: `r'<recall>.*?</recall>'` (strips recall tags to get memory)

### 3. Changing Chunk Processing Logic

**Current**: Documents are pre-chunked by token count (`chunk_size` in MemoryConfig). Chunks are stored in `gen_batch` and iterated via `self.step` in `action()`.

**Files to modify**:
- `recurrent/impls/memory_revisit.py` — `action()` method and possibly `MemoryDataset.__getitem__()`

**Steps to change chunking strategy**:
1. To change chunk size: modify `chunk_size` in config YAML (no code change needed)
2. To change how chunks are created (e.g., sentence-based, paragraph-based):
   - Modify `MemoryDataset.__getitem__()` which tokenizes and pads context
   - The context is split into fixed-size token chunks at dataset level
3. To change chunk iteration order (e.g., skip irrelevant chunks):
   - Modify `action()` where `chunk_i` is extracted from `gen_batch.batch['input_ids']`
   - The `active_mask` controls which samples still have chunks; modify the termination condition
4. To process multiple chunks per turn:
   - Modify `action()` to concatenate multiple chunk tensors
   - Update `max_input_length` calculation to accommodate larger inputs

**Key variables in `action()`**:
- `self.step` — current chunk index (0-indexed)
- `self.active_mask` — bool tensor, True for samples with remaining chunks
- `self.is_final` — set True when step >= max_chunks or all chunks consumed
- `gen_batch.batch['input_ids']` — shape `[bsz, max_chunks, chunk_size]`, the pre-chunked context

### 4. Modifying Memory Update Logic

**Current**: The LLM's full response (with `<recall>` tags stripped) becomes the new memory. All past memory strings are stored in `history_memory` (a set per sample).

**Files to modify**:
- `recurrent/impls/memory_revisit.py` — `update()` and `update_memory()` methods

**How memory currently flows**:
```
LLM response → _parse_update_memory() strips <recall> tags → full text = new memory
                                                            → tokenized → self.memory[i]
              → _parse_recall_query() extracts query       → TF-IDF on history_memory
              → update_memory() adds string to history_memory set
```

**Steps for common changes**:

**a) Extract only `<update>` tag content as memory** (instead of full response):
```python
def _parse_update_memory(self, text_response: str) -> str:
    try:
        match = re.search(r'<update>(.*?)</update>', text_response, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
    except (ValueError, TypeError):
        pass
    return None
```

**b) Limit history_memory size** (e.g., sliding window):
```python
def update_memory(self, memory_strings, active_indices):
    for idx, memory in zip(active_indices, memory_strings):
        mem = memory if memory is not None else self.NO_MEMORY_STRING
        self.history_memory[int(idx)].add(mem)
        # Keep only last N memories
        if len(self.history_memory[int(idx)]) > MAX_HISTORY:
            oldest = list(self.history_memory[int(idx)])[0]
            self.history_memory[int(idx)].discard(oldest)
```
Note: sets are unordered; if you need ordering, change `history_memory` from `set` to `list` (update `start()` too).

**c) Change memory data structure** (e.g., set → list for ordered history):
- `start()`: change `self.history_memory = [set() ...]` → `[[] ...]`
- `update_memory()`: change `.add()` → `.append()`
- Retriever: `self.history_memory[idx]` is passed to `top1_retrieve()` as corpus — ensure your retriever handles the new type

### 5. Changing the Callback Mechanism

**Current**: The LLM outputs `<recall>query</recall>` in its response. The `update()` method parses this, runs TF-IDF retrieval on `history_memory`, and stores the result in `self.recall_memories`. On the **next** turn, `action()` formats the recalled memory into the prompt's `{recalled_memory}` slot.

**Files to modify**:
- `recurrent/impls/memory_revisit.py` — `_parse_recall_query()`, `update()`, `action()`, and templates

**The callback flow**:
```
Turn N:
  action()  → formats prompt with recalled_memory from turn N-1
  LLM       → generates response with <recall>query</recall>
  update()  → parses <recall> → retriever.top1_retrieve(query, history_memory)
           → stores result in self.recall_memories[i]

Turn N+1:
  action()  → uses self.recall_memories[i] as {recalled_memory} in template
```

**Steps for common changes**:

**a) Add multiple recall queries per turn**:
1. Modify `_parse_recall_query()` to return a list (use `re.findall` instead of `re.search`)
2. Modify `update()` to call retriever for each query and concatenate results
3. Update template to explain multiple `<recall>` tags are allowed

**b) Replace callback with a different mechanism** (e.g., attention-based retrieval):
1. Remove `<recall>` from template instructions
2. In `update()`, replace the recall parsing + retrieval with your mechanism
3. Still store results in `self.recall_memories` (or remove it and update `action()`)

**c) Add new callback types** (e.g., `<search>`, `<delete>`):
1. Add new parsing functions similar to `_parse_recall_query()`
2. Add new state arrays in `start()` similar to `self.recall_memories`
3. Process them in `update()` after response parsing
4. Update template to describe the new actions
5. Update `action()` to include new state in the prompt formatting

### 6. Adding a New Agent Variant

**Files to modify**:
- Create a new file in `recurrent/impls/` (e.g., `my_agent.py`)
- Reference it in your config YAML via `recurrent.path`

**Steps**:
1. Create your agent file with:
   - A config dataclass extending `RConfig`
   - A dataset class extending `RDataset`
   - An agent class extending `RAgent` (implement all abstract methods)
   - A `REGISTER` variable at module level: `REGISTER = RRegister(config_cls=..., dataset_cls=..., agent_cls=...)`
2. The registration system uses `recurrent.path` in config to find your module and loads `REGISTER` (or custom name via `recurrent.name`)
3. Your config YAML should set:
   ```yaml
   recurrent:
     path: recurrent.impls.my_agent
     # name: REGISTER  # default, change if your variable has a different name
   ```

---

## Testing Approach

### Unit Testing Individual Components

**Test retriever changes**:
```python
from transformers import AutoTokenizer
from recurrent.impls.tf_idf_retriever import TfidfRetriever

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
retriever = TfidfRetriever(tokenizer)

corpus = {"The cat sat on the mat", "Dogs love to play fetch", "Birds fly south in winter"}
result = retriever.top1_retrieve("What do dogs like?", corpus)
print(result)  # Should return the dogs sentence
```

**Test template parsing**:
```python
from recurrent.impls.memory_revisit import MemoryAgent

# Test _parse_recall_query
agent = MemoryAgent.__new__(MemoryAgent)  # skip __init__
query = agent._parse_recall_query("Some thinking <recall>test query</recall> more text")
assert query == "test query"

# Test _parse_update_memory
memory = agent._parse_update_memory("Memory content <recall>query</recall> more memory")
assert "query" not in memory
assert "Memory content" in memory
```

### Integration Testing

Run a minimal rollout to verify the full loop works after changes:

```bash
# Use a small model and short context for quick testing
# Adjust the config to use small chunk_size, max_chunks=2-3
python -m recurrent.main \
    --config configs/your_test_config.yaml \
    --model Qwen/Qwen2.5-3B \
    --data path/to/small_test_data.parquet
```

### Debugging Tips

1. **Enable logging**: The `log_step()` method in MemoryAgent prints detailed per-step info (message, response). Check logs for `[RECURRENT] STEP` markers.
2. **Check memory evolution**: Add `logger.info(f"Memory: {self.tokenizer.decode(self.memory[0])}")` in `update()` to track memory state.
3. **Verify retrieval**: Add `logger.info(f"Recalled: {recalled_memories[0]}")` in `update()` to see what's being retrieved.
4. **Token length issues**: If generation is truncated, check `max_memorization_length` and `max_input_length`. The `max_raw_input_length` property must accommodate all template slots.

---

## Quick Reference: What to Change for Common Goals

| Goal | Primary File | Key Function/Section |
|---|---|---|
| Replace TF-IDF with dense retrieval | `tf_idf_retriever.py` + `memory_revisit.py:__init__` | `TfidfRetriever` class, `self.retriever = ...` |
| Change prompt format | `memory_revisit.py` | `TEMPLATE`, `TEMPLATE_FINAL_BOXED` strings |
| Change chunk size | Config YAML | `chunk_size` parameter |
| Change chunking strategy | `memory_revisit.py` | `MemoryDataset.__getitem__()` |
| Add memory compression | `memory_revisit.py` | `update()`, `_parse_update_memory()` |
| Limit history size | `memory_revisit.py` | `update_memory()` |
| Change callback tags | `memory_revisit.py` | `TEMPLATE`, `_parse_recall_query()`, `_parse_update_memory()` |
| Add new callback actions | `memory_revisit.py` | `update()`, `start()`, `action()`, template |
| Change final answer format | `memory_revisit.py` | `TEMPLATE_FINAL_BOXED` |
| Create new agent type | New file in `recurrent/impls/` | Extend `RAgent`, define `REGISTER` |
| Async variant changes | `recurrent/impls/async_memory.py` | Mirror changes from sync MemoryAgent |

