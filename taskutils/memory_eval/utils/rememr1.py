from .aio import get_async_client
from utils import extract_solution
from .envs import URL, API_KEY, RECURRENT_CHUNK_SIZE, RECURRENT_MAX_NEW, RECURRENT_MAX_CONTEXT_LEN
import re

from .tf_idf_retriever import TfidfRetriever

def _preprocess_text(text: str) -> set[str]:
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Split into words and return a set of unique tokens
    return set(text.split())

TEMPLATE = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. You should generate response in the following format:
- Output your thinking process in <thinking>your_thinking_process</thinking>.
- Read the provided section carefully and update the memory with the new information that helps to answer the problem in only one <update>the_updated_memory</update> action. Be sure to retain all relevant details from the previous memory while adding any new, useful information.
- If you notice partial key evidence that is not enough to answer the problem, also output only one `<recall>query</recall>` (e.g. `<recall>who's the president of the United States?</recall>`) to retrieve information in previous memories.

<problem> 
{prompt}
</problem>

<recalled_memory>
{recalled_memory}
</recalled_memory>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Updated memory:
"""

TEMPLATE_FINAL_BOXED = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

<problem> 
{prompt}
</problem>

<recalled_memory>
{recalled_memory}
</recalled_memory>

<memory>
{memory}
</memory>

Your answer:
"""


NO_MEMORY = "No previous memory"
NO_RECALLED_MEMORY = "No memory was recalled."

def _parse_recall_query(text_response: str) -> str:
    """Extracts the chunk ID from a string like '<recall>who's the president of the United States?</recall>'."""
    try:
        match = re.search(r'<recall>(.+)</recall>', text_response)
        if match:
            query = match.group(1) # we pick the first group, which is the query
            return query
    except (ValueError, TypeError):
        pass # Fall through to return None if parsing fails
    return None

def _parse_update_memory(text_response: str) -> str:
    try:
        match = re.search(r'<update>(.+)</update>', text_response)
        if match:
            return match.group(1)
    except (ValueError, TypeError):
        pass # Fall through to return None if parsing fails
    return None

def _parse_update_memory2(text_response: str) -> str:
    try:
        # Remove <recall>...</recall> tags and their content (non-greedy to match properly)
        cleaned = re.sub(r'<recall>.*?</recall>', '', text_response, flags=re.DOTALL)
        return cleaned.strip()
    except (ValueError, TypeError):
        return None

def clip_long_string(string, max_length=2000):
    """Clip long string to a maximum length."""
    # assert max_length > 50, "max_length must be greater than 50"
    if not len(string) > max_length:
        return string
    target_len = max_length - len('\n\n...(truncated)\n\n')
    return string[:target_len//2] + '\n\n...(truncated)\n\n' + string[-target_len//2:]


async def async_query_llm(item, model, tokenizer, temperature=0.7, top_p=0.95, stop=None, nocallback=False):
    idx = item["_id"]
    context = item["context"].strip()
    prompt = item['input'].strip()
    session = await get_async_client()
    history_memory = set()
    retriever = TfidfRetriever(tokenizer=tokenizer)
    async with session:
        max_len = RECURRENT_MAX_CONTEXT_LEN
        input_ids = tokenizer.encode(context)
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
        memory = NO_MEMORY
        recall_query = None
        recalled_memory = NO_RECALLED_MEMORY
        for i in range(0, len(input_ids), RECURRENT_CHUNK_SIZE):
            chunk = input_ids[i:i+RECURRENT_CHUNK_SIZE]
            chunk_str = tokenizer.decode(chunk)

            try:
                msg = TEMPLATE.format(prompt=prompt, chunk=chunk_str, memory=memory, recalled_memory=recalled_memory)
                async with session.post(
                    url=URL + "/chat/completions",
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    json=dict(model=model,
                        messages=[{"role": "user", "content": msg}],
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=RECURRENT_MAX_NEW
                    )
                ) as resp:
                    status = resp.status
                    if status!= 200:
                        print(f"{status=}, {model=}")
                        return ''
                    data = await resp.json()
                    memory = _parse_update_memory2(data['choices'][0]['message']['content'])
                    if idx == 0:
                        print("assistant:")
                        print(clip_long_string(memory))
                    history_memory.add(memory)
                    recall_query = _parse_recall_query(data['choices'][0]['message']['content'])
                    recalled_memory = NO_RECALLED_MEMORY if recall_query is None else retriever.top1_retrieve(recall_query, history_memory)
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                import traceback
                traceback.print_exc()
                return ''

        msg_final = TEMPLATE_FINAL_BOXED.format(prompt=prompt, memory=memory, recalled_memory=recalled_memory)
        if idx == 0:
            print("user:")
            print(clip_long_string(msg_final))
        try:
            async with session.post(
                url=URL + "/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json=dict(model=model,
                    messages=[{"role": "user", "content": msg_final}],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=RECURRENT_MAX_NEW
                )
            ) as resp:
                status = resp.status
                if status!= 200:
                    print(f"{status=}, {model=}")
                    return ''
                data = await resp.json()
                if idx == 0:
                    print("assistant:")
                    print(data['choices'][0]['message']['content'])
                return data['choices'][0]['message']['content']
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            import traceback
            traceback.print_exc()
        return ''

