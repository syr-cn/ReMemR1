import os
from openai import OpenAI
import time


from .envs import URL, API_KEY, MAX_INPUT_LEN, MAX_OUTPUT_LEN

template_0shot = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

$Q$

Format your response as follows: "Therefore, the answer is (insert answer here)"."""
from .aio import get_async_client
async def async_query_llm(item, model, tokenizer, temperature=0.7, top_p=0.95, stop=None):
    max_input_tokens=MAX_INPUT_LEN
    max_new_tokens=MAX_OUTPUT_LEN
    context = item["context"]
    prompt = template_0shot.replace('$DOC$', context.strip()).replace('$Q$', item['input'].strip())
    session = await get_async_client()
    async with session:
        max_len = max_input_tokens
        input_ids = tokenizer.encode(prompt)
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
            prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
        try:
            async with session.post(
                url=URL + "/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json=dict(model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_new_tokens
                )
            ) as resp:
                status = resp.status
                if status!= 200:
                    print(f"{status=}, {model=}")
                    return ''
                data = await resp.json()
                return data['choices'][0]['message']['content']
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            import traceback
            traceback.print_exc()
        return ''


def query_llm(prompt, model, tokenizer, temperature=0.7, top_p=0.95, max_input_tokens=120000, max_new_tokens=10000, stop=None):
    client = OpenAI(
        base_url=URL,
        api_key=API_KEY,
        timeout=1800
    )
    max_len = max_input_tokens
    input_ids = tokenizer.encode(prompt)
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
        prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
    tries = 0
    while tries < 5:
        tries += 1
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
            )
            return completion.choices[0].message.content
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print("Error Occurs: \"%s\"        Retry ..."%(str(e)))
            time.sleep(1)
    print("Max tries. Failed.")
    return ''