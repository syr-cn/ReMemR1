from operator import length_hint
import time
import asyncio

import aiohttp
SESSION: aiohttp.ClientSession | None = None
# unfortunately, aiohttp cannot reuse connection for some reason, I'm not sure if this is a bug
# 'Cannot write to closing transport' will be triggered....
# async def get_async_client():
#     global SESSION
#     if SESSION is None:
#         conn = aiohttp.TCPConnector(limit=1024, limit_per_host=1024, keepalive_timeout=600)
#         SESSION = aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=3600))
#     return SESSION
# async def close_async_client():
#     global SESSION
#     if SESSION is not None:
#         await SESSION.close()
#         SESSION = None
async def get_async_client():
    return aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=86400))
async def close_async_client():
    pass

        
async def async_main(tasks, max_workers=2048):
    start = time.time()
    import tqdm
    sem = asyncio.Semaphore(max_workers)
    with tqdm.tqdm(total=length_hint(tasks)) as pbar:
        async def _tsk(coro):
            async with sem:
                ret = await coro
                pbar.update(1)
            return ret
        tasks = [_tsk(t) for t in tasks]
        responses = await asyncio.gather(*tasks)
    end = time.time()
    print(f"Time taken: {end - start} seconds")
    return responses