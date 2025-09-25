# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import dill
import pickle
from rich import print
from operator import length_hint
import time
import asyncio
import uvloop
uvloop.install()

def map_worker_dill(arg):
    func, args, const = arg
    return dill.loads(func)(*args, **const)
def map_worker_pickle(arg):
    func, args, const = arg
    return pickle.loads(func)(*args, **const)
class Executor:
    def __init__(self, map_func, **map_func_kwargs):
        """Wrap for a multiprocessing map function.
        e.g multiprocessing.Pool.map
            concurrent.futures.ProcessPoolExecutor.map
            tqdm.contrib.concurrent.process_map

        Args:
            map_func (callable): The map function to use.
            map_func_kwargs (dict): Additional keyword arguments to pass to the map function.
        """
        self.map_func = map_func
        self.map_func_kwargs = map_func_kwargs

    def run(self, func, *iters, **const):
        """Run the map function with the given arguments.

        Args:
            func (callable): The function to map. Can be any dill-serializable function.
            iters (iterable): The iterables to map the function over.
            const (dict): Constant arguments to pass to the function.

        Returns:
            _type_: _description_
        """
        class MapArgs:
            def __init__(self, func, *iters, **const):
                assert isinstance(func, bytes), f"func must be serialized, get func={func}"
                self.func = func
                self.iters = iters
                self.const = const

            def __iter__(self):
                for args in zip(*self.iters):
                    yield (self.func, args, self.const) if self.func else (args, self.const)
            def __len__(self):
                return len(self.iters[0])
        try:
            pickled_func = pickle.dumps(func)
            print("using pickle to serialize")
            return self.map_func(map_worker_pickle, MapArgs(pickled_func, *iters, **const), **self.map_func_kwargs)
        except Exception as e:
            if "Can't pickle" not in str(e):
                raise e
            print("using dill to serialize")
            return self.map_func(map_worker_dill, MapArgs(dill.dumps(func), *iters, **const), **self.map_func_kwargs)

class TqdmExecutor(Executor):

    def __init__(self, max_workers=None, total=None, chunksize=1, **kwargs):
        """Wrap for tqdm.contrib.concurrent.process_map.

        Args:
            chunksize (int, optional): The number of tasks to assign to each worker at a time. Defaults to 1.
            max_workers (int, optional): The maximum number of workers to use. Defaults to None.
            kwargs: Additional keyword arguments to pass to the map function.
        """
        from tqdm.contrib.concurrent import process_map
        super().__init__(process_map, max_workers=max_workers, chunksize=chunksize, **kwargs)

def plus(x, y, a=0):
    return x + y + a
class Plus:
    def __init__(self):
        pass
    def __call__(self, x, y, a=0):
        return x + y + a

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

if __name__ == '__main__':
    import itertools
    result = TqdmExecutor(chunksize=1).run(lambda x, y, a: x + y + a, range(10), range(10, 20), a=1)
    print(list(result))
    result = TqdmExecutor(chunksize=1).run(plus, range(10), range(10, 20), a=1)
    print(list(result))
    result = TqdmExecutor(chunksize=1).run(Plus(), range(10), range(10, 20), a=1)
    print(list(result))
    breakpoint()
