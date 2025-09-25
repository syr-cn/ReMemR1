import ray
import asyncio
ray.init()
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pprint import pprint
pprint(ray.nodes()) # just to show
FIB_NUM = 32
def compute_score_1(data):
    def some_cpu_heavy_func(data):
        def fib(n):
            if n <= 1:
                return n
            else:
                return fib(n-1) + fib(n-2)
        fib(FIB_NUM)
        print("done", data)
    # this function can be non-pickleable
    p = multiprocessing.Process(target=some_cpu_heavy_func, args=(data, ))
    p.start()
    p.join()


def compute_score_2(data):
    print("2", data)

def get_custom_reward_score(choice):
    if choice == 1:
        return compute_score_1
    else:
        return compute_score_2

@ray.remote(max_concurrency=1)
class ConcurrentRewardWorker:
    def __init__(self):
        pass

    def call(self, func, *args, **kwds):
        return func(*args, **kwds)
    

class RewardManager:
    def __init__(self, choice):
        self.func = get_custom_reward_score(choice)
        # 获取Ray集群中所有节点的CPU资源，假设是同构资源
        self.node_to_cpus = {}
        nodes = ray.nodes()
        for node in nodes:
            resources = node.get("Resources", {})
            if "CPU" in resources:
                node_ip = node.get("NodeManagerAddress", "unknown")
                self.node_to_cpus[node_ip] = int(resources["CPU"] - resources.get("GPU", 0))

        # 为每个节点创建worker，并发度设置为节点的CPU数量
        self.workers = []
        for node_ip, cpu_count in self.node_to_cpus.items():
            worker = ConcurrentRewardWorker.options(
                num_cpus=1, # 虚假的，反正ray管不了我们用多少cpu，不能用真实值防止超发
                resources={f"node:{node_ip}": 0.1}, # 手动绑定节点亲和性
                max_concurrency=cpu_count
            ).remote()
            self.workers.append(worker)
    @property
    def lb_index(self):
        def load_balancer_index():
            # 创建资源计数器副本
            cnts = list(self.node_to_cpus.values())
            while any(c > 0 for c in cnts):
                for i in range(len(cnts)):
                    if cnts[i] > 0:
                        yield i
                        cnts[i] -= 1
        if not hasattr(self, '_lb_index'):
            self._lb_index = list(load_balancer_index())
        return self._lb_index
        
    
    def call(self, data_list):
        async def worker(i):
            # 使用round-robin方式分配任务到不同worker，如果是异构，就需要针对性修改了
            selected_worker = self.workers[self.lb_index[i % len(self.workers)]]
            await selected_worker.call.remote(self.func, data_list[i])
            
        start = time.time()
        async def main():
            tasks = [worker(i) for i in range(len(data_list))]
            await asyncio.gather(*tasks)
        asyncio.run(main())
        end = time.time()
        print("time cost ray+process", end - start)
        return worker

def pickleable_fib(n):
    if n <= 1:
        return n
    else:
        return fib(n-1) + fib(n-2)
if __name__ == "__main__":
    import os
    start = time.time()
    def fib(n):
        if n <= 1:
            return n
        else:
            return fib(n-1) + fib(n-2)
    fib(FIB_NUM)
    end = time.time()
    print("time cost single", end - start)

    # 测试多进程
    start = time.time()
    processes = []
    for i in range(os.cpu_count()):
        p = multiprocessing.Process(target=fib, args=(FIB_NUM, ))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    end = time.time()
    print("time cost multi", end - start)

    # 测试进程池
    start = time.time()
    with multiprocessing.Pool(os.cpu_count()) as pool:
        pool.map(fib, [FIB_NUM] * os.cpu_count())
    end = time.time()
    print("time cost pool", end - start)

    # 测试线程+子进程
    start = time.time()
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(compute_score_1, [FIB_NUM] * os.cpu_count()))
    end = time.time()
    print("time cost thread+process", end - start)

    # 测试线程
    start = time.time()
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(fib, [FIB_NUM] * 20))
    end = time.time()
    print("time cost thread", end - start)


    caller = RewardManager(1)
    caller.call([i for i in range(os.cpu_count())])