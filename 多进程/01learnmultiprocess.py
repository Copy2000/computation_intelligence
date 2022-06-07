import multiprocessing
import time
import concurrent.futures
'''
1.定义一个函数
2.使用p = multiprocessing.Process(target=do_something,args=[1])定义一个进程
    1）p.start()为开始进程
    2）p.join()为执行进程
    我认为上面两个的区别是：如果只有start那么可能在本script执行完成之后再开始进程，
    但是有了join就可以实现一种中断去完成进程再回来进行本script的后续内容
'''
# def do_something(seconds):
#     print(f'Sleeping {seconds} seconds')
#     time.sleep(seconds)
#     print('Done sleeping...')
#
# if __name__ == '__main__':
#     processes=[]
#     start = time.perf_counter()
#     for i in range(10):
#         p = multiprocessing.Process(target=do_something,args=[1])
#         p.start()
#         processes.append(p)
#
#     for process in processes:
#         process.join()
#     end = time.perf_counter()
#     print(f'Time cost: {end - start} s')


#=========================
# def do_something(seconds):
#     print(f'Sleeping {seconds} seconds')
#     time.sleep(seconds)
#     return 'Done sleeping...'
#
# if __name__ == '__main__':
#     start = time.perf_counter()
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         results = [executor.submit(do_something,1) for i in range(10)]
#         for f in concurrent.futures.as_completed(results):
#             print(f.result())
#
#
#     end = time.perf_counter()
#     print(f'Time cost: {end - start} s')


#====================
def do_something(seconds):
    print(f'Sleeping {seconds} seconds')
    time.sleep(seconds)
    return 'Done sleeping...'


if __name__ == '__main__':
    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        secs = [5,4,3,2,1]
        results = executor.map(do_something,secs)
        for result in results:
            print(result)

    end = time.perf_counter()
    print(f'Time cost: {end - start} s')