from algoHelper import *


def lc_0001():
    # Two Sum
    def sol(arr, t):
        dic = {}
        for i, ele in enumerate(arr):
            if t - ele in dic:
                return [ele, t-ele]
            dic[ele] = True
        return

def lc_0002():
    pass
