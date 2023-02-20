import math
import random
import copy
import collections
import functools
import time
from sortedcontainers import SortedDict
import heapq

from typing import Set, Dict, List

# 装饰器
def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        return value
    return wrapper_timer

# 数据结构
class ListNode:
    def __init__(self, val=0, next_=None):
        self.val = val
        self.next = next_

class BinaryTree:
    def __init__(self, value, left=None, right=None, parent=None):
        '''
        二叉树类，若不写parent即为没有父节点的二叉树，写了parent就是有父节点的二叉树
        :param value:
        :param left:
        :param right:
        :param parent:
        '''
        self.value = value
        self.left = left
        self.right = right
        self.parent = parent


class Graph:
    def __init__(self):
        '''
        图实现，这个比较综合的类定义是为了处理online coding一般图问题的实现，每一个图可以看作是一堆顶点和一堆边的集合，我们这里的顶点使用哈希表
        记录，哈希key是一个顶点的值，value为该值对应的顶点，这里需要保证每个顶点的值不同，否则需要想办法处理
        nodes: 节点列表： { 节点值：节点object }
        edges: 边列表 { 边object }
        :param:
        '''

        self.nodes: Dict[int, Node] = dict()
        self.edges: Set[Edge] = set()

    def __str__(self):
        return f"{self.__class__}_{id(self)}"

    def __repr__(self) -> str:
        return str(self)


class Node:
    def __init__(self, value: int = float("inf")):
        '''
        图的顶点实现，这个比较综合的类定义是为了处理online coding一般图问题的实现，图的每个顶点包含一些关于这个点的额外信息
        value: 当前节点的值
        in_: 顶点的边入度：以这个节点为终点的边有几个
        out_: 顶点的边出度：以这个节点为起点的边有几个
        nexts: 从这个节点出发的边连接的相邻点的list
        edges: 从这个节点出发的边的list
        :param value:
        '''
        self.value: int = value
        self.in_: int = 0
        self.out_: int = 0
        self.nexts: List[Node] = []
        self.edges: List[Edge] = []

    def __str__(self):
        return f"{self.__class__}_{self.value}_{id(self)}"

    def __repr__(self) -> str:
        return str(self)

    # def __eq__(self, other):
    #     return self == other

    def __hash__(self):
        return hash(str(self))


class Edge:
    def __init__(self, weight: int, from_: Node, to_: Node):
        '''
        图的边实现，这个比较综合的类定义是为了处理online coding一般图问题的实现，图的每条边有一个权重（边长），还有两个顶点 from 和 to，可以
        实现无向图和有向图，两条边可以比较大小，需要重载lt和eq魔术函数使用边的权重作为大小比较的基准
        :param weight: 当前这条边的权重(权重图的时候会用到)
        :param from_: 当前这条边的起点
        :param to_: 当前这条边的终点
        '''
        self.weight: int = weight
        self.from_: Node = from_
        self.to_: Node = to_

    def __str__(self):
        '''
        print(object) 表达的内容
        :return:
        '''
        return f"{self.__class__}_{str(self.weight)}_{str(self.from_.value)}_{str(self.to_.value)}_{id(self)}"

    def __hash__(self):
        '''
        如果你打算把这个object放在set中或作为字典的key那么需要定义magic __hash__
        :return:
        '''
        return hash(str(self))

    def __lt__(self, other):
        ## 用于权重比较，排序
        return self.weight < other.weight

    def __eq__(self, other):
        ## not for hashing,
        return self.weight == other.weight
    
    def __repr__(self) -> str:
        return str(self)


class mySets:
    '''
    简易版union find
    '''

    def __init__(self, nodes: Set[Node]):
        self.setMap: Dict[Node: Set[Node]] = {}
        for cur in nodes:
            self.setMap[cur] = set(cur)

    def inSameSet(self, nfrom: Node, nto: Node) -> bool:
        return self.setMap[nto] == self.setMap[nfrom]

    def union(self, nfrom: Node, nto: Node):
        self.setMap[nfrom].update(self.setMap[nto])
        for nd in self.setMap[nto]:
            self.setMap[nd] = self.setMap[nfrom]
        return


class Element:
    def __init__(self, val):
        '''
        对任意数据的包裹类
        :param val:
        '''
        self.val_ = val


class UnionFind:
    def __init__(self, val_lst):
        '''
        Union Find 数据结构python实现，包含path-compression优化
        Union Find叫并查集，记录哪几个元素属于同一个集合，以及集合的大小
        提供union,find以及查询两个元素是否属于同一个集合的方法
        '''
        # 元素值 -> 元素本身,这里假设元素值是独一无二的整数
        self.eleMap: Dict[int, Element] = dict()
        # 元素 -> 上一级元素(父)
        self.fatherMap: Dict[Element, Element] = dict()
        # 集合代表元素 -> 该代表元素代表的集合的大小
        self.sizeMap: Dict[Element, int] = dict()
        for val in val_lst:
            # 刚开始每个元素各自成为一个集合，代表元素为他自己
            ele = Element(val)
            self.eleMap[val] = ele
            self.fatherMap[ele] = ele
            self.sizeMap[ele] = 1
        return

    def findRepresentative(self, ele: Element) -> Element:
        '''
        给一个元素，返回这个元素的集合代表元素,同时进行路径压缩优化
        :param ele:
        :return:
        '''
        from collections import deque
        visited_ele = deque([])
        # 通过fathermap不断向上一级父亲寻找代表元素
        while self.fatherMap[ele] is not ele:
            visited_ele.append(ele)
            ele = self.fatherMap[ele]

        # 路径压缩，通过某一条路径找到最终代表元素之后，把这条路上每一个元素的代表元素通过fathermap设置成最终的代表元素
        while len(visited_ele) > 0:
            self.fatherMap[visited_ele.pop()] = ele

        return ele

    def isSameSet(self, val1, val2) -> bool:
        '''
        给两个元素值，判断这两个元素是否属于同一个集合
        :param ele1:
        :param ele2:
        :return:
        '''
        ret = False
        # 如果两个值所对应的元素的代表元素相同，则这两个值属于同一个集合
        if val1 in self.eleMap and val2 in self.eleMap:
            ele1_rep = self.findRepresentative(self.eleMap[val1])
            ele2_rep = self.findRepresentative(self.eleMap[val2])
            ret = ele1_rep == ele2_rep
        return ret

    def union(self, val1, val2):
        '''
        给两个元素，合并这两个元素所属的集合
        :param val1:
        :param val2:
        :return:
        '''
        if val1 in self.eleMap and val2 in self.eleMap:
            ele1F = self.findRepresentative(self.eleMap[val1])
            ele2F = self.findRepresentative(self.eleMap[val2])
            if ele1F is not ele2F:
                eleBig = ele1F if self.sizeMap[ele1F] > self.sizeMap[ele2F] else ele2F
                eleSmall = ele2F if eleBig is ele1F else ele1F
                self.fatherMap[eleSmall] = eleBig
                self.sizeMap[eleBig] += self.sizeMap[eleSmall]
                del self.sizeMap[eleSmall]
        return


class UnionFindCheck:
    def __init__(self, val_lst):
        self.eleMap: Dict[int, Element] = dict()
        self.fatherMap: Dict[Element, Element] = dict()
        self.sizeMap: Dict[Element, int] = dict()
        for val in val_lst:
            ele = Element(val)
            self.eleMap[val] = ele
            self.fatherMap[ele] = ele
            self.sizeMap[ele] = 1

    def findRepresentative(self, ele):
        from collections import deque
        q = deque([])
        while self.fatherMap[ele] != ele:
            ele = self.fatherMap[ele]
            q.append(ele)

        # 压缩
        while len(q) > 0:
            poped = q.popleft()
            self.fatherMap[poped] = ele
        return ele

    def isSameSet(self, val1, val2):
        if val1 in self.eleMap and val2 in self.eleMap:
            return self.findRepresentative(self.eleMap[val1]) == self.findRepresentative(self.eleMap[val2])
        else:
            return False

    def union(self, val1, val2):
        if val1 in self.eleMap and val2 in self.eleMap:
            rep1 = self.findRepresentative(self.eleMap[val1])
            rep2 = self.findRepresentative(self.eleMap[val2])
            bigger = rep1 if self.sizeMap[rep1] > self.sizeMap[rep2] else rep2
            smaller = rep2 if bigger == rep1 else rep1
            self.fatherMap[smaller] = bigger
            self.sizeMap[bigger] += self.sizeMap[smaller]
            del self.sizeMap[smaller]


class TrieNode:
    def __init__(self):
        '''
        前缀树的节点，构造树的时候字符串从左向右，从第一个节点出发，如果有现成的路就走现成的路，没有就新建路，更新pass和end，
        :param pass_: 构造前缀树的时候当前节点通过过几次
        :param end_: 当前节点是否是字符串的结尾节点，如果是的话是多少个字符串的结尾节点
        :param nexts_: 因为是字符串所以先初始化26个坑，None代表没有走向各自字母的路，非None代表有路
        '''
        self.pass_: int = 0
        self.end_: int = 0
        self.nexts_: List[TrieNode] = [None] * 26
        # self.nexts2_: Dict[str, TrieNode] = {} # 如果字符太多用hashtable


class Trie:
    # 前缀树：用途：存储字符串，查询字符串，复杂度为被查询字符串的长度；查询前缀为特定字符串的字符串数量
    def __init__(self):
        # 初始化一个前缀树节点作为根
        self.root_ = TrieNode()

    def insert(self, word):
        if not word:
            return
        chars = list(word)
        node = self.root_
        node.pass_ += 1
        for i in range(len(chars)):
            index = ord(chars[i]) - ord('a')
            if not node.nexts_[index]:
                node.nexts_[index] = TrieNode()
            node = node.nexts_[index]
            node.pass_ += 1
        node.end_ += 1

    def search(self, word: str) -> int:
        '''
        查询word这个单词之前加入过几次
        :param word:
        :return: int
        '''
        charArr = list(word)
        node = self.root_
        for char_ in charArr:
            index = ord(char_) - ord("a")
            if not node.nexts_[index]:
                return 0
            node = node.nexts_[index]
        return node.end_

    def prefixNum(self, pre) -> int:
        '''
        查询有多少个字符串以pre为前缀
        :param pre:
        :return: int
        '''
        preArr = list(pre)
        node = self.root_
        for char_ in preArr:
            index = ord(char_) - ord("a")
            if not node.nexts_[index]:
                return 0
            node = node.nexts_[index]
        return node.pass_

    def delete(self, word):
        '''
        删除某个字符串的记录
        :param word:
        :return:
        '''
        if not self.search(word):
            return

        charArr = list(word)
        node = self.root_

        for char_ in charArr:
            index = ord(char_) - ord('a')
            reducePass = node.nexts_[index].pass_ - 1
            if reducePass == 0:
                node.nexts_[index] = None
                return
            node.nexts_[index].pass_ -= 1
            node = node.nexts_[index]
        node.end_ -= 1
        return


class Meeting:
    def __init__(self, start: int, end: int):
        self.start_ = start
        self.end_ = end

    def __lt__(self, other):
        return self.end_ < other.end_

    def __eq__(self, other):
        return self.end_ == other.end_


class MaxHeap:
    def __init__(self, data=None):
        if not data:
            self.data_ = []
        else:
            self.data_ = [-i for i in data]
        heapq.heapify(self.data_)

    def heappop(self):
        return -heapq.heappop(self.data_)

    def heappush(self, data_ele):
        heapq.heappush(self.data_, -data_ele)
        return

    def peek(self):
        return -self.data_[0]


class MinHeap:
    def __init__(self, data=None):
        if not data:
            self.data_ = []
        else:
            self.data_ = data
        heapq.heapify(self.data_)

    def heappop(self):
        return heapq.heappop(self.data_)

    def heappush(self, data_ele):
        heapq.heappush(self.data_, data_ele)
        return

    def peek(self):
        return self.data_[0]

from collections import deque

class MinValueFuncStack:
    def __init__(self) -> None:
        self._stack = deque([])
        self._minStack = deque([])
        return
    
    def heappush(self, ele):
        self._stack.append(ele)
        if len(self._minStack) == 0:
            self._minStack.append(ele)
        elif ele < self._minStack[-1]:
            self._minStack.append(ele)
        else:
            self._minStack.append(self._minStack[-1])
        return

    def heappop(self, ele):
        _ = self._minStack.pop()
        return self._stack.pop()
         

    def peekSmallest(self):
        return self._minStack[-1]

class MinCostProject:
    def __init__(self, cost, profit):
        self.cost_ = cost
        self.profit_ = profit

    def __lt__(self, other):
        return self.cost_ < other.cost_

    def __eq__(self, other):
        return self.cost_ == other.cost_


class MaxProfitProject(MinCostProject):
    def __lt__(self, other):
        return self.profit_ > other.profit_

    def __eq__(self, other):
        return self.profit_ == other.profit_


class NodeRecord:
    def __init__(self, node: Node, distance: int):
        self.node_ = node
        self.distance_ = distance


class NodeHeap:
    # 自定义的小根堆，在堆内元素改变的时候可以实现动态调整的堆实现,用于改进的dijkstra算法，这个heap以各个node到head的距离作为比较依据
    # 初始化时要预置一个长度的list，堆的大小最多不能超过这个值
    def __init__(self, size: int):
        # 真正的堆
        self.nodes_: List[Node] = [None for i in range(size)]
        # Node在堆中的位置 Node -> Idx
        self.heapIndexMap_: Dict[Node, int] = dict()
        # Node到head目前的最短距离
        self.distanceMap_: Dict[Node, int] = dict()
        # 堆中元素的数量
        self.size_ = 0

    def isEmpty(self) -> bool:
        '''
        堆是否为空
        :return:
        '''
        return self.size_ == 0

    def isEntered(self, node: Node) -> bool:
        '''
        判断Node进没进来过heap，进来过有key，但是他的值不一定是有效的，因为node还可能会被弹出，
        弹出时，需要将index改为-1
        :param node:
        :return:
        '''
        return node in self.heapIndexMap_

    def inHeap(self, node: Node) -> bool:
        '''
        Node现在是不是在heap里，如果在，首先得有key，其次对应的index需要不是-1
        :param node:
        :return:
        '''
        return self.isEntered(node) and self.heapIndexMap_[node] != -1

    def swap(self, idx1: int, idx2: int):
        '''
        交换堆中两个node，首先把node对应的index交换
        然后在list里把node本身交换
        :param idx1:
        :param idx2:
        :return:
        '''
        self.heapIndexMap_[self.nodes_[idx1]] = idx2
        self.heapIndexMap_[self.nodes_[idx2]] = idx1
        self.nodes_[idx1], self.nodes_[idx2] = self.nodes_[idx2], self.nodes_[idx1]
        return

    def insertHeapify(self, node: Node, index: int):
        '''
        试图将插入的node向根节点移动
        :param node:
        :param index:
        :return:
        '''
        self.nodes_[index] = node
        while self.distanceMap_[self.nodes_[index]] < self.distanceMap_[self.nodes_[int((index - 1) / 2)]]:
            self.swap(index, int((index - 1) // 2))
            index = int((index - 1) / 2)
        return

    def heapify(self, index: int, size: int):
        left = 2 * index + 1
        while left < size:
            smallest = left + 1 if (left + 1 < size and self.distanceMap_[self.nodes_[left + 1]] < self.distanceMap_[
                self.nodes_[left]]) else left
            smallest = self.heapIndexMap_[self.nodes_[smallest]] if self.distanceMap_[self.nodes_[smallest]] < \
                                                                   self.distanceMap_[self.nodes_[index]] else \
            self.heapIndexMap_[self.nodes_[index]]
            if smallest == index:
                break

            self.swap(index, smallest)
            index = smallest
            left = 2 * index + 1
        return

    def addOrUpdateOrIgnore(self, node: Node, distance: int):
        if self.inHeap(node):
            self.distanceMap_[node] = min(distance, self.distanceMap_[node])
            self.insertHeapify(node, self.heapIndexMap_[node])
        elif not self.isEntered(node):
            self.nodes_[self.size_] = node

            self.distanceMap_[node] = distance
            self.heapIndexMap_[node, self.size_]
            self.insertHeapify(node, self.size_)
            self.size_ += 1
        return

    def pop(self) -> NodeRecord:
        nodeRecord: NodeRecord = NodeRecord(self.nodes[0], self.distanceMap_[self.nodes[0]])
        self.swap(0, self.size_ - 1)
        self.heapIndexMap_[self.nodes[self.size_ - 1]] = -1
        del self.distanceMap_[self.nodes[self.size_ - 1]]
        self.nodes_[self.size_ - 1] = None
        self.size_ -= 1
        self.heapify(0, self.size_)
        return nodeRecord


class RandomPool:
    def __init__(self):
        '''
        RandomPool 结构，insert, delete常数时间， 等概率随机返回一个key常数时间
        '''
        self.key2idx: Dict[str, int] = dict()
        self.idx2key: Dict[int, str] = dict()
        self.size = 0

    def insert(self, key):
        if key not in self.key2idx:
            self.key2val[key] = self.size
            self.val2key[self.size] = key
            self.size += 1

    def delete(self, deletekey):
        if deletekey in self.key2idx:
            last_idx = self.size
            last_key = self.idx2key[last_idx]
            deleteIdx = self.key2idx[deletekey]
            self.key2idx[last_key] = self.key2idx[deletekey]
            self.idx2key[deleteIdx] = last_key
            del self.key2idx[deletekey]
            del self.idx2key[last_idx]
            self.size -= 1
        return

    def getRandom(self) -> str:
        if self.size == 0:
            return None
        idx = random.randint(0, self.size)
        return self.idx2key[idx]

class stackConstructedQueue:
    '''
    用栈实现队列
    '''
    def __init__(self) -> None:
        self._c = deque([])
        self._p = deque([])
        return

    def pop(self):
        if len(self._p) == 0 and len(self._c) == 0:
            return
        if len(self._p) == 0:
            while self._c:
                self._p.append(self._c.pop())
        self._p.pop()
    
    def push(self, ele):
        self._c.append(ele)

class queueConstructedStack:
    '''
    用队列实现栈
    '''
    def __init__(self) -> None:
        self._q1 = deque([])
        self._q2 = deque([])
        return
    
    def pop(self):
        q_e = self._q1 if len(self._q1) == 0 else self._q2
        q_f = self._q2 if len(self._q1) == 0 else self._q1
        while len(q_f) > 1:
            q_e.appendleft(q_f.popleft())
        ele = q_f.popleft()
        return ele

    def push(self, ele):
        if len(self._q1) == 0:
            self._q2.append(ele)
        else:
            self._q1.append(ele)
        return

class MonoQueue:
    '''
    单调队列实现,用一个双向队列当内部容器，假设最左到最右是降序排列
    '''
    def __init__(self):
        self.q = deque([])

    def pop(self, val):
        # 要删掉的元素如果大于当前单调队列里的最大值，则单调队列里删掉最大值(最左边)
        while self.q and val == self.q[0]:
            self.q.popleft()
        return

    def push(self, val):
        # 要push进来的值如果大于单调队列里的最小值（最右），需要把单调队列里的值删掉，直到单调队列里的值都是大于新push进来的值为止，然后才能把
        # 新值push进来
        while self.q and val > self.q[-1]:
            self.q.pop()
        self.q.append(val)
        return

    def front(self):
        return self.q[0]

# 二叉树打印code
def printBinaryTree(root: BinaryTree):
    def printBinaryTreeInOrder(head:BinaryTree, height:int, decoration:str, length:int):
        def getSpace(num: int):
            return " " * num

        if not head:
            return
        printBinaryTreeInOrder(head.right, height + 1, "v", length)
        val = f"{decoration}{head.value}{decoration}"
        lenM = len(val)
        lenL = (length - lenM) >> 1
        lenR = length - lenM - lenL
        val = f"{getSpace(lenL)}{val}{getSpace(lenR)}"
        print(f"{getSpace(height * length)}{val}")
        printBinaryTreeInOrder(head.left, height + 1, "^", length)
        return

    printBinaryTreeInOrder(root, 0, "H", 17)


def dailyCheck():
    # # 树的反序列化和对称性
    # string_1 = "_".join([str(i) for i in [1,2,3,4,5,"#",7,"#","#","#","#","#","#"]])+"_"
    # string_2 = "_".join([str(i) for i in [1,3,2,7,"#",5,4,"#","#","#","#","#","#"]])+"_"
    # string_3 = "_".join([str(i) for i in [1,2,3,4,5,"#",7,"#","#","#","#","#","#"]])+"_"
    # tree1 = treeLevelOrderDeSerialization(string_1)
    # tree2 = treeLevelOrderDeSerialization(string_2)
    # tree3 = treeLevelOrderDeSerialization(string_3)

    # tree4 = treeLevelOrderDeSerializationCheck(string_1)
    # tree5 = treeLevelOrderDeSerializationCheck(string_2)
    # tree6 = treeLevelOrderDeSerializationCheck(string_3)
    # serialized = [tree1, tree2, tree3]
    # serializedCheck = [tree4, tree5, tree6]
    # print("Tree 1")
    # printBinaryTree(tree1)
    # print("Tree 2")
    # printBinaryTree(tree2)
    # print("Tree 3")
    # printBinaryTree(tree3)
    # print(f"Tree 1 and Tree 2 symmetric: {isSymmetricBinaryTreeCheck(tree1, tree2)}")
    # print(f"Tree 1 and Tree 3 same: {isSameBinaryTreeCheck(tree1, tree3)}")

    # print("validate de-serialization")
    # print([isSameBinaryTree(h1, h2) for h1,h2 in zip(serialized, serializedCheck)])

    # # heapify 方法
    # print("-"*30)
    # ooo_array1 = [100, 75, 9, 60, 40, 45, 30, 10, 8, 7, 15, 6, 3, 2, 1]
    # ooo_array2 =  [100, 75, 50, 60, 101, 45, 30, 10, 8, 7, 15, 6, 3, 2, 1]
    # print("可以通过heapify调整的大根堆(第三个元素)：")
    # print(f"调整前: {ooo_array1}")
    # heapify(ooo_array1, 2, len(ooo_array1))
    # print(f"调整后: {ooo_array1}")

    # ooo_array1 = [100, 75, 9, 60, 40, 45, 30, 10, 8, 7, 15, 6, 3, 2, 1]
    # heapifyCheck(ooo_array1, 2, len(ooo_array1))
    # print(f"调整后: {ooo_array1}validate")

    # print("无法通过heapify调整的大根堆(第五个元素)：")
    # print(f"调整前: {ooo_array2}")
    # heapify(ooo_array2, 4, len(ooo_array2))
    # print(f"调整后: {ooo_array2}")
    # ooo_array2 =  [100, 75, 50, 60, 101, 45, 30, 10, 8, 7, 15, 6, 3, 2, 1]
    # heapifyCheck(ooo_array2, 4, len(ooo_array2))
    # print(f"调整后: {ooo_array2}validate")

    # # heap insert 方法
    # ooo_array1 = [100, 75, 9, 60, 40, 45, 30, 10, 8, 7, 15, 6, 3, 2, 1]
    # ooo_array2 =  [100, 75, 50, 60, 101, 45, 30, 10, 8, 7, 15, 6, 3, 2, 1]
    # print("可以通过heapify insert调整的大根堆(第五个元素)：")
    # print(f"调整前: {ooo_array2}")
    # heapInsert(ooo_array2, 4)
    # print(f"调整后: {ooo_array2}")

    # ooo_array2 =  [100, 75, 50, 60, 101, 45, 30, 10, 8, 7, 15, 6, 3, 2, 1]
    # heapInsertCheck(ooo_array2, 4)
    # print(f"调整后: {ooo_array2}validate")

    # print("无法通过heap insert调整的大根堆(第三个元素)：")
    # print(f"调整前: {ooo_array1}")
    # heapInsert(ooo_array1, 2)
    # print(f"调整后: {ooo_array1}")
    # ooo_array1 = [100, 75, 9, 60, 40, 45, 30, 10, 8, 7, 15, 6, 3, 2, 1]
    # heapInsertCheck(ooo_array1, 2)
    # print(f"调整后: {ooo_array1}validate")

    # Tree 遍历
    root = BinaryTree(1)
    root.left = BinaryTree(2)
    root.right = BinaryTree(3)
    root.left.parent = root
    root.right.parent = root

    root.left.left = BinaryTree(4)
    root.left.right = BinaryTree(5)
    root.left.left.parent = root.left
    root.left.right.parent = root.left

    root.right.left = BinaryTree(6)
    root.right.right = BinaryTree(7)
    root.right.left.parent = root.right
    root.right.right.parent = root.right

    root.right.right.right = BinaryTree(8)
    root.right.right.right.parent = root.right.right
    print("-" * 30)
    print(lowestCommonAncestor(root, root.left.left, root.right.left).value)
    print("-" * 30)
    print(lowestCommonAncestorCheck(root, root.left.left, root.right.left).value)
    print("-" * 30)
    treeLevelOrderIterative(root)
    print("-" * 30)
    treeLevelOrderIterativeCheck(root)
    print("-" * 30)
    treePreOrderIterative(root)
    print("-" * 30)
    treePreOrderIterativeCheck(root)
    print("-" * 30)
    treeInOrderIterative(root)
    print("-" * 30)
    treeInOrderIterativeCheck(root)
    print("-" * 30)
    treePostOrderIterative(root)
    print("-" * 30)
    treePostOrderIterativeCheck(root)
    print("-" * 30)
    # 链表反转, Definition for singly-linked list.
    head = ListNode(0)
    head.next = ListNode(1)
    head.next.next = ListNode(2)
    head.next.next.next = ListNode(3)
    printLinkedList(head)
    print("-" * 30)
    head = reverseList(head)
    printLinkedList(head)
    print("-" * 30)
    head = reverseListCheck(head)
    printLinkedList(head)
    print("-" * 30)
    matrix = [
        [1, 2, 20], [2, 1, 20],
        [1, 3, 15], [3, 1, 15],
        [1, 4, 3], [4, 1, 3],
        [2, 5, 25], [5, 2, 25],
        [2, 6, 2], [6, 2, 2],
        [3, 7, 5], [7, 3, 5],
        [4, 8, 4], [8, 4, 4],
        [4, 9, 10], [9, 4, 10],
        [5, 10, 1], [10, 5, 1],
        [7, 11, 2], [11, 7, 2],
        [9, 12, 2], [12, 9, 2],
    ]

    graph = createGraphEdgeList(matrix)

    dfs(graph.nodes[1])
    print("-" * 30)
    dfsCheck(graph.nodes[1])
    print("-" * 30)
    bfs(graph.nodes[1])
    print('-' * 30)
    bfsCheck(graph.nodes[1])
    print("-" * 30)

    ## 快排
    arr = [5, 4, 3, 5, 6, 7, 6, 5, 6, 1, 2, 6, 8, 7, 10, 9]
    print(arr)
    quickSort(arr)
    print(arr)
    arr = [5, 4, 3, 5, 6, 7, 6, 5, 6, 1, 2, 6, 8, 7, 10, 9]
    print(arr)
    quickSortCheck(arr)
    print(arr)
    print("-" * 30)

    ## 融合排序
    arr = [5, 4, 3, 5, 6, 7, 6, 5, 6, 1, 2, 6, 8, 7, 10, 9]
    print(arr)
    mergeSort(arr)
    print(arr)
    arr = [5, 4, 3, 5, 6, 7, 6, 5, 6, 1, 2, 6, 8, 7, 10, 9]
    print(arr)
    mergeSortCheck(arr)
    print(arr)

    ## kruskal
    print("-"*30 + "kruskal" + "-"*30)
    print([eg.weight for eg in sorted(list(kruskalMST(graph)))])
    print("-"*30)
    print([eg.weight for eg in sorted(list(kruskalMSTCheck(graph)))])

    ## prim
    print("-"*30 + "prim" + "-"*30)
    print([eg.weight for eg in sorted(list(primMST(graph)))])
    print("-"*30)
    print([eg.weight for eg in sorted(list(primMSTCheck(graph)))])

    ## dijkstra'q
    print("-"*30 + "dijkstra'q" + "-"*30)
    print([(k.value, v) for k, v in dijkstra(graph.nodes[1]).items()])
    print("-"*30)
    print([(k.value, v) for k, v in dijkstraCheck(graph.nodes[1]).items()])

    # 递归
    print("子集")
    string_ = "dream"
    all_resa, all_resb = [], []
    allSubString(string_, 0, "", all_resa)
    print(all_resa)
    print("-"* 30)
    allSubStringCheck(string_, 0, "", all_resb)
    print(all_resb)

    print("全排列")
    all_resc, all_resd = [], []
    allPermutation(string_, all_resc)
    print(all_resc)
    print("-"* 30)
    allPermutationCheck(string_, all_resd)
    print(all_resd)

    print("4层汉诺塔")
    hanoi(4)
    print("-"* 30)
    hanoiCheck(4)

    print("N皇后")
    NQueen(6)
    print("-"* 30)
    NQueenCheck(6)


    # 二叉树的性质
    string_complete = "_".join([str(i) for i in [1,2,3,4,5,6,"#","#","#","#","#","#","#"]]) + "_"
    tree_complete = treeLevelOrderDeSerialization(string_complete)
    printBinaryTree(tree_complete)
    print(f"完美二叉树：{isPerfectBinaryTreeCheck(tree_complete)}")
    print(f"完满二叉树：{isFullBinaryTreeCheck(tree_complete)}")
    print(f"完全二叉树：{isCompleteBinaryTreeCheck(tree_complete)}")
    
    string_perfect = "_".join([str(i) for i in [1,2,3,4,5,6,7,"#","#","#","#","#","#","#","#"]]) + "_"
    tree_perfect = treeLevelOrderDeSerialization(string_perfect)
    printBinaryTree(tree_perfect)
    print(f"完美二叉树：{isPerfectBinaryTreeCheck(tree_perfect)}")
    print(f"完满二叉树：{isFullBinaryTreeCheck(tree_perfect)}")
    print(f"完全二叉树：{isFullBinaryTreeCheck(tree_perfect)}")
    
    string_full = "_".join([str(i) for i in [1,2,3,"#","#",6,7,"#","#","#","#"]]) + "_"
    tree_full = treeLevelOrderDeSerialization(string_full)
    printBinaryTree(tree_full)
    print(f"完美二叉树：{isPerfectBinaryTreeCheck(tree_full)}")
    print(f"完满二叉树：{isFullBinaryTreeCheck(tree_full)}")
    print(f"完全二叉树：{isCompleteBinaryTreeCheck(tree_full)}")
    print(f"平衡二叉树：{isBalancedBinaryTreeCheck(tree_full)}")
    
    string_imbalanced = "_".join([str(i) for i in [1,"#",3,6,7,"#","#","#","#"]]) + "_"
    tree_imba = treeLevelOrderDeSerialization(string_imbalanced)
    printBinaryTree(tree_imba)
    print(f"平衡二叉树：{isBalancedBinaryTreeCheck(tree_imba)}")


# 二叉树的遍历
def treeFullOrderRecursive(root: BinaryTree):
    '''
    递归实现树的全序遍历（前中后均打印）
    root = BinaryTree(1)
    root.left = BinaryTree(2)
    root.right = BinaryTree(3)
    root.left.left = BinaryTree(4)
    root.left.right = BinaryTree(5)
    root.right.left = BinaryTree(6)
    root.right.right = BinaryTree(7)
    root.right.right.right = BinaryTree(8)

    treeLevelOrderIterative(root)
    print("-" * 30)
    treeLevelOrderIterativeCheck(root)
    print("-" * 30)
    treePreOrderIterative(root)
    print("-" * 30)
    treePreOrderIterativeCheck(root)
    print("-" * 30)
    treeInOrderIterative(root)
    print("-" * 30)
    treeInOrderIterativeCheck(root)
    print("-" * 30)
    treePostOrderIterative(root)
    print("-" * 30)
    treePostOrderIterativeCheck(root)
    :param root:
    :return:
    '''
    if not root:
        return

    print(f"{root.value}_前")
    treeFullOrderRecursive(root.left)
    print(f"{root.value}_中")
    treeFullOrderRecursive(root.right)
    print(f"{root.value}_后")


def treePreOrderRecursive(root: BinaryTree):
    if not root:
        return
    print(f"{root.value}_前")
    treePreOrderRecursive(root.left)
    treePreOrderRecursive(root.right)


def treePreOrderIterative(root: BinaryTree):
    ''':
    循环写树的前序遍历，使用stack，遵循 pop 头，处理node，add 右，add左的顺序
    ivar'''
    from collections import deque
    if not root:
        return

    stack = deque([root])
    while len(stack) > 0:
        cur_node = stack.pop()
        print(f"{cur_node.value}_前")

        if cur_node.right:
            stack.append(cur_node.right)

        if cur_node.left:
            stack.append(cur_node.left)
    return


def treePreOrderIterativeRL(root: BinaryTree):
    ''':
    前序遍历改：头右左顺序打印
    ivar'''
    from collections import deque
    if not root:
        return

    stack = deque([root])
    while len(stack) > 0:
        cur_node = stack.pop()
        print(cur_node.value)

        if cur_node.left:
            stack.append(cur_node.left)

        if cur_node.right:
            stack.append(cur_node.right)
    return


def treeInOrderRecursive(root: BinaryTree):
    if not root:
        return

    treeInOrderRecursive(root.left)
    print(f"{root.value}_中")
    treeInOrderRecursive(root.right)


def treeInOrderIterative(root: BinaryTree):
    '''
    循环实现中序遍历，上来申请一个栈，先把包括根节点的树的最左枝所有点按访问顺序压进栈，之后开始pop，处理点，若pop出的点有右节点，右节点压进栈，
    并且从该右节点开始的最左枝每个节点按访问顺序
    :param root:
    :return:
    '''
    from collections import deque
    if not root:
        return

    stack = deque([root])
    while root.left:
        stack.append(root.left)
        root = root.left

    while len(stack) > 0:
        cur_node = stack.pop()
        print(f"{cur_node.value}_中")
        if cur_node.right:
            cur_R_node = cur_node.right
            stack.append(cur_R_node)
            while cur_R_node.left:
                stack.append(cur_R_node.left)
                cur_R_node = cur_R_node.left
    return


def treePostOrderRecursive(root: BinaryTree):
    if not root:
        return

    treePostOrderRecursive(root.left)
    treePostOrderRecursive(root.right)
    print(f"{root.value}_后")


def treePostOrderIterative(root: BinaryTree):
    '''
    循环实现后序遍历，使用一个弹出栈和一个收集栈，弹出栈按pop头，add左，add右的顺序压栈，每次弹出栈弹出node，放入收集栈，所有点收集完毕后，从收
    集栈中弹出并处理node
    :param root:
    :return:
    '''
    from collections import deque
    if not root:
        return

    pop_stack = deque([root])
    coll_stack = deque([])
    while len(pop_stack) > 0:
        cur_node = pop_stack.pop()
        coll_stack.append(cur_node)
        if cur_node.left:
            pop_stack.append(cur_node.left)

        if cur_node.right:
            pop_stack.append(cur_node.right)

    while len(coll_stack) > 0:
        out_node = coll_stack.pop()
        print(f"{out_node.value}_后")

    return


def treeLevelOrderIterative(root: BinaryTree):
    '''
    树的层序遍历 = 树的广度优先搜索bfs
    :param root:
    :return:
    '''
    from collections import deque
    # do not use list[], o(n) time complexity of pop
    queue = deque([root])

    while len(queue) > 0:
        cur_node = queue.popleft()
        print(f"{cur_node.value}_层")
        if cur_node.left:
            queue.append(cur_node.left)
        if cur_node.right:
            queue.append(cur_node.right)

    return



# 树的序列化和反序列化
def treePreOrderSerialization(root: BinaryTree) -> str:
    if not root:
        return "#_"

    res = f"{root.value}_"
    res += treePreOrderSerialization(root.left)
    res += treePreOrderSerialization(root.right)
    return res

def treeInOrderSerialization(root: BinaryTree) -> str:
    '''
    中序遍历序列化意义不大，因为没有对应的反序列化方法，但这个序列化的序列删掉空值之后可以看出值是递增的
    :param root:
    :return:
    '''
    if not root:
        return "#_"
    res = ""
    res += treeInOrderSerialization(root.left)
    res += f"{root.value}_"
    res += treeInOrderSerialization(root.right)
    return res

def treePostOrderSerialization(root: BinaryTree) -> str:
    if not root:
        return "#_"

    res = ""
    res += treePostOrderSerialization(root.left)
    res += treePostOrderSerialization(root.right)
    res += f"{root.value}_"

    return res

def treeLevelOrderSerialization(root: BinaryTree) -> str:
    if not root:
        return "#_"
    res = ""
    from collections import deque
    q = deque([root])
    while len(q) > 0:
        cur = q.popleft()
        res += "#_" if not cur else f"{cur.value}_"
        if cur:
            q.append(cur.left)
            q.append(cur.right)
    return res


def treePreOrderDeSerialization(string_:str) -> BinaryTree:
    '''
    前序遍历反序列化，用一个队列存序列值并弹出，构建节点时先构建根节点然后递归构建左节点和右节点
    :param string_:
    :return:
    '''
    from collections import deque

    def preOrderDeSerial(queue:deque):
        if not queue or len(queue) == 0:
            return None

        val = queue.popleft()
        if val == "#":
            return None
        # 使用队列作为辅助结构，最终需要头->左->右，则queue出来的顺序就是头->左->右

        head = BinaryTree(int(val))
        head.left = preOrderDeSerial(queue)
        head.right = preOrderDeSerial(queue)
        return head

    values = string_.split("_")

    q = deque([])
    for val in values:
        q.append(val)

    return preOrderDeSerial(q)

def treePostOrderDeSerialization(string_: str) -> BinaryTree:
    '''
    后序遍历序列的反序列化，使用栈存节点值，然后按照构建头节点，递归构建右节点，然后递归构建左节点的方法构建树
    :param string_:
    :return:
    '''
    from collections import deque

    def postOrderDeSerial(stack: deque):
        if not stack or len(stack) == 0:
            return None
        val = stack.pop()
        if val == "#":
            return None
        # 需要最终顺序左->右->头，stack中pop顺序头->右->左
        head = BinaryTree(int(val))
        head.right = postOrderDeSerial(stack)
        head.left = postOrderDeSerial(stack)
        return head

    values = string_.split("_")

    q = deque([])
    for val in values:
        q.append(val)

    return postOrderDeSerial(q)

def treeLevelOrderDeSerialization(string_: str) -> BinaryTree:
    '''
    层序遍历反序列化，用一个队列存序列值并弹出，另一个序列存构建产生的节点，用遍历的方式创造左右节点
    :param string_:
    :return:
    '''
    from collections import deque

    def levelOrderDeSerial(queue: deque):
        if not queue or len(queue) == 0:
            return None
        val = queue.popleft()

        head = None
        if val is not None and val != "#":
            head = BinaryTree(int(val))

        # 辅助队列:用于存放产生的节点
        q2 = deque([head])
        while len(q2) > 0:
            node = q2.popleft()

            # 左child
            leftChild = queue.popleft()
            if leftChild is not None and leftChild != "#":
                node.left = BinaryTree(int(leftChild))
                q2.append(node.left)
            else:
                node.left = None

            # 右child
            rightChild = queue.popleft()
            if rightChild is not None and rightChild != "#":
                node.right = BinaryTree(int(rightChild))
                q2.append(node.right)
            else:
                node.right = None

        return head

    values = string_.split("_")
    q = deque([])
    for val in values:
        q.append(val)
    # 这个队列作为参数传进主逻辑函数，存放parse出来的节点的值
    return levelOrderDeSerial(q)

# 暴力递归和回溯
@timer
def NQueen(n: int) -> int:
    ''':
    N皇后问题
    ivar'''
    def isValid(rrecord, ii, jj):
        for i in range(ii):
            # 需要看从第0行开始到目前ii-1行是否有冲突，ii行及以后还没填所以不看
            # 第i行的jj列已经摆了皇后， 那么来到ii行jj列就不能再摆了
            if rrecord[i] == jj:
                return False
            # 对角线有重复皇后
            if abs(ii - i) == abs(jj - rrecord[i]):
                return False
        return True

    def nQueenProcess(i, record, N):
        '''
        :param i: 当前在考虑第i行往哪里摆皇后
        :param record: records数组记录每行在哪一列放皇后
        :param N: 一共要在几乘几的棋盘放几个皇后
        :return: 总共有几种摆法
        '''
        if i == N:
            # 找到一个可行的排列法
            print(record)
            return 1

        res = 0
        for j in range(N):
            # 遍历当前行的所有列看是否有皇后冲突
            if isValid(record, i, j):
                record[i] = j
                res += nQueenProcess(i + 1, record, N)
        return res

    records = [None] * n
    if n < 1:
        return 0

    return nQueenProcess(0, records, n)


def projectArrangement(numProj: int, money: int, projects: List[List[int]]):
    '''
    numProj:一共最多做几个project
    money:手里的初始资金
    projects:project list,每个项目给一个cost给一个profit
    求给定初始资金，给定project list，给定最多做几个project，最大的利润是多少
    先把project按照cost从小到大排序进入minheap，然后对给定的初始资金，依次找到所有cost小于资金的项目弹出，这些项目按利润从小到大排序进入
    maxheap，每次做利润最高的项目并update初始资金，直至做完numProj个项目或者无项目可做为止返回
    :param numProj:
    :param money:
    :param projects:
    :return:
    '''
    # 初始化heap,目前heap里面还没有任何data
    minCost, maxProfit = heapq.heapify([MinCostProject]), heapq.heapify([MaxProfitProject])

    for proj in projects:
        # 初始化每一个project并压进mincost heap
        heapq.heappush(minCost, MinCostProject(proj[0], proj[1]))

    for i in range(numProj):
        while len(minCost) > 0 and minCost[0].cost_ <= money:
            popped_proj = heapq.heappop(minCost)
            heapq.heappush(maxProfit, MaxProfitProject(popped_proj.cost_, popped_proj.profit_))
        if len(maxProfit) == 0:
            return money

        money += heapq.heappop(maxProfit).profit_
    return money

    pass


def minimizeCost(golds: List[int]) -> int:
    '''
    分金子问题，总数为n的金子分成golds里的状态，n 分成 a + b = n的操作会产生n的cost，函数返回最小的cost
    策略是反过来想，因为最终是一堆金子分成几堆，那可以用分好的堆加回去，
    用golds里的数字两两组合看如何产生最小的cost，先把他变成一个min heap，然后每次heap里弹出
    最小和第二小的元素，加和并把和push回minheap里去，直至heap为空则结束
    :ivar
    '''
    heapq.heapify(golds)
    sum_ = 0
    res = []
    while len(golds) > 1:
        mi_ = heapq.heappop(golds)
        mi2_ = heapq.heappop(golds)
        cur = mi_ + mi2_
        res.append([mi_, mi2_])
        sum_ += cur
        heapq.heappush(golds, cur)
    return sum_, res


def meetingArrangement(meetings: List[Meeting], timeStamp) -> int:
    '''
    给会议起止时间，遍历一次安排最多个会议，这个贪心算法的本质是要想到用会议结束的时间从小到大先排个序
    nlog(n)复杂度，之后每过一个会议就update一个外部的timestamp记录前一个结束的会议的结束时间，然后依次判断

    :param meetings:
    :param timeStamp:
    :return:
    '''
    # 为了排序meetings，需要在类中定义魔术函数__lt__，根据这个算法我们定义结束时间小的应该排在前面（更小）
    meetings_sorted = sorted(meetings)
    res = 0
    for mt in meetings_sorted:
        if timeStamp <= mt.start_:
            res += 1
            timeStamp = mt.end_

    return res


# 最小生成树图算法
def kruskalMST(graph: Graph) -> Set[Edge]:
    ''':ivar
    给一个无向图，返回最小生成树的kruskal算法
    '''
    uf = UnionFind(list(graph.nodes.values()))
    mheap = list(graph.edges)
    heapq.heapify(mheap)

    res: Set[Edge] = set()
    while len(mheap) > 0:
        ed = heapq.heappop(mheap)
        if not uf.isSameSet(ed.from_, ed.to_):
            res.add(ed)
            uf.union(ed.from_, ed.to_)
    return res


def primMST(graph: Graph) -> Set[Edge]:
    '''
    最小生成树的prim算法，每次选择一条权重最小且能到达一个新点的边
    :param graph:
    :return:
    '''

    mHeap: List[Edge] = []
    heapq.heapify(mHeap)
    st = set()
    res = set()
    for nd in graph.nodes.values():
        # 遍历所有点防止森林问题，如果图联通的话就随便找一个点作为起点
        if nd not in st:
            # 只有新的node才考虑去找边
            st.add(nd)
            for ed in nd.edges:
                heapq.heappush(mHeap, ed)
            while len(mHeap) > 0:
                # pop出权重最轻的edge
                pickedEd: Edge = heapq.heappop(mHeap)
                toNd: Node = pickedEd.to_
                if toNd not in st:
                    # 通过这条边连接到新的点了，把这条边加进结果
                    res.add(pickedEd)
                    st.add(toNd)
                    for nextEdge in toNd.edges:
                        # 新的点发散出去的边也加到heap里
                        heapq.heappush(mHeap, nextEdge)
    return res


# 图最短路径算法
def bellmanFord(graph: Graph):
    head = graph.nodes.values[0]
    n = len(graph.nodes.values()) - 1
    distanceMap = dict()
    for node in graph.nodes.values():
        if head == node:
            distanceMap[head] = 0
        else:
            distanceMap[node] = float("inf")

    for i in range(n):
        for eg in graph.edges:
            from_ = eg.from_
            to_ = eg.to_
            weight = eg.weight
            distanceMap[to_] = min(distanceMap[to_], distanceMap[from_] + weight)

    return distanceMap


def spfa(graph: Graph, head: Node):
    '''
    Shortest path faster algorithm 产生从某个点出发到图中剩余各个点的最短距离，
    使用一个Queue来盛放通过某些边到达的点，这些边可以减小某个最短距离，最后返回distancemap
    此算法以点为处理对象
    :param graph:
    :return:
    '''
    dmap = dict()
    for node in graph.nodes.values:
        if node == head:
            dict[node] = 0
        else:
            dict[node] = float("inf")
    from collections import deque
    q = deque([head])
    q_ele = set() # 为了实现o(1) element in set 检查，牺牲额外的内存把node的地址存在集合里面
    q_ele.append(head)
    while len(q) > 0:
        cur = q.popleft()
        for eg in cur.edges:
            to_ = eg.to_
            if dmap[cur] + eg.weight < dmap[to_]:
                dmap[to_] = dmap[cur] + eg.weight
                if to_ not in q_ele:
                    q.append(to_)
                    q_ele.add(to_)
    return dmap


def dijkstra(head: Node) -> Dict[Node, int]:
    ''':
    Dijkstra算法，适用于边权重不为负的图，规定起点，返回从起点出发到各个点的最短距离
    从起点出发，创一个起点到各个点距离的map，创一个不再改距离的点集合
    从map中返回要处理的从起点出发距离最小的点，找从这个点发散出去的边从而找到toNode，对每个toNode更新distancemap中的距离
    ivar'''
    def getMinDistanceAndUnselectedNode(distanceMap, selectedNodes):
        '''
        选择distanceMap里距离最小且不在selectedNodes里的node返回，
        否则返回None,代表全部点已经在selectedNodes里或distanceMap为空
        :param distanceMap:
        :param selectedNodes:
        :return:
        '''
        mindistance = float("inf")
        nd = None
        for node, distance in distanceMap.items():
            if node not in selectedNodes and distance < mindistance:
                nd = node
                mindistance = distance
        return nd

    distanceMap: Dict[Node, int] = dict()  # 记录每个node到起点的距离
    distanceMap[head] = 0
    selectedNodes: Set[Node] = set()  # 哪些点的距离不再更改了就放这里面
    minNode = getMinDistanceAndUnselectedNode(distanceMap, selectedNodes)
    while minNode:
        for edge in minNode.edges:
            toNode: Node = edge.to_
            # 从起点出发到目前minNode的最小距离为distance
            distance: int = distanceMap[minNode]
            if toNode not in distanceMap:
                # toNode还没在map里，说明从起点到toNode当前距离是正无穷，用从起点到minNode的距离加上从minNode到toNode的距离来更新map中
                # 从起点到toNode的距离
                distanceMap[toNode] = distance + edge.weight
            else:
                # 如果从当前minNode通过edge的距离让到toNode的距离减小了，就更新map
                distanceMap[toNode] = min(distanceMap[toNode], distance + edge.weight)
        # 从当前minNode出发所有的边处理完之后，把这个点加到选择过的node中，从起点到这个点的距离就定下来了
        selectedNodes.add(minNode)
        # 更新过的map中再找一个minNode, 重复
        minNode = getMinDistanceAndUnselectedNode(distanceMap, selectedNodes)
    return distanceMap


def dijkstra_modified(head: Node, size: int) -> Dict[Node, int]:
    ''':
    改进的Dijkstra算法，适用于边权重不为负的图，规定起点，返回从起点出发到各个点的最短距离
    从起点出发，创一个起点到各个点距离的map，创一个不再改距离的点集合
    从map中返回要处理的从起点出发距离最小的点，找从这个点发散出去的边从而找到toNode，对每个toNode更新distancemap中的距离
    用一个改进的heap结构来处理distance map,这样每次不需要遍历distancemap而是可以直接弹出最小的node
    ivar'''
    nodeheap = NodeHeap(size)
    nodeheap.addOrUpdateOrIgnore(head, 0)
    distanceMap: Dict[Node, int] = dict()
    while not nodeheap.isEmpty():
        nodeRecord = nodeheap.pop()
        minNode = nodeRecord.node_
        distance = nodeRecord.distance_
        for edge in minNode.edges:
            nodeheap.addOrUpdateOrIgnore(edge.to_, edge.weight + distance)
        distanceMap[minNode] = distance
        return distanceMap


def graphNegativeCycle(graph: Graph):
    '''
    使用bellman ford算法检查负权重的环, 暴力跑v-1次，每次遍历所有边然后减小选定点到其他点的距离
    额外再跑一次边的遍历，如果图里没有负权环，则距离不会被更新到更小
    :param graph:
    :return:
    '''
    n = len(graph.nodes)
    dMap = dict()
    head = next(iter(graph.nodes.values()))
    for node in graph.nodes.values():
        dMap[node] = 0 if head == node else float("inf")
    for i in range(n-1):
        for eg in graph.edges:
            from_ = eg.from_
            to_ = eg.to_
            weight = eg.weight
            dMap[to_] = min(dMap[to_], dMap[from_] + weight)
    for eg in graph.edges:
        from_ = eg.from_
        to_ = eg.to_
        weight = eg.weight
        if dMap[from_] + weight < dMap[to_]:
            return True
    return False


# 图的生成
def createGraphEdgeList(matrix: List[List]) -> Graph:
    ''':
    输入一个表示图的矩阵，每个行的列表元素分别为weight,from节点值和to节点值
    这个函数可以拓展到其他的input格式
    返回一个自定义Graph数据结构，方便写算法，
    ivar
    matrix = [
        [1, 2, -1], [2, 1, -1],
        [1, 3, -1], [3, 1, -1],
        [1, 4, -1], [4, 1, -1],
        [2, 5, -1], [5, 2, -1],
        [2, 6, -1], [6, 2, -1],
        [3, 7, -1], [7, 3, -1],
        [4, 8, -1], [8, 4, -1],
        [4, 9, -1], [9, 4, -1],
        [5, 10, -1], [10, 5, -1],
        [7, 11, -1], [11, 7, -1],
        [9, 12, -1], [12, 9, -1],
    ]

    graph = createGraph(matrix)
    print("深度优先搜索")
    dfs(graph.nodes[1])
    print("广度优先搜索")
    bfs(graph.nodes[1])
    '''

    graph = Graph()
    for row in matrix:
        fromVal, toVal, weight = row
        if fromVal not in graph.nodes:
            graph.nodes[fromVal] = Node(fromVal)
        if toVal not in graph.nodes:
            graph.nodes[toVal] = Node(toVal)

        fromNode = graph.nodes[fromVal]
        toNode = graph.nodes[toVal]
        newEdge = Edge(weight, fromNode, toNode)
        fromNode.nexts.append(toNode)
        fromNode.out_ += 1
        toNode.in_ += 1
        fromNode.edges.append(newEdge)
        graph.edges.add(newEdge)
    return graph


def createGraphAdjList(adjtable: Dict[int, List]) -> Graph:
    '''
    adjTable = dict()
    adjTable[1] = [2, 3, 5]
    adjTable[2] = [1, 6, 8]
    adjTable[3] = [1, 4, 9]
    adjTable[4] = [3, 5, 8]
    adjTable[5] = [1, 4, 7]
    adjTable[6] = [2]
    adjTable[7] = [5]
    adjTable[8] = [2, 4, 9, 10]
    adjTable[9] = [3, 8]
    adjTable[10] = [8]
    graph = createGraphAdjList(adjTable)
    从邻接表input生成一个图
    :param adjtable: key为node的值，value为从这个点出发可以一步到达哪几个点
    :return: graph object
    '''
    graph = Graph()
    for from_val, to_values in adjtable.items():
        if from_val not in graph.nodes:
            graph.nodes[from_val] = Node(from_val)
        from_node = graph.nodes[from_val]

        for to_val in to_values:
            if to_val not in graph.nodes:
                graph.nodes[to_val] = Node(to_val)
            to_node = graph.nodes[to_val]
            from_node.out_ += 1
            to_node.in_ += 1
            from_node.nexts.append(to_node)

            edge = Edge(weight=0, from_=from_node, to_=to_node)
            if edge not in graph.edges:
                graph.edges.add(edge)
                from_node.edges.append(edge)
    return graph


def createGraphAdjMatrix(nodeList: List[int], matrix: List[List[int]]) -> Graph:
    '''
    图的邻接矩阵表示和初始化
    nodeList = [1,2,3,4,5,6,7,8,9,10]
    matrix = [
        [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 1, 1],
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    ]
    graph = createGraphAdjMatrix(nodeList, matrix)

    :param nodeList: 每个元素对应一个node的value
    :param matrix: matrix[i][j]代表从nodeList[i]出发到nodeList[j]是否存在一条边 (以及如果有的话这条边的weight是多少)
    :return:
    '''

    graph = Graph()

    for i in range(len(nodeList)):
        if nodeList[i] not in graph.nodes:
            graph.nodes[nodeList[i]] = Node(nodeList[i])
        from_node = graph.nodes[nodeList[i]]
        for j in range(len(nodeList)):
            if nodeList[j] not in graph.nodes:
                graph.nodes[nodeList[j]] = Node(nodeList[j])
            to_node = graph.nodes[nodeList[j]]
            if matrix[i][j] != 0:
                from_node.out_ += 1
                to_node.in_ += 1
                edge = Edge(weight=matrix[i][j], from_=from_node, to_=to_node)
                if edge not in graph.edges:
                    graph.edges.add(edge)
                    from_node.edges.append(edge)
                    from_node.nexts.append(to_node)
    return graph

# 图的遍历
def bfs(node: Node):
    ''':ivar
    node为图中的一个节点，此算法为图的广度优先遍历，使用队列，从任意一个点开始，弹出点，处理点，把此点未出现过的邻居点放在队列中，直至整个队列
    变为空，函数里加了一个set来保证每个点只会加进队列一次
    '''
    from collections import deque
    if not node:
        return
    st = set()
    # 根节点进队列
    queue = deque([node])
    st.add(node)
    while len(queue) > 0:
        # 先进先出一个节点
        nd = queue.popleft()
        # 此print函数可以替换成任何处理当前节点的函数
        print(f"{nd.value}_广")
        # 把当前节点的邻居节点依次加进队列
        for neighbour in nd.nexts:
            if neighbour not in st:
                queue.append(neighbour)
                st.add(neighbour)
    return

def dfs(node: Node):
    ''':ivar
    深度优先遍历：使用stack，先放进去开始的点，dfs当加入新节点的时候处理节点，之后每次弹出栈最上面的点，找到第一个还没进过栈的neighbour，处理
    并加入栈，neighbour循环停止；这里实际上有两个过程，第一个过程是把所有点加到栈中，后面等所有点都加进去之后再依次弹出来
    '''
    from collections import deque
    if not node:
        return

    st = set()
    # 根节点进栈
    stack = deque([node])
    # 已经进过栈的节点记录在集合里
    st.add(node)
    # 处理节点（可替换成任何操作）
    print(f"{node.value}_深")
    while len(stack) > 0:
        # 栈结构弹出最上面的元素
        cur_node = stack.pop()
        for neighbour in cur_node.nexts:
            if neighbour not in st:
                # 先加回当前节点，再加入这个邻居节点，然后处理这个邻居节点
                stack.append(cur_node)
                stack.append(neighbour)
                print(f"{neighbour.value}_深")
                # 处理完成后把这个邻居节点加进set保证不重复处理
                st.add(neighbour)
                break
    return

def dfsRecursive(node: Node):
    def dfsHelper(node: Node, visited: Set):
        if not node:
            return
        else:
            print(f"{node.value}_深")
            visited.add(node)
        for nd in node.nexts:
            if nd not in visited:
                dfsHelper(nd, visited)
        return

    # main
    visited = set()
    dfsHelper(node, visited)
    return


def topologySort(graph: Graph) -> List[Node]:
    ''':ivar
    拓扑排序实现：建立一个 node:入度的map，遍历图的每个点，把每个点的入度放进去
    把入度为0的点（代表没有其他点的依赖性）放在zeroIn队列中，在zeroIn队列不为空的情况下，从弹出一个node，遍历所有从这个这个node出发经过某
    边的下一个点，在map中把这些点的入度减一，代表擦掉这些点对弹出点的依赖，如果擦除依赖的点的入度降为0，加入zeroIn队列，重复这个过程直到zeroIn
    队列为空，这个算法可以处理有依赖关系的问题比如先修课要求之类的
    '''
    res: List[Node] = []
    inMap: Dict[Node: int] = {}
    from collections import deque
    zeroInQueue: List[Node] = deque([])

    for nd in graph.nodes.values():
        inMap[nd] = nd.in_
        if nd.in_ == 0:
            zeroInQueue.append(nd)

    while len(zeroInQueue) > 0:
        cur_node = zeroInQueue.popleft()
        res.append(cur_node)
        for neighbour in cur_node.nexts:
            inMap[neighbour] -= 1
            if inMap[neighbour] == 0:
                zeroInQueue.append(neighbour)
    return res


def powerset(array):
    # Write your code here.
    if len(array) == 0:
        return [[]]
    elif len(array) == 1:
        return [[], array]
    else:
        ret = [[]]

        for num in array:
            temp_ret = ret.copy()
            for item in temp_ret:
                ret.append(item + [num])
        return ret


def getPermutations(array):
    def helper(arr, perm, perms):
        if len(arr) == 0:
            perms.append(perm)
        else:
            for ele in arr:
                newArr = arr.copy()
                newArr.remove(ele)
                newPerm = perm + [ele]
                helper(newArr, newPerm, perms)

    # Write your code here.
    perms = []
    perm = []
    if len(array) == 0:
        return perms
    else:
        helper(array, perm, perms)
    return perms


def getPermutations2(array):
    def permutationsHelper(idx, array, permutations):
        if idx == len(array) - 1:
            permutations.append(array.copy())
        else:
            for j in range(idx, len(array)):
                swap(array, idx, j)
                permutationsHelper(idx + 1, array, permutations)
                swap(array, idx, j)

    def swap(array, i, j):
        array[i], array[j] = array[j], array[i]

    # Write your code here.
    permutations = []
    if len(array) == 0:
        return permutations
    else:
        permutationsHelper(0, array, permutations)

    return permutations


def fourNumberSum(array, targetSum):
    # Write your code here.
    # break down to 2 sum problem by calculating pair sum first
    ret = []
    sum2 = {}
    for i in range(len(array)):
        # forward (i+1 to the right most): check if we find a match
        # if a match is found, create all associated quadruples, if not, continue
        for j in range(i + 1, len(array)):
            p = array[i] + array[j]
            diff = targetSum - p
            if diff not in sum2:
                continue
            else:
                for arr in sum2[diff]:
                    ret.append([array[i], array[j], arr[0], arr[1]])
        # backward(left most to i): check if the pair sum already in hashtable, if so, append the pair
        # otherwise we find a new pair sum, create empty list
        for k in range(0, i):
            q = array[k] + array[i]
            if q in sum2:
                sum2[q].append([array[k], array[i]])
            else:
                sum2[q] = [[array[k], array[i]]]

    return ret


def largestRange(array):
    # Write your code here.
    if len(array) == 1:
        return [array[0], array[0]]
    array.sort()
    list_of_ranges = []
    left = array[0]
    right = array[0]
    for i in range(1, len(array)):
        if array[i] - 1 == array[i - 1]:
            right = array[i]
        else:
            temp_range = (left, right)
            list_of_ranges.append(temp_range)
            print(temp_range)
            left = array[i]
            right = array[i]
        if i == len(array) - 1:
            list_of_ranges.append((left, right))

    largest_range = -1
    lleft, rright = -1, -1
    for left, right in list_of_ranges:
        if right - left > largest_range:
            largest_range = right - left
            lleft = left
            rright = right

    return [lleft, rright]


def hasSingleCycle(array):
    # Write your code here.
    n = len(array)
    for i in range(n):
        visited_idx = {}
        for num in range(len(array)):
            visited_idx[num] = 0
        current_idx = i
        counter = 0
        while counter < n:
            current_idx += array[i]
            current_idx = current_idx % n
            visited_idx[current_idx] += 1
            counter += 1
        if all([value for key, value in visited_idx.items()]) and (current_idx == i) and (
                sum([value for key, value in visited_idx.items()]) == n):
            return True
    return False


def neatherland1(arr, num):
    '''
    荷兰国旗问题1： 将数组中小于num的放左边，大于等于num的放右边
    :param arr: 原数组
    :param num: 分界值
    :return:
    '''
    # 两个指针，一个不断往右走，另一个代表小于num的边界
    boundary, floating = 0, 0
    while floating < len(arr):
        # 如果当前值恰好小于num，将边界指针指向的value和浮动指针指向的value交换，
        # 边界指针和浮动指针同时往右走
        if arr[floating] <= num:
            arr[boundary], arr[floating] = arr[floating], arr[boundary]
            boundary += 1
            floating += 1
        else:
            # 如果当前值大于等于num，边界指针不动，浮动指针向右走
            floating += 1
    return arr


def neatherland2(arr, num):
    '''
    For quick-sort, partition2, Neatherland flag problem with equal elements
    荷兰国旗问题2： 将数组中小于num的放左边，等于num的放中间，大于num的放右边，同时也是快速排序中partition的思想
    :param arr:
    :param num:
    :return:
    '''

    # 左右指针，分别代表小于num和大于num的边界，分别初始化在数组开始和末尾
    left, right = -1, len(arr)

    # 浮动指针，从左向右走直至和right指针相撞
    floating = 0
    while floating < right:
        if arr[floating] < num:
            # 当前值大于num，找左边界的下一个和当前值交换，左指针往右走一步，浮动指针也走一步
            arr[left + 1], arr[floating] = arr[floating], arr[left + 1]
            left += 1
            floating += 1
        elif arr[floating] > num:
            # 当前值大于num，右指针前一个和当前值交换，右指针左移一个(浮动指针不动，因为新换来的值不还确定大小)
            arr[right - 1], arr[floating] = arr[floating], arr[right - 1]
            right -= 1
        else:
            # 当前值等于num，浮动指针向右走一步
            floating += 1

    return arr


def reverseStack(stack: List[int]) -> List[int]:
    # new_stack = stack[::-1]
    # 使用系统函数栈来逆序一个栈
    def f(stack_: List[int]) -> int:
        # 此函数递归实现从一个栈中把最底下的元素返回，整个栈下移的操作
        result = stack_.pop(0)
        if len(stack_) == 0:
            return result
        else:
            last = f(stack_)
            stack_.insert(0, result)
            return last

    if len(stack) == 0:
        return
    i = f(stack)
    reverseStack(stack)
    stack.insert(0, i)


def heapInsert(arr, idx):
    ''':ivar
    以数组实现大根堆，假设已经在idx位置插入了一个新值，试图将这个值向heap的根节点移动以使整个数据结构满足heap的性质
    max heap, insert value at idx and try to MOVE VALUE UP if possible
    change the direction in while condition this can be changed to a min heap
    '''

    # use array to represent a heap, then given parant at idx, left child at 2 * idx + 1, right child at 2 * idx + 2
    # given child at idx, parent at int((idx - 1) / 2)

    while arr[idx] > arr[int((idx - 1) / 2)]:
        # 当前节点值大于其父节点值时，交换两个节点的值，并将节点指针移动到父节点位置
        arr[int((idx - 1) / 2)], arr[idx] = arr[idx], arr[int((idx - 1) / 2)]
        idx = int((idx - 1) / 2)
    return


def heapify(arr, idx, heapSize):
    '''
    max heap, Insert at idx position and heapify this value DOWN if possible
    change definition of largest_idx this can be changed to a min heap
    heapify: 假设之前的步骤修改了大根堆arr中idx位置的数字，这个程序试图将这个数向叶节点移动，使数据结构重新满足大根堆的性质
    注意这个函数不会修改idx之前的位置的结构

    :param arr:
    :param idx:
    :param heapSize:
    :return:
    '''

    # 开始把这个值向下尽量挪动到叶子节点
    left = 2 * idx + 1  # 父节点的左孩子节点
    while left < heapSize:
        # 看左右孩子哪个值大，哪个值和父节点比较，如果比父节点值大，交换
        largest_idx = left + 1 if (left + 1 < heapSize) and arr[left + 1] > arr[left] else left
        largest_idx = largest_idx if arr[largest_idx] > arr[idx] else idx

        if largest_idx == idx:
            break

        arr[idx], arr[largest_idx] = arr[largest_idx], arr[idx]
        idx = largest_idx
        left = idx * 2 + 1
    return


def heapSort(arr):
    if len(arr) < 2:
        return

    # o(N*logN) construct a heap, top-down using insert
    # for i in range(len(arr)):
    #     heapInsert(arr, i)

    # o(N) construct a heap bottom up using heapify
    for i in range(len(arr) - 1, -1, -1):
        heapify(arr, i, len(arr) - 1)
    print(arr)
    heapSize = len(arr) - 1
    while heapSize > 0:
        # 弹出的过程，交换heap中头尾两个节点
        arr[0], arr[heapSize] = arr[heapSize], arr[0]
        # 把新的头向下heapify到对的位置
        heapify(arr, 0, heapSize)
        # heap大小减一，如果删掉的元素是自定义结构还需要del掉
        heapSize -= 1


def quickSort(arr):
    '''
    快速排序，选择一个值当做pivot，将数组中小于这个pivot的值放左边，等于这个值的放中间，大于这个值的放右边，后递归小于部分和大于部分
    :param arr:
    :return:
    '''

    def partition(ARR, ll, rr):
        '''
        For quick-sort, partition2, Neatherland flag problem with equal elements
        :param arr:
        :param num:
        :return:
        '''

        if ll > rr:
            return -1, -1

        if ll == rr:
            return ll, rr
        # ARR[r]作为pivot值
        pivot = ARR[rr]
        # 小于区域，大于区域的边界index
        less_ptr, more_ptr = ll - 1, rr
        floating_ptr = ll
        # 小于区域从左往右，大于区域从右往左，遍历一遍array
        while floating_ptr < more_ptr:
            if ARR[floating_ptr] < pivot:
                # 小于pivot的值放到小于区域，小于区域右移一个，当前ptr右移一个
                ARR[less_ptr + 1], ARR[floating_ptr] = ARR[floating_ptr], ARR[less_ptr + 1]
                less_ptr += 1
                floating_ptr += 1
            elif ARR[floating_ptr] > pivot:
                # 大于pivot的值放到大于区域，小于区域右移一个，当前ptr不动
                ARR[more_ptr - 1], ARR[floating_ptr] = ARR[floating_ptr], ARR[more_ptr - 1]
                more_ptr -= 1
            else:
                # 等于区域的话就只动ptr
                floating_ptr += 1

        ARR[more_ptr], ARR[rr] = ARR[rr], ARR[more_ptr]

        return less_ptr + 1, more_ptr

    def quickSortHelper(arrr, l, r):
        if l >= r:
            return
        # 在l到r-1位置随意找一个位置和r交换当作pivot
        rand_int = random.randint(l, r - 1)
        arrr[rand_int], arrr[r] = arrr[r], arrr[rand_int]
        # 通过partition找到arr中大于arr[r]和小于arr[r]的位置
        # 递归调用quicksort排序同样的小规模问题直至l，r相遇
        lessUpper, greatLower = partition(arrr, l, r)
        quickSortHelper(arrr, l, lessUpper - 1)
        quickSortHelper(arrr, greatLower + 1, r)
        return

    if len(arr) < 2:
        return
    quickSortHelper(arr, 0, len(arr) - 1)
    return


def mergeSortNaive(arr):
    '''
    需要额外申请很多数组的merge sort，时间复杂度一致
    :param arr:
    :return:
    '''
    def merge(arr1, arr2):
        res = []
        pt1, pt2 = 0, 0
        while pt1 < len(arr1) and pt2 < len(arr2):
            if arr1[pt1] < arr2[pt2]:
                res.append(arr1[pt1])
                pt1 += 1
            else:
                res.append(arr2[pt2])
                pt2 += 1
        while pt1 < len(arr1):
            res.append(arr1[pt1])
            pt1 += 1
        while pt2 < len(arr2):
            res.append(arr2[pt2])
            pt2 += 1
        return res
    if len(arr) < 2:
        return arr
    mid = len(arr) // 2
    ar_l = mergeSortNaive(arr[:mid])
    ar_r = mergeSortNaive(arr[mid:])
    sorted_arr = merge(ar_l, ar_r)
    return sorted_arr


def mergeSort(arr):
    def process(ar, l, r):
        if l == r:
            return
        mid = l + ((r - l) >> 1)
        process(ar, l, mid)
        process(ar, mid + 1, r)
        merge(ar, l, mid, r)

    def merge(ar, l, m, r):
        res = [None] * (r - l + 1)
        i = 0
        p1 = l
        # 这里等同于传进来两个小数组，我们默认所有的值传进来时，l到m,m+1到r各自是有序的
        p2 = m + 1
        while p1 <= m and p2 <= r:
            res[i] = ar[p1] if ar[p1] < ar[p2] else ar[p2]
            i += 1
            if ar[p1] < ar[p2]:
                p1 += 1
            else:
                p2 += 1

        while p1 <= m:
            res[i] = ar[p1]
            i += 1
            p1 += 1
        while p2 <= r:
            res[i] = ar[p2]
            i += 1
            p2 += 1
        for i in range(len(res)):
            ar[l+i] = res[i]
        return

    if arr is None or len(arr) < 2:
        return
    process(arr, 0, len(arr) - 1)


def sortedArrayDistanceLessThanK(arr, K):
    '''
    Sort almost ordered array using min heap, assume each number is at most k positions away from correct position
    in the array
    假设array几乎有序，其中任何一个值距离有序时这个值在数组中的位置距离不超过k，则可以线性复杂度完成排序
    :param arr:
    :param K:
    :return:
    '''
    pass


def radixSort(arr):
    '''
    Limited usage, 这里我们假设arr里面只有正整数

    :param arr:
    :return:
    '''

    def maxbits(arr):
        max_val = max(arr)
        return int(math.log10(max_val)) + 1

    def getDigit(num, d):
        while d > 0:
            ret = num % 10
            num = num // 10
            d -= 1
        return ret

    def radixSortHelper(arr, l, r, digit):
        radix = 10
        i, j = 0, 0
        bucket = [0] * (r - l + 1)  # 初始化一个桶，长度和arr长度相同

        for d in range(1, digit + 1):
            # for each digit of max number
            # create a frequency count vector
            count = [0] * radix
            for i in range(l, r + 1):
                # put digit d into bucket, at this digit, what is the frequency for each number from 0 to 9
                j = getDigit(arr[i], d)
                count[j] += 1
            for i in range(1, radix):
                # accumulate frequency (pre-fix sum, cumsum)
                # here it is a trick, this make sure that if a number at digit d has a large value (like 9), this number
                # will guarantee that the count for this is the largest (and at last element of count)
                count[i] += count[i - 1]

            for i in range(r, l - 1, -1):
                # loop from right to left, guarantee 2 things:
                # 1. number with digit d smaller will appear earlier in bucket
                # 2. number with digit d the same will preserve the order in original array in bucket
                j = getDigit(arr[i], d)
                bucket[count[j] - 1] = arr[i]
                count[j] -= 1

            j = 0
            for i in range(l, r + 1):
                arr[i] = bucket[j]
                j += 1

    if len(arr) < 2:
        return

    radixSortHelper(arr, 0, len(arr) - 1, maxbits(arr))


def printLinkedList(head: ListNode):
    while head:
        print(head.val)
        head = head.next
    return


def reverseList(head: ListNode):
    """
    # 链表反转, Definition for singly-linked list.
    head = ListNode(0)
    head.next = ListNode(1)
    head.next.next = ListNode(2)
    head.next.next.next = ListNode(3)
    printLinkedList(head)
    print("-"*30)
    head = reverseList(head)
    printLinkedList(head)
    print("-"* 30)
    head = reverseListCheck(head)
    # printLinkedList(head)

    :type head: ListNode
    :rtype: ListNode
    """
    # 特殊情况：空链表
    if head is None:
        return
    # 定义前节点，现节点
    prev_node = None
    curr_node = head
    while curr_node is not None:
        # 遍历链表，先把下一个节点存在next_node里
        next_node = curr_node.next
        # 把现节点的下一个设为前节点（原来指向下一个节点的link打破了）
        curr_node.next = prev_node
        # 前节点变为现节点， 现节点变为下一个节点，循环
        prev_node = curr_node
        curr_node = next_node

    # while一顿操作之后最后的curr_node指向尾部最后一个元素的下一个，也就是null节点，我们需要返回前一个节点
    return prev_node


def checkPalindrome(arr):
    if len(arr) < 2:
        return True
    from collections import deque
    stack = deque([])
    for i in range(len(arr)):
        stack.append(arr[i])
    for j in range(len(arr)):
        ele = stack.pop()
        if arr[j] is not ele:
            return False
    return True


def fastSlowPointer(arr):
    '''
    这是个数组找中点的一般方法，根据i，j初值不同可以处理不同的需求
    :param arr:
    :return:
    '''
    i, j = -1, 0  # arr长度为奇数，返回正中的元素，arr长度为偶数，返回中间两数中靠前的数
    # i, j = 0, 1 # arr长度为奇数，返回正中的元素，arr长度为偶数，返回中间两数中靠后的数

    while j < len(arr):
        j += 2
        i += 1
    print(arr)
    print(arr[i])
    return


def camelBananaPuzzle():
    dp = [[-1 for i in range(1001)] for j in range(101)]

    # Recursive function to find the maximum
    # number of bananas that can be transferred
    # to A distance using memoization
    def recBananaCnt(A, B, C):

        # Base Case where count of bananas
        # is less that the given distance
        if (B <= A):
            return 0

        # Base Case where count of bananas
        # is less that camel'q capacity
        if (B <= C):
            return B - A

        # Base Case where distance = 0
        if (A == 0):
            return B

        # If the current state is already
        # calculated
        if (dp[A][B] != -1):
            return dp[A][B]

        # Stores the maximum count of bananas
        maxCount = -2 ** 32

        # Stores the number of trips to transfer
        # B bananas using a camel of capacity C
        tripCount = ((2 * B) // C) - 1 if (B % C == 0) else ((2 * B) // C) + 1

        # Loop to iterate over all the
        # breakpoints in range [1, A]
        for i in range(1, A + 1):

            # Recursive call over the
            # remaining path
            curCount = recBananaCnt(A - i, B - tripCount * i, C)

            # Update the maxCount
            if (curCount > maxCount):
                maxCount = curCount

                # Memoize the current value
                dp[A][B] = maxCount

        # Return answer
        return maxCount

    # Function to find the maximum number of
    # bananas that can be transferred
    def maxBananaCnt(A, B, C):

        # Function Call
        return recBananaCnt(A, B, C)

    # Driver Code
    A = 100
    B = 1000
    C = 100
    print(maxBananaCnt(A, B, C))


def generateInputArr(N: int) -> List[int]:
    rand_int = random.randint(0, N)
    ret = []
    for i in range(N):
        if i == rand_int:
            ret.append(i)
        else:
            ret.append(i)
            ret.append(i)
    return ret


def findUniqueDuplicate(arr: List[int]) -> int:
    '''
    利用异或操作找一个array中出现了奇数次的唯一数字(剩余数字均出现偶数次)
    :param arr:
    :return:
    '''
    ret = 0
    for i in arr:
        ret = ret ^ i
    return ret

# 递归应用
def hanoi(n: int):
    def hanoiHelper(i, from_, to_, other_):
        if i == 1:
            print(f"Move 1 from {from_} to {to_}")
        else:
            hanoiHelper(i - 1, from_, other_, to_)
            print(f"Move {str(i)} from {from_} to {to_}")
            hanoiHelper(i - 1, other_, to_, from_)

    if n > 0:
        hanoiHelper(n, from_="左", to_="右", other_="中")


def allSubString(string_: str, i: int, res: str, all_res: List[str]):
    '''
    递归实现当前集合的全部子集，all_res为记录最终结果的变量，string_为全排列组合的原始字符串，i代表当前递归处理第i个位置的字符，
    res为当前构造的子集， 主函数调用的时候初始化all_res为空，i为0，res为空字符串
    :param string_: 原始字符串
    :param i: 当前来到的字符串的第几个字符
    :param res: 当前构造的结果
    :param all_res: 所有结果的列表
    :return: 不返回值，所有结果存在all_res里面
    '''
    if i == len(string_):
        # 当i来到了string_末尾，说明当前子集的构造完毕，加入最终结果
        all_res.append(res)
        return

    # 当我们有一个构造不完整的子集并来到某一个字符时，有两种选择，一是把当前字符加入这个子集，另一种是不把当前字符加入子集，我们可以根据这两种
    # 选择分别进行下一个字符的递归操作，这样即可包含所有的子集选择
    # 拷贝一下当前的子集
    resKeep = copy.deepcopy(res)
    # 如果把当前字符加入子集则为resKeep
    resKeep += string_[i]
    # 以resKeep作为最新构造好的子集，从下一个index开始递归
    allSubString(string_, i + 1, resKeep, all_res)
    # 如果不把当前字符加入子集则为resNoInclude
    resNoInclude = copy.deepcopy(res)
    # 以resNoInclude作为最新构造好的子集，从下一个index开始递归
    allSubString(string_, i + 1, resNoInclude, all_res)


def allPermutation(string_: str, all_res: List[str]) -> List[str]:
    def swap(array, i, j):
        array[i], array[j] = array[j], array[i]
        return

    def process1(string_lst: List[str], i: int, all_res: List[str]):
        '''
        递归实现全排列，all_res为记录最终结果的变量，string_为全排列组合的原始列表，i代表当前递归处理第i个位置的字符
        主函数调用的时候初始化all_res为空，i为0
        :param string_:
        :param i:
        :param all_res:
        :return:
        '''
        if i == len(string_lst):
            all_res.append("".join(string_lst))
        visited = [False] * 26
        for j in range(i, len(string_lst)):
            if not visited[ord(string_lst[j]) - ord("a")]:
                visited[ord(string_lst[j]) - ord("a")] = True
                # 当前在i时，可选的字符是从i开始到结尾的任意一个，于是我们以交换的方式做选择，尝试之后的每一个字符
                string_lst[i], string_lst[j] = string_lst[j], string_lst[i]
                # 选择完当前字符之后，递归到下一个index,同样可以选择从i+1开始到结尾的任意一个
                process(string_lst, i + 1, all_res)
                # 尝试完成之后换回来
                string_lst[i], string_lst[j] = string_lst[j], string_lst[i]

    def process(string_lst: List[str], i: int, all_res: List[str]):
        if i == len(string_lst):
            all_res.append("".join(string_lst))

        for j in range(i, len(string_lst)):
            swap(string_lst, i, j)
            process(string_lst, i + 1, all_res)
            swap(string_lst, i, j)
        return

    # main
    if not string_ or len(string_) == 0:
        return all_res

    lst = list(string_)
    process(lst, 0, all_res)
    return all_res


# 二叉树应用
def lowestCommonAncestor(head: BinaryTree, n1: BinaryTree, n2: BinaryTree):
    '''
    最低公共祖先问题
    :param head: 整棵树的根节点
    :param n1: 要找公共祖先的第一个节点
    :param n2: 要找公共祖先的第二个节点
    :return: 公共祖先节点
    '''
    # base case为，递归到某个node为n1或n2，直接返回， None时说明是叶节点的左或者右，直接返回none
    if head == n1 or head == n2 or head is None:
        return head

    left = lowestCommonAncestor(head.left, n1, n2)
    right = lowestCommonAncestor(head.right, n1, n2)

    if left is not None and right is not None:
        return head

    return left if left else right


def isBSTRecursive(head: BinaryTree) -> bool:
    '''
    检查一棵树是不是二叉搜索树，搜索二叉树的定义为，树的每个节点左树所有元素小于等于该节点值，右树所有元素大于该节点值。
    如果是二叉搜索树，最直观的方法是他的中序遍历形成的列表是个不降的列表，也可以找每个节点左树最右节点判断是不是小于等于当前节点值，如果违反了规则
    就返回否,或者根据定义用递归的方法做
    :param head: 被判断二叉树的根节点
    :return:
    '''
    class Info:
        def __init__(self, isBST, Max=None, Min=None):
            self.Max = Max
            self.Min = Min
            self.isBST = isBST

    def helper(root: BinaryTree) -> Info:
        if not root:
            # 节点为空时，定义最大值为-inf，最小值为inf，接下来的单独节点一定可以满足大于左max小于右min
            return Info(True, Max=float("-inf"), Min=float("inf"))
        if not root.left and not root.right:
            # 节点为叶子节点，则把max和min值都设为当前节点值
            return Info(True, Max=root.value, Min=root.value)

        leftInfo = helper(root.left)
        rightInfo = helper(root.right)
        leftMax = leftInfo.Max
        rightMin = rightInfo.Min
        isBST = leftInfo.isBST and rightInfo.isBST and leftMax < root.value < rightMin
        return Info(isBST, max([leftInfo.Max, root.value, rightInfo.Max]), min(leftInfo.Min, root.value, rightInfo.Min))

    return helper(head).isBST


def isBSTInOrderRecursive(head: BinaryTree) -> bool:
    def inorder(root, prevalue):
        if not root:
            return True
        # 当前节点左树需要满足inorder
        if not inorder(root.left):
            return False
        # 左树最大值需要小于当前节点
        if prevalue >= root.val:
            return False
        else:
            # 更新左树最大值
            prevalue = root.val

        # 判断右树满足inorder
        return inorder(root.right, prevalue)

    return inorder(head, float("-inf"))


def isCompleteBinaryTree(head: BinaryTree) -> bool:
    '''
    检查一棵二叉树是否为一棵完全二叉树, 叶子结点只能出现在最下层和次下层，且最下层的叶子结点集中在树的左部
    层序遍历整棵树
    1. 若任何一个节点有右孩子无左孩子，直接返回false， 2. 若遇到左右孩子不全的节点且不是1的情况，则从这个节点开始接下来的节点必须是叶节点
    :param head:
    :return:
    '''
    # 空树是完全二叉树
    if not head:
        return True

    from collections import deque

    q = deque([head])
    firstLeaf = False
    while q:
        cur = q.popleft()
        if firstLeaf:
            # 如果已经找到了第一个叶节点，那么只要目前的节点不是叶节点，就返回false
            if cur.left or cur.right:
                return False
        # 还没找到第一个叶节点的时候
        if cur.left and cur.right:
            # 左右双全，就正常按层序遍历进行
            q.append(cur.left)
            q.append(cur.right)
        elif cur.right and not cur.left:
            # 有右无左，返回false
            return False
        elif cur.left and not cur.right:
            # 有左无右，找到了叶节点起始位置，把左节点加进队列
            firstLeaf = True
            q.append(cur.left)
        else:
            # 无左无右，找到叶节点
            firstLeaf = True
    return True


def isFullBinaryTree(head: BinaryTree)-> bool:
    '''
    检验树是否为完满二叉树，即除了叶子节点每个节点都有两个子节点，判断方法和完全二叉树类似，即层序遍历二叉树，找到第一个叶节点，则之后的节点必须
    都是叶节点，否则就不是完满树
    :param head:
    :return:
    '''
    if not head:
        return True
    firstLeaf = False
    from collections import deque
    q = deque([head])
    while len(q) > 0:
        cur = q.popleft()
        if firstLeaf:
            # 如果之前已经找到了第一个叶节点，但当前节点出现了孩子，则返回False
            if cur.left or cur.right:
                return False
        else:
            if (cur.left and (not cur.right)) or ((not cur.left) and cur.right):
                # 节点只有左孩子或者只有右孩子，直接返回False
                return False
            elif cur.left is None and cur.right is None:
                # 没有左孩子和右孩子，则找到了第一个叶节点
                firstLeaf = True
            else:
                # 有左孩子和右孩子，按照遍历的方法继续下去
                q.append(cur.left)
                q.append(cur.right)
    # 遍历完了，返回True
    return True


def isPerfectBinaryTree(head: BinaryTree):
    '''
    检查一棵二叉树是否为完美二叉树（国内教材管这个叫满二叉树，傻逼翻译）
    除了叶节点，每个节点都有两个child,一颗树深度为h，最大层数为k，深度与最大层数相同，k=h;
    它的叶子数是： 2^h, 第k层的结点数是： 2^(k-1), 总结点数是： 2^k-1 (2的k次方减一), 总节点数一定是奇数。
    则可以利用根节点层数和整棵树节点个数之间的关系来判断是否为满树，为此需要递归找到每一个子树的高度和节点个数信息
    :param head:
    :return:
    '''
    class Info:
        def __init__(self, height, numofnodes):
            self.height = height
            self.numofnodes = numofnodes

    def helper(node: BinaryTree) -> Info:
        if not node:
            return Info(0, 0)

        leftInfo = helper(node.left)
        rightInfo = helper(node.right)
        height = max(leftInfo.height, rightInfo.height) + 1
        nunofnodes = leftInfo.numofnodes+rightInfo.numofnodes+1
        return Info(height, nunofnodes)

    # main
    if not head:
        return True
    info = helper(head)

    return info.numofnodes == 1 << info.height - 1


def isBalancedBST(head: BinaryTree):
    '''
    检查一棵二叉树是否为平衡二叉树，平衡二叉树的定义为，二叉树中任意一个节点为头的子树左右树高度差不超过1
    :param head:
    :return:
    '''
    class Info:
        def __init__(self, balanced, height):
            self.balanced = balanced
            self.height = height

    def helper(root: BinaryTree) -> Info:
        if not root:
            return Info(True, 0)

        leftInfo = helper(root.left)
        rightInfo = helper(root.right)
        treeBalanced = leftInfo.balanced and rightInfo.balanced and abs(leftInfo.height - rightInfo.height) <= 1
        height = 1 + max(leftInfo.height, rightInfo.height)
        return Info(treeBalanced, height)

    return helper(head).balanced


def isSameBinaryTree(h1: BinaryTree, h2: BinaryTree):
    # 检查两个二叉树结构上是不是一样
    if h1 is None and h2 is None:
        return True
    elif h1 is None and h2 is not None:
        return False
    elif h1 is not None and h2 is None:
        return False
    elif h1.value != h2.value:
        return False

    return isSameBinaryTree(h1.left, h2.left) and isSameBinaryTree(h1.right, h2.right)


def isSymmetricBinaryTree(h1: BinaryTree, h2: BinaryTree):
    if h1 is None and h2 is None:
        return True
    if h1 is not None and h2 is None:
        return False
    if h1 is None and h2 is not None:
        return False
    if h1.value != h2.value:
        return False
    return isSymmetricBinaryTree(h1.left, h2.right) and isSymmetricBinaryTree(h1.right, h2.left)


def inOrderSuccessorWithP(head: BinaryTree, node: BinaryTree) -> BinaryTree:
    '''
    找二叉树中序后继节点的函数,带父节点指针
    :param head: 二叉树的根节点
    :param node: 找哪个节点的前驱节点
    :return:
    '''
    if node.right:
        node_R = node.right
        while node_R.left:
            node_R = node_R.left
        return node_R
    else:
        cur = node
        parent = node.parent
        while cur == parent.right:
            cur = parent
            parent = cur.parent

        return parent


def inOrderPredecessorWithP(head: BinaryTree, node: BinaryTree) -> BinaryTree:
    '''
    寻找中序前驱节点的函数，带父节点指针
    :param head: 二叉树的根节点
    :param node: 找哪个节点的后继节点
    :return:
    '''
    if node.left:
        node_L = node.left
        while node_L.right:
            node_L = node_L.right
        return node_L
    else:
        cur = node
        parent = node.parent
        while cur == parent.left:
            cur = parent
            parent = cur.parent
        return parent


# 递归改动态规划
def digit2Letters(arr: List[str], i: int) -> int:
    '''
    1 -> a, 2 -> b ... 26 -> z,给一个字符数组，返回所有可能的字符串组合的数量
    :param arr: 数字array
    :param i:
    :return:
    '''

    if i == len(arr):
        # 数组合数量，所以到了结尾就多一种组合
        return 1
    # 0
    if arr[i] == "0":
        return 0
    # 1
    if arr[i] == "1":
        res: int = digit2Letters(arr, i + 1)
        if i + 1 < len(arr):
            res += digit2Letters(arr, i + 2)
        return res
    # 2
    if arr[i] == "2":
        res: int = digit2Letters(arr, i + 1)
        if i + 1 < len(arr) and "6" >= arr[i + 1] >= "0":
            res += digit2Letters(arr, i + 2)
        return res
    # 3~9
    return digit2Letters(arr, i + 1)


def bagValue(weights: List[int], values: List[int], bag: int) -> int:
    '''
    01背包问题给定一列重量，给定每个重量对应的价值，给定一个包的重量上限，返回这个包的最大价值
    每个只能物品用一次
    :param weights:
    :param values:
    :param bag:
    :return:
    '''

    def process(weights: List[int], values: List[int], i: int, agg_weight: int, agg_val: int,
                res_list: List[List[int]]):
        '''

        :param weights: 每个物品的重量
        :param values: 每个物品的价值
        :param i: 当前来到list的哪个位置
        :param agg_weight: 目前的总重量
        :param agg_val: 目前的总价值
        :param res_list:
        :return:
        '''


        if i == len(weights):
            res_list.append([agg_weight, agg_val])
            return

        # 当前位置的物品加进背包或不加进背包分别递归
        # 把所有可能的2**n结果放进一个list

        w_withCur, v_withCur = agg_weight + weights[i], agg_val + values[i]
        process(weights, values, i + 1, w_withCur, v_withCur, res_list)
        process(weights, values, i + 1, agg_weight, agg_val, res_list)
        return

    res = []
    process(weights, values, 0, 0, 0, res)
    running_v = float("-inf")
    # 循环所有可能的结果，找出符合条件的最大价值
    for [w, v] in res:
        if v > running_v and bag >= w:
            running_v = v

    return running_v

def bagValueDP(weights: List[int], values: List[int], bag: int) -> int:
    N = len(weights)
    dp = [[None for _ in range(bag + 1)] for __ in range(N + 1)]
    for index in range(N -1, -1, -1):
        pass


def countIsland(grid: List[List[int]]) -> int:
    '''
    岛问题: 1为陆地0为水，1上下左右为1时说明陆地联通，给一个矩阵返回里面有几个岛
    leetcode衍生问题还有求这些岛最大的面积是多少之类的， 解法很相似
    :ivar'''

    def infect(grid_, i_, j_, M_, N_):
        if i_ < 0 or i_ >= M or j_ < 0 or j_ >= N or grid_[i_][j_] != 1:
            return
        grid_[i_][j_] = 2
        infect(grid_, i_ + 1, j_, M_, N_)
        infect(grid_, i_ - 1, j_, M_, N_)
        infect(grid_, i_, j_ + 1, M_, N_)
        infect(grid_, i_, j_ - 1, M_, N_)
        return

    M = len(grid)
    if M == 0:
        return 0
    N = len(grid[0])
    res = 0
    for i in range(M):
        for j in range(N):
            if grid[i][j] == 1:
                res += 1
                infect(grid, i, j, M, N)
    return res


def _01Matrix(mat):
    '''
    上下左右相邻的格距离为1
    :param mat:一个值只有0或1的矩阵
    :return: 一个距离矩阵，每个元素代表最近的0到当前位置的距离
    '''
    m, n = len(mat), len(mat[0])
    ret = [[float("inf") for j in range(n)] for i in range(m)]
    # 从上向下从左向右，只根据左和上的neigbour 更新当前位置的结果
    for i in range(m):
        for j in range(n):
            if mat[i][j] == 0:
                ret[i][j] = 0
            else:
                if i >= 1:
                    ret[i][j] = min(ret[i][j], ret[i-1][j] + 1)
                if j >= 1:
                    ret[i][j] = min(ret[i][j], ret[i][j-1] + 1)
    # 从下向上从右向左，只根据下和右的neigbour 更新当前位置的结果
    for i in range(m-1, -1, -1):
        for j in range(n-1, -1, -1):
            if i < m - 1:
                ret[i][j] = min(ret[i][j], ret[i+1][j]+1)
            if j < n - 1:
                ret[i][j] = min(ret[i][j], ret[i][j+1]+1)
    return ret


def rottenOranges(grid):
    '''
    grid中有0，1，2三种值，0代表空，1代表新鲜橘子，2代表烂橘子，烂橘子的上下左右四个方向的新鲜橘子过一分钟会变烂
    求所有橘子都烂掉一共要经历几分钟。这是一个推广的BFS问题
    :param grid:
    :return: 时间
    '''
    from collections import deque
    queue = deque([])

    fresh_oranges = 0
    R, C = len(grid), len(grid[0])
    for i in range(R):
        for j in range(C):
            if grid[i][j] == 1:
                fresh_oranges += 1
            elif grid[i][j] == 2:
                queue.append((i, j))

    # 初始化后加一个停止符，代表这一轮感染结束了
    queue.append((-1, -1))
    # 时间初始化为-1是因为我们会在所有橘子都process完之后仍然存在一个停止符所以会多加一次1
    time_elapsed = -1

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while len(queue) > 0:
        r, c = queue.popleft()
        if r == -1:
            #一轮烂结束，时间加一分钟
            time_elapsed += 1
            if len(queue) > 0:
                # 到这里时，我们还要继续process下一轮，而且下一轮所有的烂橘子已经进queue了
                # 于是我们加一个停止符
                queue.append((-1 , -1))
        else:
            #找到了一个烂橘子，看上下左右四个位置是不是新鲜橘子，是的话，烂掉并更新一些变量
            for direction in directions:
                r_n, c_n = r + direction[0], c + direction[1]
                if 0 <= r_n < R and 0 <= c_n < C:
                    if grid[r_n][c_n] == 1:
                        # 如果这个位置是个新鲜橘子，那么就烂掉，如果不是的话就过了
                        grid[r_n][c_n] = 2
                        queue.append((r_n, c_n))
                        fresh_oranges -= 1
    return time_elapsed if fresh_oranges == 0 else -1


def KMPfind(q: str, m: str) -> int:
    '''
    KMP算法，从s中找子串m, 返回第一个找到的位置，如果没找到返回-1
    :param q:
    :param m:
    :return:
    '''
    if (not q) or (not m) or len(m) < 1 or len(q) < len(m):
        return -1

    str1 = list(q)
    str2 = list(m)
    i1, i2 = 0, 0
    nextArr = getNextArr(str2)  # O(M)

    while i1 < len(str1) and i2 < len(str2):  # O(N)
        if str1[i1] == str2[i2]:
            # 字符match上的时候同时往后移动
            i1 += 1
            i2 += 1
        elif i2 == 0: # next[i2]==-1
            # i2 == 0,因为只有这个位置的nextArr是-1，此时i2没有办法再在str2中向前跳，这时str1需要往后走一个
            i1 += 1
        else:
            # 此时不match，但是i2还可以在str2中往前跳,于是通过nextArr往前跳
            i2 = nextArr[i2]

    return i1 - i2 if i2 == len(str2) else -1


def getNextArr(str2: List[str]) -> List[int]:
    '''
    str2的每个字符前的sub string，如果要求前缀和后缀相等，最长的前/后缀长度是多少
    [asdasdc] -> [-1, 0, 0, 1, 2, 3]
    :param str2:
    :return:
    '''
    res = [None] * len(str2)
    if len(str2) == 1:
        res[0] = -1
        return res
    res[0] = -1
    res[1] = 0
    # 0 和 1位置根据定义是固定的数字，从2开始
    i = 2
    cn = res[1]  # 用哪个位置的字符和i-1位置字符比较,同时也是i-1位置最长缀的信息 # cn初始化= 0
    while i < len(str2):
        # 前一个位置最长前缀的下一个字符如果等于当前位置前一个位置的字符，则当前位置最长前缀为前一个位置最长前缀加1
        # eg: abc[] ... abcd, 若[]==d则最长等长前后缀为4
        if str2[i - 1] == str2[cn]:
            # cn位置和i-1位置字符对上了，当前i位置前后缀长为前一个位置前后缀长度加1
            cn += 1  # cn=res[i-1]所以这里等同于res[i-1]+1
            res[i] = cn
            i += 1
        elif cn > 0:
            # 当前不匹配，且还可以向前找，cn向前
            cn = res[cn]
        else:
            # 当前不匹配，而且cn没法再往前找了，当前i位置前后缀长为0，继续看下一个
            res[i] = 0
            i += 1

    return res


def longestPalindromeManacher(str1: str) -> int:
    '''
    Manacher是一个找序列回文半径的O(N)复杂度算法，利用之前位置的回文半径信息以及回文的对称性来加速查找过程
    :param str1:
    :return:
    '''

    if len(str1) == 0 or (not str1):
        return 0

    # 增加占位符: 1221 -> #1#2#2#1#
    str_arr = list("#" + "#".join(list(str1)) + "#")
    # 初始化回文半径数组
    pArr = [0] * len(str_arr)
    # 中心，和最右边界+1， 即最右有效边界到R-1
    C = -1
    R = -1
    maxx = float("-inf")
    for i in range(len(str_arr)):
        # 不用检验的，从i为中心往两边扩最少的回文区域:逻辑为：A.如果i在R外面，则至少长度为1(因为自己跟自己一定是回文)，B.若i在R里面，
        # 则为i'半径和R到i距离的最小值，其中i'半径时为当i'位置，回文区域完全被L..R包住的情况；R到i时为i'半径恰好在边界上，
        # 此时当前i位置半径至少为R-i，还可能往外扩，但是这里我们先set成R-i
        # ** i'为C向左距离(C-i)
        pArr[i] = min(R - i, pArr[C - (i - C)]) if R > i else 1

        # B的恰好在边界的case：从不用验区域的下一个开始，看能否往外扩，可以的话就往外扩并增加pArr当前位置回文半径的长度，否则break
        while i + pArr[i] < len(str1) and i - pArr[i] > -1:
            if str_arr[i + pArr[i]] == str[i - pArr[i]]:
                pArr[i] += 1
            else:
                break

        if i + pArr[i] > R:
            R = i + pArr[i]
            C = i
        maxx = max(maxx, pArr[i])
    return maxx - 1


def maxPathInBinaryTree(root: BinaryTree) -> int:
    '''
    二叉树中从任意节点a出发，每次可以向上或向下走，但每个点只能经过一次，到达节点b时路径上节点个数为a到b的距离，据此，每两个节点间都有距离，
    求整棵树上的最大距离
    :param root:
    :return:
    '''

    class Info:
        def __init__(self, maxPathLength: int = 0, height: int = 0):
            self.maxLength = maxPathLength
            self.height = height

    def process(root_: BinaryTree) -> Info:
        if not root_:
            return Info()

        linfo = process(root_.left)
        rinfo = process(root_.right)

        maxL = max([linfo.maxLength, rinfo.maxLength, 1 + linfo.height + rinfo.height])
        hei = max(linfo.height, rinfo.height) + 1

        return Info(maxL, hei)

    return process(root).maxLength


def maxHappiness(boss) -> int:
    class Employee:
        def __init__(self, happyness: int, suboridates: List):
            self.happness = happyness
            self.subordinates = suboridates

    class Info:
        def __init__(self, shown_happy, not_shown_happy):
            '''

            :param shown_happy:如果当前节点员工去party，以这个员工开始的结构整体的最大happy值
            :param not_shown_happy: 如果当前节点员工不去party，以这个员工开始的结构整体的最大happy值
            '''
            self.shown_happy = shown_happy
            self.not_shown_happy: not_shown_happy

    def process(root: Employee) -> Info:
        if len(root.subordinates) == 0:
            return Info(root.happness, 0)
        curr_shown_happy, curr_no_shown_happy = 0, 0
        for employee in root.subordinates:
            # 如果当前employee参加，则需要加上subordinates不参加的快乐值，
            # *反之需要加上subordinates参加或不参加时更大的的快乐值
            curr_info = process(employee)
            curr_shown_happy += curr_info.shown_happy
            curr_no_shown_happy += max(curr_info.shown_happy, curr_info.not_shown_happy)

        # 如果当前employee参加，加上当前employee的快乐值
        curr_shown_happy += root.happness
        return Info(curr_shown_happy, curr_no_shown_happy)

    # 主函数：从boss出发返回info，选二者里快乐值较大的返回
    boss_info = process(boss)
    return max(boss_info.shown_happy, boss_info.not_shown_happy)


def morris(head: BinaryTree):
    '''
    Morris遍历实现，时间复杂度0（N），空间复杂度O(1)，利用叶节点的空指针来节省空间
    :param head:
    :return:
    '''
    if not head:
        return

    mostRight = None
    cur = head
    while cur:  # 循环直至来到空节点
        mostRight = cur.left
        if mostRight:
            # 当前节点有左树：找到左树最右节点，若最右节点的右节点为空，将其指向当前节点，当前节点向左移
            while (mostRight is not None) and (mostRight is not cur):
                mostRight = mostRight.right
            if not mostRight.right:
                mostRight.right = cur
                cur = cur.left
                continue
            else:
                # 当前节点有左树：找到左树最右节点，若最右节点的右节点不为空，将其指向空节点，当前节点向右移
                mostRight.right = None
        # 当前节点没有左树，则当前节点向右移动
        cur = cur.right
    return


def morrisPreTraversal(head: BinaryTree):
    '''
    Morris遍历完成前序遍历，没有左树的点经过即打印，有右树的点第一次经过时打印
    :param head:
    :return:
    '''
    if not head:
        return

    mostRight = None
    cur = head
    while cur:  # 循环直至来到空节点
        mostRight = cur.left
        if mostRight:
            # 当前节点有左树：找到左树最右节点，若最右节点的右节点为空，将其指向当前节点，当前节点向左移
            while (mostRight is not None) and (mostRight is not cur):
                mostRight = mostRight.right
            if not mostRight.right:
                print(cur.value)
                mostRight.right = cur
                cur = cur.left
            else:
                # 当前节点有左树：找到左树最右节点，若最右节点的右节点不为空，将其指向空节点，当前节点向右移
                mostRight.right = None
                cur = cur.right
        else:
            # 当前节点没有左树，则当前节点向右移动
            print(cur.value)
            cur = cur.right
    return


def morrisInTraversal(head: BinaryTree):
    '''
    Morris遍历完成in order traversal，没有左树的点经过即打印，有左树的点第二次经过时打印
    :param head:
    :return:
    '''

    if not head:
        return

    mostRight = None
    cur = head
    while cur:  # 循环直至来到空节点
        mostRight = cur.left
        if mostRight:
            # 当前节点有左树：找到左树最右节点，若最右节点的右节点为空，将其指向当前节点，当前节点向左移
            while (mostRight is not None) and (mostRight is not cur):
                mostRight = mostRight.right
            if not mostRight.right:
                mostRight.right = cur
                cur = cur.left
            else:
                # 当前节点有左树：找到左树最右节点，若最右节点的右节点不为空，将其指向空节点，当前节点向右移
                print(cur.value)
                mostRight.right = None
                cur = cur.right
        else:
            # 当前节点没有左树，则当前节点向右移动
            print(cur.value)
            cur = cur.right
    return


def morrisPostTraversal(head: BinaryTree):
    '''
    Morris遍历完成 post order traversal，没有左树的点经过即打印，
    有左树的点第二次经过时打印左树有边界的逆序，整体停下来之后，打印最右边界的逆序
    最右边界逆序打印: 单链表逆序打印，打印完再逆回来
    :param head:
    :return:
    '''

    def printEdge(node: BinaryTree):
        pass

    if not head:
        return

    mostRight = None
    cur = head
    while cur:  # 循环直至来到空节点
        mostRight = cur.left
        if mostRight:
            # 当前节点有左树：找到左树最右节
            while (mostRight is not None) and (mostRight is not cur):
                mostRight = mostRight.right
            if not mostRight.right:
                # 若最右节点的右节点为空，将其指向当前节点，当前节点向左移。第一次来到这个节点
                mostRight.right = cur
                cur = cur.left
            else:
                # 当前节点有左树：找到左树最右节点，若最右节点的右节点不为空，将其指向空节点，当前节点向右移
                # 第二次来到这个节点
                mostRight.right = None
                cur = cur.right
                printEdge(cur.left)
        else:
            # 当前节点没有左树，则当前节点向右移动
            print(cur.value)
            cur = cur.right
        printEdge(head)
    return

def rotate(nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: None Do not return anything, modify nums in-place instead.
    """
    n = len(nums)
    k %= n

    start_idx = count = 0
    while count < n:
        current_idx, prev = start_idx, nums[start_idx]
        while True:
            #从起点开始，把当前数字向右旋转k个位置，再把k个位置后的数字继续旋转，直到返回起始位置
            next_idx = (current_idx + k) % n
            nums[next_idx], prev = prev, nums[next_idx]
            current_idx = next_idx
            count += 1

            if start_idx == current_idx:
                break
        # 正常情况下，返回起始位置时，所有array元素都已经挪动过了一次，特殊情况是中间形成了环，在还没走完所有元素的时候就回到了起点，这时候要
        # 人工把起点向右挪动一个以打破循环继续做旋转，直至所有的元素都移动了一次
        start_idx += 1

def superWashingMachine(arr):
    # 洗衣机问题，单独考虑当前位置在左侧右侧和的各种情况下需要怎么向左向右操作以及需要操作几轮，整体的瓶颈就是最后的答案
    # 洗衣机问题：一个array n个元素，每个位置代表一个洗衣机里面的衣服，每次移动可以任选m个洗衣机，把每个被选中的洗衣机里的一件衣服同时放在当前洗衣机
    # 相邻的位置（可左可右），问多少轮后可以把状态变成每台洗衣机里面的衣服数量相等
    pre_sum = [0 for _ in arr]
    pre_sum[0] = arr[0]
    N = len(arr)
    for i in range(1, N):
        pre_sum[i] = pre_sum[i-1] + arr[i]
    totalSum = pre_sum[-1]
    if totalSum % N != 0:
        return -1
    avg = totalSum // N
    res = [None for _ in arr]
    ans = 0
    for i in range(N):
        leftAll = pre_sum[i] - arr[i]
        rightAll = totalSum - pre_sum[i]
        leftDiff = leftAll - i * avg
        rightDiff = rightAll - (N - i) * avg
        ans = max(ans, abs(leftDiff + rightDiff) if leftDiff < 0 and rightDiff < 0 else max(abs(leftDiff), abs(rightDiff)))

    return ans

# 滑动窗口
def maxInWindow(arr: List[int], size: int) -> List[int]:
    '''
    滑动窗口求窗口里的最大值
    :param arr:
    :param size:
    :return:
    '''
    if len(arr) <= size:
        return [max(arr)]
    l, r = 0, 0
    # window这里是list,需要考虑使用collections.deque保证从头和尾pop append均为O(1)时间
    window = collections.deque([])
    window_ = []
    ret = []
    # 以右指针为遍历变量
    while r < len(arr):
        if r - l >= size:
            # 若l，r距离大于等于size，说明滑动窗口达到要求长度了，这时首先把最值放进返回序列中
            ret.append(window[0])
            # 之后要移动l边界了，如果滑出窗口的值刚好是最值，则把他从窗口中pop掉
            if arr[l] == window[0]:
                window.popleft()
            l += 1

        # l已经向右移了一个，这时r也要向右移一个，window为单调队列结构，即结构中数据有序排列(此题中从大到小)，每次新元素进来时，要删掉数据结构
        # 中比新进来元素小的元素
        while len(window) > 0 and window[-1] < arr[r]:
            window.pop()
        window.append(arr[r])
        r += 1
    # 最后一个元素操作完后之前的while loop就停了，所以最后这里需要再加一次window中的最值
    ret.append(window[0])
    return ret


def minInWindow(arr: List[int], size: int) -> List[int]:
    if len(arr) <= size:
        return [min(arr)]

    l, r = 0, 0
    # window这里是list,考虑使用collections.deque来保证头尾pop的复杂度为O(1)
    window = collections.deque([])
    window_ = []
    ret = []
    while r < len(arr):
        if r - l >= size:
            ret.append(window[0])
            if window[0] == arr[l]:
                window.popleft()
            l += 1

        while len(window) > 0 and window[-1] > arr[r]:
            window.pop()
        window.append(arr[r])
        r += 1

    ret.append(window[0])
    return ret


def longestSubstringWithoutRepeatingChars(q):
    '''
    寻找一个字符串中最长的没有重复字符的子字符串的长度(子串而非子序列，子串需要连续)

    :param q:
    :return:
    '''
    n = len(q)
    i = 0 # 不含重复字符的字符子串左边界（包含）
    ans = 0
    mp = {}

    # 以j位置结尾的串最长不含重复字符的子串长度
    for j in range(n):
        char = q[j]
        # j 为字符字串右边界(包含)
        if q[j] in mp:
            # 找到了一个前面出现过的字符，那么更新左边界到这个最新的位置
            i = max(i, mp[char])
        ans = max(ans, j - i + 1)
        mp[char] = j + 1 # 这里要加1以保证i在更新之后子串中不包含重复的字符
    return ans


def permutationMatchInAString(s1, s2):
    '''
    判断一个长字符串是不是包含一个短字符串的某个排列作为子串
    利用滑动窗口更新一个字典
    :param s1: 短字符串
    :param s2: 长字符串
    :return:
    '''
    def isSameDict(d1, d2):
        for i in range(26):
            if d1[i] != d2[i]:
                return False
        return True

    dict_s1 = [0] * 26
    dict_sld = [0] * 26
    for char_ in s1:
        dict_s1[ord(char_) - ord('a')] += 1
    l_s1 = len(s1)
    l_s2 = len(s2)
    for i in range(l_s1 - 1, l_s2):
        if i == l_s1 - 1:
            for j in range(l_s1):
                dict_sld[ord(s2[j]) - ord('a')] += 1
        else:
            dict_sld[ord(s2[i-l_s1]) - ord('a')] -= 1
            dict_sld[ord(s2[i]) - ord('a')] += 1
        if isSameDict(dict_sld, dict_s1):
            return True
    return False



# 数学
def gcd(a, b):
    # 辗转相除法求最大公约数
    d = a if b == 0 else gcd(b, a % b)
    return d

def lcm(a, b):
    # 通过关系式求最小公倍数
    return int(a * b / gcd(a, b))

def coprime(n):
    '''
    求从1到n-1中和n互质的数的数量
    :param n:
    :return:
    '''
    pass

def primefactor(n):
    '''
    正整数n的质因数分解，返回一个字典
    key -> value: 质数->个数
    :param n:
    :return:
    '''
    if n in {1,2,3}:
        return {n:1}
    curr = 2
    ret = dict()
    while curr ** 2 <= n:
        while n % curr == 0:
            n = n // curr
            if curr not in ret:
                ret[curr] = 1
            else:
                ret[curr] += 1
        curr += 1
    if n > 1:
        ret[n] = 1
    return ret

def isprime(n):

    if n <= 2:
        return False
    else:
        i = 2
        while i ** 2 <= n:
            if n % i == 0:
                return True
            i += 1
        return False
# 二进制数神奇操作
def flip(n: int):
    '''
    n为0或者1
    :param n:
    :return:
    '''
    # n = 1 or 0
    return n ^ 1


def sign(n):
    '''
    n  >> 31: 非负返回0,负数返回-1
    (n  >> 31) & 1: 非负返回0,负数返回1
    after flip: 非负返回1，负数返回0
    :param n:
    :return:
    '''
    return flip(n >> 31) & 1


def getMax1(a, b):
    '''
    不用比较运算返回a和b中最大值
    :param a:
    :param b:
    :return:
    '''
    c = a - b
    atimes = sign(c)
    btimes = flip(atimes)
    return atimes * a + btimes * b


def getMax2(a, b):
    '''
    考虑 a-b溢出的可能，改进
    :param a:
    :param b:
    :return:
    '''
    c = a - b
    sa = sign(a)
    sb = sign(b)
    sc = sign(c)
    difSab = sa ^ sb
    sameSab = flip(difSab)
    # 返回a的条件：a和b符号相同且a大于等于b，即c大于等于0，或者a和b符号不同且a大于等于0,这两个条件互斥
    returnA = difSab * sa + sameSab * sc
    returnB = flip(returnA)
    return a * returnA + b * returnB


def ispowerof2(n):
    # return n > 0 & ((n & (-n)) == n)
    return (n & (n - 1) == 0) & (n != 0)


def ispowerof4(n):
    '''
    0x55555555 -> b010101010101....
    :param n:
    :return:
    '''
    return (n & (n - 1) == 0) & (n != 0) & (n & 0x55555555 != 0)


def add_a_b(a: int, b: int):
    '''
    利用位运算和递归完成两个整数加法
    :param a:
    :param b:
    :return:
    '''

    # 进位信息若为0，则可以直接返回结果，结果为a
    if b == 0:
        return a

    # 异或等同于无进位相加
    x = a ^ b
    # 求与找到进位信息，左移表示进位发生
    y = (a & b) << 1

    # 异或结果传为a
    return add_a_b(x, y)


def negNum(n):
    '''
    位取反，加一，即为相反数
    :param n:
    :return:
    '''
    return add_a_b(~n, 1)


def sub_a_b(a: int, b: int):
    return add_a_b(a, negNum(b))


def mul_a_b(a: int, b: int):
    def unsigned_right_shift(n, i):
        import ctypes
        # 数字小于0，则转为32位无符号uint
        if n < 0:
            n = ctypes.c_uint32(n).value
        return n >> i

    res = 0
    # 等同于10进制乘法规则的二进制乘法，被乘数每次往左移一位和乘数最后一位相乘，然后加在res上
    while b != 0:
        if (b & 1) != 0:
            res = add_a_b(res, a)
        a = a << 1
        b = unsigned_right_shift(b, 1)
    return res


def div_a_b(a: int, b: int):
    x = negNum(a) if a < 0 else a
    y = negNum(b) if b < 0 else b
    res = 0

    i = 31
    while i > 0:
        # 找到除数往左移动几位可以和被除数对齐且小于被除数，把这一位加到结果中，除数右移一位，直至除数变为0
        if y <= (x >> i):
            res = res | (1 << i)
            x = sub_a_b(x, y << i)
        i = sub_a_b(i, 1)
    return negNum(res) if (a < 0) ^ (b < 0) else res


def robotWalk(N, S, E, K) -> int:
    '''
    递归版本求机器人从起始位置走到终点位置，限制只能走K步，一共有多少种走法
    :param N: 机器人可以走的位置1...N
    :param K: 机器人可以走几步
    :param S: 机器人的起始位置
    :param E: 机器人的终止位置
    '''
    # current = 1..N
    # rest = 0..K
    def f1(N, current, end, rest):
        if rest == 0:
            return 1 if current == end else 0
        if current == 1:
            return f1(N, 2, end, rest - 1)
        if current == end:
            return f1(N, end - 1, end, rest - 1)
        return f1(N, current + 1, end, rest - 1) + f1(N, current - 1, end, rest - 1)

    # 主函数
    return f1(N, S, E, K)

def dpMemoryRobotWalk(N, K, S, E) -> int:
    '''
    记忆化搜索版本的机器人走路问题，相比较递归版本多了一个缓存性质的二维数组，这样已经算出的结果可以直接从缓存里查出来，不需要再次走递归
    :param N: 机器人可以走的位置1...N
    :param K: 机器人可以走几步
    :param S: 机器人的起始位置
    :param E: 机器人的终止位置
    :return:
    '''

    def f2(N, rest, cur, E, cache):
        if cache[cur][rest] is not None:
            return cache[cur][rest]

        if rest == 0:
            cache[cur][rest] = 1 if cur == E else 0

        if cur == 1:
            cache[cur][rest] = f2(N, rest - 1, 2, E, cache)
        elif cur == E:
            cache[cur][rest] = f2(N, rest - 1, E - 1, E, cache)
        else:
            cache[cur][rest] = f2(N, rest - 1, cur - 1, E, cache) + f2(N, rest - 1, cur + 1, E, cache)
        return cache[cur][rest]

    # 定义dp缓存，外层括号为dp[x][y]中的x范围，内层括号为y范围
    dp_cache = [[None for i in range(K + 1)] for j in range(N + 1)]
    return f2(N, K, S, E, dp_cache)

def dpRobotWalk(N, K, S, E) -> int:
    '''
    动态规划解机器人问题
    :param N: 机器人可以走的位置1...N
    :param K: 机器人必须走几步
    :param S: 机器人的起始位置
    :param E: 机器人的终止位置
    :return:
    '''
    # current = 1..N
    # rest = 0..K
    # dp[current][rest]
    # dp表代表：在还剩rest步并且当前处于current位置时，最后成功到终点E的方法数
    dp = [[None for _ in range(K+1)] for __ in range(N+1)]
    for current in range(K+1):
        dp[current][0] = 1 if current == E else 0

    for rest in range(1, K+1):
        dp[1][rest] = dp[2][rest-1]
        for curr in range(2,N):
            dp[curr][rest] = dp[curr-1][rest-1] + dp[curr+1][rest-1]
        dp[N][rest] = dp[N-1][rest-1]
    return dp[S][K]


def minNumOfCoin01(arr, target):
    '''
    https://zhuanlan.zhihu.com/p/93857890
    https://github.com/tianyicui/pack/blob/master/V2.pdf
    0,1背包问题，arr为硬币面值，每个硬币最多用一次，问凑出target钱数的最小硬币数
    :param arr:
    :param target:
    :return:
    '''
    def process(arr, index, rest):
        '''
        递归子过程，表明从index开始，凑出rest这么多钱，需要的最小硬币数
        :param arr:
        :param index:
        :param rest:
        :return:
        '''
        if rest < 0:
            # 如果当前rest是负数，则之后几枚硬币也凑不出来, 返回inf
            return -1  # float("inf")
        if rest == 0:
            # 如果当前还需要凑的钱数是0,则还需要0枚硬币
            return 0
        # 还需要凑的钱超过0,但是已经走完了整个arr，即没有其他硬币可以选了，返回 inf
        if index == len(arr):
            return -1  # float("inf")

        # 接下来为一般情况，当前有硬币可选，也有rest的钱需要凑，那么可以选择要或者不要当前的硬币，返回两种选择中硬币数小的
        p1 = process(arr, index + 1, rest)
        p2post = process(arr, index + 1, rest - arr[index])
        if p1 == -1 and p2post == -1:
            return -1
        elif p1 == -1:
            return 1 + p2post
        elif p2post == -1:
            return p1
        return min(p1, 1 + p2post)

    num = process(arr, 0, target)
    return num


def minNumOfCoin01Memory(arr, target):
    # TODO:根据这个改一个纯动态规划的版本
    def processMemory(arr_, index, rest, cache):
        if rest < 0:
            return -1
        if cache[index][rest] != -2:
            return cache[index][rest]

        if rest == 0:
            cache[index][rest] = 0
            return cache[index][rest]
        if index == len(arr_):
            cache[index][rest] = -1
            return cache[index][rest]

        p2Next = processMemory(arr_, index + 1, rest - arr_[index], cache)
        p1 = processMemory(arr_, index + 1, rest, cache)

        if p2Next == -1 and p1 == -1:
            cache[index][rest] = -1
        elif p2Next == -1:
            cache[index][rest] = p1
        elif p1 == -1:
            cache[index][rest] = p2Next + 1
        else:
            cache[index][rest] = min(p1, 1 + p2Next)

        return cache[index][rest]

    dp = [[-2 for i in range(target + 1)] for j in range(len(arr) + 1)]
    res = processMemory(arr, 0, target, dp)
    return res


def minNumOfCoin01_dp(arr, target):
    res = [[float("inf") for _ in range(target + 1)] for __ in range(len(arr))]
    # res[i][rest_target], i means array
    # 首列
    for i in range(len(arr)):
        res[i][0] = 0
    # 首行
    for j in arr:
        res[0][j] = 1
    
    # 一般情况
    for i in range(1, len(arr)):
        for j in range(1, target+ 1):
            if j - arr[i] >= 0: # 不越界，则找上一行相同位置和上一行减去当前数目位置的较小值，即为了凑j我可以不使用或者使用当前位置的硬币
                res[i][j] = min(res[i-1][j], res[i-1][j-arr[i]]+1)
            else: # 左边越界了，说明当前位置的硬币没法用来凑j这个值，所以只能考虑上一行相同位置的值
                res[i][j] = res[i-1][j]
    return res[-1][-1] if res[-1][-1] != float("inf") else -1


def minNumOfCoin01_dpopt(arr, target):
    # 一个空间优化，根据状态转移方程，min(res[i-1][j], res[i-1][j-arr[i]]+1)，当前行存在一个对上一行左端的依赖，
    # 所以当前行需要从右往左算以保证在当前行更新的时候，依赖的上一行的值还没有被改动过
    res = [float("inf") for _ in range(target + 1)]
    # res[i][rest_target], i means array
    # 首列
    # 首行
    for j in (arr):
        res[j] = 1
    
    # 一般情况
    for i in range(0, len(arr)):
        res[0] = 0
        for j in range(target, -1, -1):
            if j - arr[i] >= 0: # 不越界，则找上一行相同位置和上一行减去当前数目位置的较小值，即为了凑j我可以不使用或者使用当前位置的硬币
                res[j] = min(res[j], res[j-arr[i]]+1)
    return res[-1] if res[-1] != float("inf") else -1


def waysofChange(arr, target):
    # TODO:写一下记忆化搜索和动态规划的版本
    '''
    完全背包找零问题，每个面值可以使用无数次，使用斜率优化降低dp一个维度的复杂度
    :param arr:
    :param target:
    :return:
    '''

    def process(arr, index, rest) -> int:
        '''

        :param arr: 面值array
        :param index: 当前来到第几个可能面值
        :param rest: 剩下需要凑多少钱
        :return: 当前状态下凑钱的方法有几种
        '''

        if index == len(arr):
            # 选完了所有可能面值，如果凑出了钱那么返回一种解法，否则返回0种解法
            return 1 if rest == 0 else 0

        nob = 0 # 当前的面值使用几张
        ways = 0
        while arr[index] * nob <= rest:
            # 限制为总钱数不能超过target
            ways += process(arr, index + 1, rest - nob * arr[index])
            nob += 1
        return ways

    return process(arr, 0, target)


def ABSmartGame2nd(arr: List[int], l: int, r: int) -> int:
    ## 当自己是后手时的递归函数
    if l == r:
        # 如果只有一个元素了，那么因为我是后手，我拿不到这个元素，返回0
        return 0
    # 如果还有不止一个元素，由于后手，最左或最右会先被另一个人选走，选完后我会被剩下结果比较差的然后继续我先手
    return min(ABSmartGame1st(arr, l + 1, r), ABSmartGame1st(arr, l, r - 1))


def ABSmartGame1st(arr: List[int], l: int, r: int) -> int:
    ## 当自己是先手时的递归函数
    if l == r:
        # 如果只有一个元素了，那么我拿走得到这个元素的大小
        return arr[l]
    # 如果还有不止一个元素，有两种选法，选最左边的然后后手进剩下的，或者选右边的后手进左边的，两者取大的就是我的决策
    return max(arr[l] + ABSmartGame2nd(arr, l + 1, r), arr[r] + ABSmartGame2nd(arr, l, r - 1))


def survivingBob(nrow, ncol, a, b, k):
    '''

    :param nrow:
    :param ncol:
    :param a:
    :param b:
    :param k:
    :return:
    '''

    def process(N, M, x, y, rest):
        if x < 0 or x >= N or y < 0 or y >= M:
            return 0

        if rest == 0:
            return 1

        live = process(N, M, x + 1, y, rest - 1)
        live += process(N, M, x, y + 1, rest - 1)
        live += process(N, M, x - 1, y, rest - 1)
        live += process(N, M, x, y - 1, rest - 1)
        return live

    numOfSurvival = process(nrow, ncol, a, b, k)
    return numOfSurvival / 4 ** k


def horseJump(x: int, y: int, k: int):
    '''

    :param x: 目标位置横坐标
    :param y: 目标位置纵坐标
    :param k: 一共跳的步数
    :return: 一共有多少种跳法
    递归函数可以理解成从x,y出发，还剩remainstep步的情况下回到0,0的跳法数量
    '''

    def process(xx, yy, remainstep) -> int:
        if xx < 0 or xx > 8 or yy < 0 or yy > 9:
            # 如果越界就返回0种
            return 0
        if remainstep == 0:
            # 如果没有多余步数了，那么当前如果在0，0就说明找到了一种跳法，否则就是没找到
            return 1 if xx == 0 and yy == 0 else 0
        # 一般情况：从当前点出发有8个位置可以跳，从0，0到当前点的方法为分别到那8个位置方法的和
        return process(xx + 2, yy + 1, remainstep - 1) + process(xx + 2, yy - 1, remainstep - 1) + \
               process(xx - 2, yy + 1, remainstep - 1) + process(xx - 2, yy - 1, remainstep - 1) + \
               process(xx + 1, yy + 2, remainstep - 1) + process(xx + 1, yy - 2, remainstep - 1) + \
               process(xx - 1, yy + 2, remainstep - 1) + process(xx - 1, yy - 2, remainstep - 1)

    return process(x, y, k)


def maximumSubArray(arr: List[int]) -> int:
    '''
    给一个序列，返回和最大的连续子序列的和， kadane's algorithm
    :param arr:
    :return:
    '''
    curr_max = 0
    global_max = float("-inf")
    for num in arr:
        curr_max += num
        global_max = max(global_max, curr_max)
        curr_max = max(curr_max, 0)
    return global_max


def isPalindromeLinkedList(head: ListNode) -> bool:
    # 234. Palindrome Linked List
    # o(n) time, o(1) space 最优解
    # o(n) time, o(n) space 容易写 -> 弄到list里面，两个指针头尾遍历一边
    # 找时间leetcode上写一下space o(1)的方法
    def reverseLinkedList(head:ListNode, tail: ListNode):
        # 区分起始点和终点的反转链表，要注意指针的指向，头的next必须是valid的ListNode
        if not head:
            return
        if head == tail:
            return head
        prev_node = None
        curr_node = head
        while prev_node != tail:
            next_node = curr_node.next
            curr_node.next = prev_node
            prev_node = curr_node
            curr_node = next_node
        return tail

    if not head:
        return True
    if not head.next:
        return True
    fast, slow = head, head
    while fast.next is not None and fast.next.next is not None:
        fast = fast.next.next
        slow = slow.next
    if fast.next is None:
        mid = slow
        tail = fast
    elif fast.next.next is None:
        mid = slow.next
        tail = fast.next
    left2right = head
    right2left = reverseLinkedList(mid, tail)
    isP = True
    while left2right and right2left:
        if left2right.val != right2left.val:
            isP = False
        left2right = left2right.next
        right2left = right2left.next
    right2left = reverseLinkedList(tail, mid)
    return isP


def isPalindromeLinkedListCheck(head: ListNode):
    # 234. Palindrome Linked List
    # o(n) time, o(1) space 最优解
    # o(n) time, o(n) space 容易写 -> 弄到list里面，两个指针头尾遍历一边
    # 找时间leetcode上写一下space o(1)的方法
    def reverseLinkedList(head:ListNode, tail: ListNode):
        if not head:
            return
        if head == tail:
            return head
        prev_node = None
        curr_node = head
        while prev_node != tail:
            next_node = curr_node.next
            curr_node.next = prev_node
            prev_node = curr_node
            curr_node = next_node
        return tail

    if not head:
        return True
    if not head.next:
        return True
    fast, slow = head, head
    while fast.next is not None and fast.next.next is not None:
        fast = fast.next.next
        slow = slow.next
    if fast.next is None:
        mid = slow
        tail = fast
    elif fast.next.next is None:
        mid = slow.next
        tail = fast.next
    left2right = head
    right2left = reverseLinkedList(mid, tail)
    isP = True
    while left2right and right2left:
        if left2right.val != right2left.val:
            isP = False
        left2right = left2right.next
        right2left = right2left.next
    right2left = reverseLinkedList(tail, mid)
    return isP


def createLinkedList(arr):
    if len(arr) == 0:
        return None

    head = ListNode(arr[0])
    cur = head
    for i in range(1, len(arr)):
        cur.next = ListNode(arr[i])
        cur = cur.next
    return head

def detectCycle(head):
    if not head:
        return None
    fast, slow = head, head
    while (fast.next is not None) and (fast.next.next is not None) and (fast != slow):
        fast = fast.next.next
        slow = slow.next

    if fast != slow:
        return None

    fast = head

    while fast != slow:
        fast = fast.next
        slow = slow.next

    return slow


def graphCheck1():
    edgeList = [
        [0, 2, -3],
        [4, 0, 1],
        [0, 5, 2],
        [1, 4, 1],
        [1, 5, 4],
        [2, 3, 4],
        [2, 4, 1],
        [4, 5, 5],
        [0, 7, 7],
        [6, 7, 1],
        [7, 8, 9],
        [7, 9, 2]
    ]
    graph = createGraphEdgeList(edgeList)
    dfs(graph.nodes[0])
    print(graphNegativeCycle(graph))

def minPaint(q):
    '''
    染色问题：RGRGR -> RGGGG或RRGGG，问最少染几个位置
    :param q: string
    :return: 最少染几个位置
    '''
    s_lst = list(q)
    leftG = [0] * len(q)
    rightR = [0] * len(q)

    # 预处理数组，leftG: 位置i及其左边有多少个G，rightR：位置i及其右边有多少个R
    leftG[0] = 1 if s_lst[0] == "G" else 0
    rightR[-1] = 1 if s_lst[-1] == "R" else 0
    for i in range(1, len(q)):
        leftG[i] = leftG[i-1] + (1 if s_lst[i] == "G" else 0)
        rightR[len(q)-1-i] = rightR[len(q)-i] + (1 if s_lst[len(q)-1-i] == "R" else 0)

    minPaint = min(leftG[-1], rightR[0])
    for i in range(len(q)):
        minPaint = min(minPaint, leftG[i] + rightR[i])
    return minPaint

def numMatchingSubseq(S, words):
    ans = 0
    heads = [[] for _ in range(26)]
    for word in words:
        it = iter(word)
        heads[ord(next(it)) - ord('a')].append(it)

    for letter in S:
        letter_index = ord(letter) - ord('a')
        old_bucket = heads[letter_index]
        heads[letter_index] = []

        while old_bucket:
            it = old_bucket.pop()
            nxt = next(it, None)
            if nxt:
                heads[ord(nxt) - ord('a')].append(it)
            else:
                ans += 1
    return ans

def reverseBetween( head, left: int, right: int):
    if head.next is None:
        return head
    if left == right:
        return head
    orig_start, orig_end = None, None
    orig_start_prev = None
    counter = 1
    orig_start = head
    while counter < left:
        orig_start_prev=orig_start
        orig_start=orig_start.next
        counter += 1

    prev_node = None
    orig_end = orig_start
    while counter <= right:
        next_node = orig_end.next
        orig_end.next = prev_node
        prev_node = orig_end
        orig_end = next_node
        counter += 1

    if orig_start_prev:
        orig_start_prev.next = prev_node
    orig_start.next = orig_end

    return head if left != 1 else prev_node


def NineTenAvgValue(arr):
    import heapq
    arr_negate = [-ele for ele in arr]

    ret = []
    heapq.heapify(ret)
    for i in range(len(arr_negate)):
        if len(ret) < 10:
            heapq.heappush(ret, arr_negate[i])
        else:
            if arr_negate[i] > ret[0]:
                heapq.heappushpop(ret, arr_negate[i])
        
    if len(ret) < 10:
        return -ret[0]
    else:
        ten = ret[0]
        heapq.heappop(ret)
        nine = ret[0]
        return -(ten + nine) / 2


def partition(arr, target):
    def swap(arr, i, j):
        arr[i], arr[j] = arr[j], arr[i]
        return

    left, curr, right = 0, 0, len(arr) - 1
    while curr <= right:
        if arr[curr] < target:
            swap(arr, left, curr)
            left += 1
            curr += 1
        elif arr[curr] > target:
            swap(arr, right, curr)
            right -= 1
        else:
            curr += 1
    return arr


def minNumOfCoinMN(values, nums, target):
    def helper(values, nums, i, leftVal, curr_coin_num, all_coin_num):
        if leftVal == 0:
            all_coin_num.append(curr_coin_num)
            return
        if leftVal < 0:
            return

        if i == len(values):
            return

        max_allow = nums[i]
        for j in range(max_allow + 1):
            helper(values, nums, i + 1, leftVal - j * values[i], curr_coin_num + j, all_coin_num)
        
    coins = []
    helper(values, nums, 0, target, 0, coins)

    return min(coins) if len(coins) > 0 else -1

def minNumOfCoinMN_dp(values, nums, target):
    dp = [float("inf") for _ in range(target + 1)]

    for j in range(len(values)):
        dp[0] = [0]
        for tgt in range(target, -1, -1):
            for k in range(0, nums[j] + 1):
                if tgt - k * j >= 0:
                    dp[tgt] = min(dp[tgt], dp[tgt - k * j] + j) 
    return dp[-1]


def jumpGame(arr):
    # arr元素代表从这个位置出发最多向右跳几步，问从0出发能不能跳到arr的结尾处
    # o(n^2)
    
    # res = [False for _ in range(len(arr))]
    # res[0] = True
    # for i in range(1, len(res)):
    #     for j in range(i):
    #         if res[j] and i + arr[i] >= j:
    #             res[i] = True
    #             break
    # return res[-1]

    # 贪心算法 o(n) 复杂度, 假设最后一个元素能到，初始化last_idx为最后，从后往前扫一遍，每个位置如果向右能到last_idx,就更新last_idx到当前位置，
    # 最后如果last_idx能到开头，就说明可以走到最后，否则就说明走不到
    res = [False for _ in range(len(arr))]
    res[-1] = True
    n = len(res) - 1
    last_idx = n
    while n >= 0:
        if n + arr[n]>=last_idx:
            last_idx = n
        n -= 1
    return last_idx == 0

def minWeightPath(matrix:List[List]):
    # 一个元素均为非负的矩阵mxn，从左上角走到右下角，每次只能向下或向右，求路径和最小的和
    # 根据行和列谁更小可以通过行或者列来更新，省空间但不省时间
    if len(matrix) == 0:
        return 0
    
    m = len(matrix)
    n = len(matrix[0])

    # 初始化一个和原始矩阵一样大的path矩阵来填数字
    # path = [[None] * n for _ in range(m)]
    # path[0][0] = matrix[0][0]
    # for i in range(1, m):
    #     path[i][0] = path[i - 1][0] + matrix[i][0]
    
    # for j in range(1, n):
    #     path[0][j] = path[0][j - 1] + matrix[0][j]
    
    # for i in range(1, m):
    #     for j in range(1, n):
    #         path[i][j] = min(path[i-1][j], path[i][j-1]) + matrix[i][j]
    # return path[m-1][n-1]
    
    # # 矩阵是一个矮胖的矩阵时(eg: n=100000,m=4)初始化一列来更新
    # temp = [None] * m
    # temp[0] = matrix[0][0]
    # for i in range(1, m):
    #     temp[i] = matrix[i][0] + temp[i-1]
    
    # for j in range(1, n):
    #     for i in range(m):
    #         if i == 0:
    #             # 只累加左方不累加上方
    #             temp[i] = temp[i]+matrix[i][j]
    #         else:
    #             # 累加左方或累加上方，左方从原始temp[i]里面得到,上方从更新过的temp[i-1]里拿到
    #             temp[i] = min(temp[i-1], temp[i]) + matrix[i][j]
    # return temp[-1]

    # 矩阵是一个瘦高的矩阵时(eg: n=4, m=1000000),初始化一行来更新
    temp = [None] * n
    temp[0] = matrix[0][0]
    for j in range(1, n):
    # 初始化第一行
        temp[j] = temp[j-1] + matrix[0][j]
    
    for i in range(1, m):
        for j in range(n):
            if j == 0:
            # 处理第一列时，只累加上方不累加左方
                temp[j] = temp[j] + matrix[i][j]
            else:
            # 处理其他列时，考虑累加上方或累加左方
                temp[j] = min(temp[j-1], temp[j]) + matrix[i][j]
    return temp[-1]

    
def rainDrop(arr):
    # 接雨水问题：给一个arr，每个元素非负，整个数组构成的直方图为容器形状，问容器可以装多少的水，单独考虑每个位置左右两端的最大值，较小的那边和当前的高度决定当前位置可以装多少水，遍历这个arr把每个位置水量加起来
    # 就是最后的结果，可以初始化两个辅助数组计算每个点左和右的最值，也可以用有限个变量记录左右边界和最值，根据左右哪边的最值更小，可以从右往左或从左往右更新

    n = len(arr)
    l2r_preMax = [0] * n
    r2l_preMax = [0] * n
    l2r_preMax[0] = arr[0]
    r2l_preMax[-1] = arr[-1]
    
    for i in range(1, n):
        l2r_preMax[i] = max(l2r_preMax[i-1], arr[i])
    for i in range(n-2, -1, -1):
        r2l_preMax[i] = max(r2l_preMax[i+1], arr[i])
    
    res = [0] * n
    for i in range(n):
        res[i] = min(l2r_preMax[i], r2l_preMax[i]) - arr[i]

    return sum(res)

def rainDrop2(arr):
    # 接雨水问题：双指针解法
    res = 0
    lMax,rMax = arr[0], arr[-1]
    l, r = 1, len(arr)-2
    while l <= r:
        if rMax < lMax:
            # 若rMax更小，说明右侧指针位置可以结算了
           res += max(rMax - arr[r], 0)
           rMax = max(arr[r], rMax)
           r -= 1
        else:
            # 若lMax更小，说明左侧指针位置可以结算了
            res += max(lMax - arr[l], 0)
            lMax = max(arr[l], lMax)
            l += 1
    return res

def convert_digits( input_string, start_position, end_position ) :
	n = len(input_string)
	if (start_position < 1) or (end_position > n) or (start_position > end_position):
		return "INVALID"
	new_string = input_string[:start_position-1]
	digit_mapping = {
		'0': 'ZERO',
		'1': 'ONE',
		'2': 'TWO',
		'3': 'THREE',
		'4': 'FOUR',
		'5': 'FIVE',	
		'6': 'SIX',
		'7': 'SEVEN',
		'8': 'EIGHT',
		'9': 'NINE'
	}
	
	
	
	for index in range(start_position-1, end_position):
		if input_string[index].isdigit():
			mapped = digit_mapping[input_string[index]]
			new_string += mapped
		else:
			new_string += input_string[index]
		
	new_string += input_string[end_position:]
	return new_string

def fastPow(a, n):
    # 快速幂，把指数变为2进制一位一位地乘
    # 10^75 == 10 ^ (1001011)[2]
    # == 10^(2^0) * 10 ^(2^1) * 10^(2^3) * 10^(2^6)
    ans = 1
    while n > 0:
        if (n & 1) == 1:
            ans *= a
        a *= a # base 每次变成当前自己的平方
        n >>= 1
    return ans

def fastPowMatrix(A, n):
    pass


def qsort(arr):
    return arr if len(arr) < 2 else qsort([x for x in arr[1:] if x < arr[0]]) + [arr[0]] + qsort([x for x in arr[1:] if x > arr[0]])


def minLight(arr):
    '''
    路灯问题，X不能放灯，.位置可以放灯且必须被照亮，每个灯可以照亮当前位置，前一个位置和后一个位置
    返回最小路灯数
    '''
    light = 0
    i = 0
    while i < len(arr):
        if arr[i] == 'X':
            i += 1
        else:
            light += 1
            if i + 1 == len(arr):
                break
            else:
                if arr[i + 1] == 'X':
                    i += 2
                else:
                    i += 3
    return light


def PreInPost(pre_, in_):
    def helper(_pre, _in, _post, prei, prej, ini, inj, posti, postj):
        if prei > prej:
            return
        if prei == prej:
            _post[posti] = _pre[prei]
            return

        find = ini
        for find in range(ini, inj+1):
            if _in[find] == _pre[prei]:
                break
        
        # 完善一下这个条件
        helper(_pre, _in, _post, 0, 0, 0, 0, 0, 0)
        helper(_pre, _in, _post, 0, 0, 0, 0, 0, 0)
    
    post_ = [None for _ in pre_]
    n = len(post_)
    helper(pre_, in_, post_, 0, n-1, 0, n-1, 0, n-1)
    return post_

def CompleteBinaryTreeNode(head: BinaryTree):
    # O(h^2) h~Log(n)
    def mostLeftLevel(treeNode: BinaryTree, level: int):
        # 以treeNode为头的树在整棵树的第level层，求这棵树的最大深度
        while head:
            level += 1
            treeNode = treeNode.left
        return level - 1


    def helper(node: BinaryTree, level: int, h: int):
        # level: 当前层数, 整棵树根节点为第一层
        # h: 树整体高度
        # 返回以node为头的完全二叉树节点个数是多少

        if level == h:
            return 1
        
        if mostLeftLevel(node.right, level + 1) == h: # 左树是满的，下一个递归从右树根开始
            return (1 << (h - level)) + helper(node.right, level + 1, h)
        else: # mostLeftLevel(node.right, level + 1) != h # 右树是满的，
            return (1 << (h - level - 1)) + helper(node.left, level + 1, h)
        
    
    if not head:
        return 0
    hglobal = mostLeftLevel(head, 1)
    return helper(head, 1, hglobal)


def longestIncreasingSubSequence(arr):
    # 最长递增子序列
    dp = [1 for _ in arr] # dp[i] 代表子序列必须以arr[i]结尾的情况下最长的子序列长度，初始化为1
    for i in range(len(arr)):
        for j in range(i):
            if arr[j] < arr[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)


def numsNotInArr(arr):
    def modify(arr, num):
        while arr[num - 1] != num:
            arr[num - 1], num = num, arr[num - 1]
        return

    if len(arr) == 0 or not arr:
        return 

    for num in arr:
        modify(arr, num)
    
    for i in range(len(arr)):
        if arr[i] != i + 1:
            print(i + 1)

def donateYoutuber(start, end, x, y, z):
    def func(x, y, z, curr, end):
        '''
        x, y, z 分别为增加2个人气，人气翻倍，减少两个人气的花费，start 为开始时的人气，end为要达到的目标人气
        func 返回最小代价, 这个递归跑不完，存在循环递归的问题
        '''
        if curr == end:
            return 0
        p1 = func(x, y, z, curr + 2, end) + x
        p2 = func(x, y, z, curr * 2, end) + y
        p3 = func(x, y, z, curr - 2, end) + z
        return min(p1, min(p2, p3))

    def func2(x, y, z, curr, end, popMax, preCost, costMax):
        '''
        x, y, z 分别为增加2个人气，人气翻倍，减少两个人气的花费，curr 为当前的人气(可变)，end为要达到的目标人气
        popMax为递归过程中人气值可以达到的最大值，costMax为一个花费的常数平凡解，如果过程中的cost超过了
        这个值，则自动返回系统最大作为停止，preCost为进入这个方程时已经花费的代价(可变)

        '''
        if preCost > costMax:
            return float("inf")
        if curr < 0:
            return float("inf")  
        if curr > popMax:
            return float("inf")
        if curr == end:
            return 0

        ret = float("inf")
        p1 = func2(x,y,z, curr + 2, end, popMax, preCost + x, costMax)
        ret = min(p1, ret)
        p2 = func2(x,y,z,curr * 2, end, popMax, preCost + y, costMax)
        ret = min(p2, ret)
        p3 = func2(x,y,z,curr - 2, end, popMax, preCost + z, costMax)
        ret = min(p3, ret)


    return func2(x, y, z, start, end, 2 * end, 0, (end - start) // 2 * x)

def expressionNum(expression:str, target:bool):
    def isValid(expr):
        # 长度必须是奇数
        if (len(expr) & 1) == 0:
            return False
        # 单数位置不能是二元字符
        for i in range(0, len(expr), 2):
            if expr[i] != "0" or expr[i] != "1":
                return False
        # 双数位置不能不是二元字符
        for i in range(1, len(expr), 2):
            if expr[i] not in {"&", "|", "^"}:
                return False
        return True
    
    def p(expr, desired, L, R):
        if L == R:
            if expr[L] == "1":
                return 1 if desired else 0
            if expr[L] == "0":
                return 0 if desired else 1
        
        res = 0
        if desired:
            for i in range(L+1, R, 2):
                if expr[i] == "&":
                    res +=  p(expr, True, L, i - 1) * p(expr, True, i + 1, R)
                elif expr[i] == "|":
                    res += p(expr, True, L, i - 1) * p(expr, True, i + 1, R)
                    res += p(expr, False, L, i - 1) * p(expr, True, i + 1, R)
                    res += p(expr, True, L, i - 1) * p(expr, False, i + 1, R)
                elif expr[i] == "^":
                    res += p(expr, True, L, i - 1) * p(expr, False, i + 1, R)
                    res += p(expr, False, L, i - 1) * p(expr, True, i + 1, R)
        else:
            for i in range(L+1, R, 2):
                if expr[i] == "&":
                    res += p(expr, False, L, i - 1) * p(expr, False, i + 1, R)
                    res += p(expr, False, L, i - 1) * p(expr, True, i + 1, R)
                    res += p(expr, True, L, i - 1) * p(expr, False, i + 1, R)
                elif expr[i] == "|":
                    res +=  p(expr, False, L, i - 1) * p(expr, False, i + 1, R)
                elif expr[i] == "^":
                    res += p(expr, True, L, i - 1) * p(expr, True, i + 1, R)
                    res += p(expr, False, L, i - 1) * p(expr, False, i + 1, R)
        return res

    if not expression or expression == "":
        return 0
    
    if not isValid(expression):
        return 0
    return p(expression, target, 0, len(expression)-1)


def edittingDistance(str1, str2, addCost, delCost, replaceCost):
    m = len(str1)
    n = len(str2)
    # n+1 row m+1 columns
    dp = [[None for _ in range(m+1)] for __ in range(n+1)]
    # dp代表str1前缀长度为i的字符串编辑成str2前缀长度为j的字符串需要的最少cost
    dp[0][0] = 0
    for i in range(1, m+1):
        dp[0][i] = i * addCost
    for j in range(1, n+1):
        dp[j][0] = j * delCost
    
    # i, j 位置：
    # 0...i-2 == 0...j-1, str1前i-1前缀已经凑出了str2前j个，则当前位置cost为i-1位置cost加上删掉i位置的cost deleting i from str1; dp[i-1][j] + del
    # 0...i-1 == 0...j-2, str1前i个前缀凑出了str2前j-1个，add str2 j to str1 position i; dp[i][j-1] + add (我感觉这种情况是凑不出来的)
    # 0...i-1 == 0...j-1, str1前i-1前缀已经凑出了str2前j-1个，且两字符串第i和j位置字符不一样，额外加替换的cost, replace str1 i to str2 j when they are different; dp[i-1][j-1] + replace
    # 0...i-1 == 0...j-1, str1前i-1前缀已经凑出了str2前j-1个，且两字符串第i和j位置字符一样，没额外的cost，copy str1 i to str2 j when they are the same; dp[i-1][j-1] + 0 (or cost of copy if copy has a cost)

    pass


def phoneNumber_naive(n):
    map = {
        0:[4,6],
        1:[6,8],
        2:[7,9],
        3:[4,8],
        4:[0,3,9],
        5:[],
        6:[0,1,7],
        7:[2,6],
        8:[1,3],
        9:[2,4],
    }
    def count_sequences(start_position, num_hops):                  
        if num_hops == 0:                                           
            return 1                                                
                                                                
        num_sequences = 0                                           
        for position in map(start_position):                  
            num_sequences += count_sequences(position, num_hops - 1)
        return num_sequences
    
    count_sequences(0, 10)

def phoneNumber_memo(n):
    map = {
        0:[4,6],
        1:[6,8],
        2:[7,9],
        3:[4,8],
        4:[0,3,9],
        5:[],
        6:[0,1,7],
        7:[2,6],
        8:[1,3],
        9:[2,4],
    }
    def count_sequences(start_position, num_hops):
        cache = {}

        def helper(position, num_hops):
            if (position, num_hops) in cache:
                return cache[ (position, num_hops) ]

            if num_hops == 0:
                return 1

            else:
                num_sequences = 0
                for neighbor in map(position):
                    num_sequences += helper(neighbor, num_hops - 1)
                cache[ (position, num_hops) ] = num_sequences
                return num_sequences

        res = helper(start_position, num_hops)
        return res
    
    count_sequences(0, 10)

def phoneNumber_dp(n):
    map = {
        0:[4,6],
        1:[6,8],
        2:[7,9],
        3:[4,8],
        4:[0,3,9],
        5:[],
        6:[0,1,7],
        7:[2,6],
        8:[1,3],
        9:[2,4],
    }
    def count_sequences(start_position, num_hops):                
        prior_case = [1] * 10                                     
        current_case = [0] * 10                                   
        current_num_hops = 1                                      
                                                                
        while current_num_hops <= num_hops:                       
            current_case = [0] * 10                               
            current_num_hops += 1                                 
                                                                
            for position in range(0, 10):                         
                for neighbor in map(position):              
                    current_case[position] += prior_case[neighbor]
            prior_case = current_case                             
                                                                
        return current_case[start_position]
    
    count_sequences(0, 10)


def subsequenceToK(str1):
    def g(i, N):
        # 以第i号字符开头，长度为length的序列有多少个
        if N == 1:
            return 1
        summ = 0
        for j in range(i+1, 26):
            summ += g(j, N - 1)
        return summ

    def f(N):
        # 总长度为N的序列有多少个
        summ = 0
        for i in range(26):
            # 从以a开头开始加上所有的可能(i=0 =>以a开头,i=1 =>以b开头)
            summ += g(i, N)
        return summ

    l = len(str1)
    summ = 0
    for i in range(1, l):
        summ += f(i)

    # 比第一个字符小的
    first = ord(str1[0]) - ord('a') + 1
    for i in range(1, first):
        summ += g(i, l)
    
    pre = first
    for i in range(1, l):
        cur = ord(str1[i]) - ord('a') + 1
        for j in range(pre+1, cur):
            summ += g(j, l - i)
        pre = cur
    return summ + 1


def maxGap(arr):
    ## 给一个数组，求如果排序后，相邻两个数最大的差值，要求时间复杂度O(N)，要求不能用非基于比较的排序

    def bucket(num, length, mmin, mmax):
        # 返回这个值在哪个桶里
        return int((num - mmin) / (mmax - mmin) * length)
    l = len(arr)
    mmin = float("inf")
    mmax = float("-inf")
    # 找arr中最小和最大
    for i in range(l):
        mmin = min(mmin, arr[i])
        mmax = max(mmax, arr[i])

    # 最小和最大相等，则差为0
    if mmin == mmax:
        return 0
    
    # 初始化三个数组，代表桶，hasNum代表这个桶里面有没有进过数字
    # maxs代表每个桶里进去过的最大数字
    # mins代表每个桶里进去过的最小数字
    # 桶的数量永远比arr长度多一个，保证一定有空桶
    # 每个桶里面数字范围是一个固定的，这个范围根据整体数组中最大最小值的距离来决定，则这种设定可以保证最优解出自于相邻桶间的差而不是出自于桶内最大最小值的差
    hasNum = [False] * (l + 1)
    maxs = [0] * (l + 1)
    mins = [0] * (l + 1)

    bid = 0
    for i in range(l):
        bid = bucket(arr[i], l, mmin, mmax)
        maxs[bid] = max(maxs[bid], arr[i]) if hasNum[bid] else arr[i]
        mins[bid] = min(mins[bid], arr[i]) if hasNum[bid] else arr[i]
        hasNum[bid] = True

    res = 0
    lastMax = maxs[0]
    for i in range(1, len(maxs)):
        if hasNum[i]:
            res = max(mins[i] - lastMax, res)
            lastMax = maxs[i]
    return res


def mostXOR(arr):
    # 给出N个数字 a_1, a_2到a_N,将其以这些数字为端点任意分割成K个部分，K不确定，问怎么分能使得每个小部分的数字xor和为0，且K最大
    # 返回最大的K
    xor = 0
    # dp[i], arr[0...i]在最优划分的情况下，异或和为0最多的部分有几个

    dp = [0 for _ in range(len(arr))]
    
    # key: 从0位置出发产生的一个异或和
    # value:对应异或和出现的最后一个index是哪里
    # 初始化为0：-1
    mp = {0:-1}
    for i in range(len(arr)):
        # 累加一个异或和
        xor = xor ^ arr[i]
        # 如果这个和存在在map中，那么找到pre index，如果index为-1代表第一次出现0，那么0单独成一个部分
        # 否则从pre+1到现在的i作为一部分异或和为0，dp[i] = dp[pre] + 1
        if xor in mp:
            pre = mp[xor]
            dp[i] = 1 if xor == -1 else dp[pre] + 1
        # 若出现的异或和没在map里，那么说明当前位置元素不是异或和为0的部分的最后一个元素，dp[i] = dp[i-1]
        if i > 0:
            dp[i] = max(dp[i], dp[i-1])
        # 记录当前异或和对应的index
        mp[xor] = i
    return dp[-1]


def josephRing(head: ListNode, m):
    def getLive(i, m):
        # 目前有i个节点，数到m杀死节点，最终活下来的节点，请返回他在有i个节点时的编号
        if i == 1:
            # 目前只有一个节点了，存活节点编号为1
            return 1
        
        return (getLive(i - 1, m ) + m - 1) % i + 1
    if (not head) or head == head.next or m < 1:
        return head
    
    cur = head.next
    size = 1
    while cur != head:
        size += 1
        cur = cur.next
    
    size = getLive(size, m)
    size =- 1
    while size != 0:
        head = head.next
        size -= 1
    head.next = head
    return head


def maxLengthSubArraySumBelowTarget(arr, target):
    # 给一个数组arr，无序，每个值可能正负0，再给另一个正数target，求arr所有子数组中元素和相加小于等于target的最长子数组长度，需要时间复杂度O(N),空间O(1) 
    minSum = [None for _ in range(len(arr))]
    minSumEnd = [0 for _ in range(len(arr))]
    minSum[-1] = arr[-1]
    minSumEnd[-1] = len(arr) - 1
    for i in range(len(arr)-2, -1, -1):
        if minSum[i + 1] < 0:
            minSum[i] = arr[i] + minSum[i + 1]
            minSumEnd[i] = minSumEnd[i+1]
        else:
            minSum[i] = arr[i]
            minSumEnd[i] = i
    
    end = summ = res = 0
    for i in range(len(arr)):
        while end < len(arr) and summ + minSum[end] <= target:
            summ += minSum[end]
            end = minSumEnd[end] + 1
        res = max(res, end - i)
        if end > i:
            summ -= arr[i]
        else:
            end = i + 1
    
    return res

def nimWinner(arr, firsthand):
    # 组合博弈nim问题，用arr中所有的值的异或和做决定，先后手面对异或和为0的状态时会输，否则会赢
    if not arr or len(arr) == 0:
        raise Exception()
    xor0 = 0
    for item in arr:
        xor0 ^= item
    if (firsthand and xor0 != 0) or (not firsthand and xor0 == 0):
        return True
    else:
        return False
    

def snakeRecursive(matrix):
    # 贪吃蛇问题：给定一个二维数组matrix,每个单元格是一个整数，有正有负，最开始的时候玩家操纵一条长度为0的蛇从矩阵最左侧任选一个单元格进入地图，蛇没只能够到达当前位置右上，右和右下相邻
    # 的单元格，蛇到达一个单元格后，自身长度可以瞬间加上当前单元格的值，任何情况下，蛇长度为负游戏结束，玩家可以在过程中把任意一个单元格的值变为相反数，只能变这一个单元格一次，蛇也可以选择停止并结束游戏
    # 问在游戏过程中蛇的长度最多能到多长
    # O(mn)时间，matrix.shape = m, n
    class Info:
        def __init__(self, yes, no):
            self.yes = yes
            self.no = no
            return
    
    def process(matrix, row, col):
        if col == 0:
            return Info(matrix[row][col], -matrix[row][col])

        preYes = preNo = -1
        if row > 0:
            leftUpInfo = process(matrix, row - 1, col - 1)
            if leftUpInfo.no >= 0:
                preNo = leftUpInfo.no
            if leftUpInfo.yes >= 0:
                preYes = leftUpInfo.yes
        leftInfo = process(matrix, row, col - 1)
        if leftInfo.no >= 0:
            preNo = max(preNo, leftInfo.no)
        if leftInfo.yes >= 0:
            preYes = max(preYes, leftInfo.yes)

        if row < len(matrix) - 1:
            leftDnInfo = process(matrix, row + 1, col - 1)
            if leftDnInfo.no >= 0:
                preNo = max(preNo, leftDnInfo.no)
            if leftDnInfo.yes >= 0:
                preYes = max(preYes, leftDnInfo.yes)


        yes = no = -1
        yes = max(preYes + matrix[row][col], no - matrix[row][col]) if preYes != -1 else preYes
        no = preNo + matrix[row][col] if preNo != -1 else preNo

        return Info(yes, no)
    
    def process2(matrix, row, col, memo):
        if col == 0:
            memo[row][col] = Info(matrix[row][col], -matrix[row][col])
            return memo[row][col]

        preYes = preNo = -1
        if row > 0:
            leftUpInfo = process2(matrix, row - 1, col - 1, memo)
            if leftUpInfo.no >= 0:
                preNo = leftUpInfo.no
            if leftUpInfo.yes >= 0:
                preYes = leftUpInfo.yes
        leftInfo = process2(matrix, row, col - 1, memo)
        if leftInfo.no >= 0:
            preNo = max(preNo, leftInfo.no)
        if leftInfo.yes >= 0:
            preYes = max(preYes, leftInfo.yes)

        if row < len(matrix) - 1:
            leftDnInfo = process2(matrix, row + 1, col - 1, memo)
            if leftDnInfo.no >= 0:
                preNo = max(preNo, leftDnInfo.no)
            if leftDnInfo.yes >= 0:
                preYes = max(preYes, leftDnInfo.yes)


        yes = no = -1
        yes = max(preYes + matrix[row][col], no - matrix[row][col]) if preYes != -1 else preYes
        no = preNo + matrix[row][col] if preNo != -1 else preNo

        memo[row][col] = Info(yes, no)
        return memo[row][col]

        
    res = 0
    memo = [[None for _ in range(len(matrix[0]))] for __ in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            # info = process1(matrix, i, j)
            info = process2(matrix, i, j, memo) # 记忆化搜索
            res = max(res, max(info.yes, info.no))
    return res

def getFormulaValue(str_):
    # 给定一个字符串str，str表示一个公式，公式里可以有整数，加减乘除和左右括号，返回公式计算结果(假设给的str是正确的公式，负数必须用括号括起来，除非负数在公式一开头或括号一开头)
    def value(str_, i):
        # 从str[i]往下算，遇到字符串终止位置或右括号就停止，返回两个值tuple
        # 第一个值代表这一段的计算结果是多少，第二个值代表负责的这一段计算到了哪个位置
        from collections import deque
        dq = deque([])
        num = 0
        bra = None
        while i < len(str_) and str_[i] != ')':
            if str_[i] >= "0" and str_[i] <= "9":
                num = num * 10 + (int(str_[i]))
                i += 1
            elif str_[i] != "(":
                addNum(dq, num)
                dq.append(str_[i])
                i += 1
                num = 0
            else:
                bra = value(str_, i + 1)
                num, _ = bra
                i = _ + 1
        
        addNum(dq, num)
        return getNum(num), i


    def addNum(dq: deque, num):
        if dq:
            cur = 0
            top = dq.pop()
            if top in {"+", "-"}:
                dq.append(top)
            else:
                cur = int(dq.pop())
                num = cur * num if top == "*" else cur / num
        dq.append(str(num))

    def getNum(dq:deque):
        # dq里面只有数字和加减号，返回计算完的表达式的值
        res = 0
        add = True
        cur = None
        num = 0
        while dq:
            cur = dq.popleft()
            if cur == "+":
                add = True
            elif cur == "-":
                add = False
            else:
                num = int(cur)
                res += num if add else -num
        return res

    # main
    value(str_, 0)[0]

def lenLongestCommonSubStr(str1, str2):
    # 给定两个字符串str1和str2，求两个字符串的最长公共子串（子串是连续的）dp[i][j]:str1子串必须以i结尾，str2子串必须以j结尾时最长公共子串长度，且最长公共子串也必须以str1[i],str2[j]结尾
    # O(mn)时间
    res = 0
    # 原版动态规划，O(mn)空间
    dp = [[None for _ in range(len(str2))] for __ in range(len(str1))]
    # dp[i][j], i-> str1, j -> str2

    for i in range(len(str1)):
        if str2[0] == str1[i]:
            dp[i][0] = 1
        else:
            dp[i][0] = 0
        res = max(res, dp[i][0])
    
    for j in range(len(str2)):
        if str1[0] == str2[j]:
            dp[0][j] = 1
        else:
            dp[0][j] = 0
        res = max(res, dp[0][j])
    
    for i in range(1, len(str1)):
        for j in range(1, len(str2)):
            if str1[i] == str2[j]:
                dp[i][j] = 1 + dp[i-1][j-1]
            else:
                dp[i][j] = 0
            res = max(res, dp[i][j])
    # 斜率优化版，o(1)空间，因为只依赖当前值和左上角的值，从右上角开始计算，每次算一个从左上到右下的一条斜线
    return res


def lenLongestCommonSubSeq(str1, str2):
    # 给定两个字符串str1和str2，求两个字符串的最长公共子序列（子序列可以是不连续的）dp[i][j]:str1子序列到达i位置，str2子序列到达j位置时最长公共子序列长度，最长公共子序列不一定以str1[i],str2[j]结尾
    # O(mn)时间
    res = 0
    # 原版动态规划，O(mn)空间
    dp = [[None for _ in range(len(str2))] for __ in range(len(str1))]
    # dp[i][j], i-> str1, j -> str2

    for i in range(len(str1)):
        dp[i][0] = dp[i-1][0]
        if str2[0] == str1[i]:
            dp[i][0] = dp[i-1][0] + 1
    
    for j in range(len(str2)):
        dp[0][j] = dp[0][j-1]
        if str1[0] == str2[j]:
            dp[0][j] = 1 + dp[0][j-1]
            
    
    for i in range(1, len(str1)):
        for j in range(1, len(str2)):
            # 四种可能：最长公共子序列，1.不以i,j结尾 2.以i不以j结尾， 3.不以i以j结尾 4.以i且以j结尾(需要str1[i]==str2[j])
            # 求最大值
            dp[i][j] = max(max(dp[i-1][j], dp[i][j-1]), dp[i-1][j-1])
            if str1[i] == str2[j]:
                dp[i][j] = max(dp[i][j], 1 + dp[i-1][j-1])
            
                
    # 斜率优化版，o(1)空间，因为只依赖当前值和左上角的值，从右上角开始计算，每次算一个从左上到右下的一条斜线
    return dp[-1][-1]


def numofBoats(arr, limit):
    if arr is None or len(arr) == 0:
        return 0
    if arr[-1] > limit:
        return -1
    if arr[-1] < limit // 2:
        return (len(arr) + 1) // 2
    sorted(arr)
    lessR = -1
    for i in range(len(arr)):
        if arr[i] <= limit // 2:
            lessR = i
    
    if lessR == -1:
        # 所有元素都比一半的limit更大
        return len(arr)
    L = lessR
    R = lessR + 1
    lessUnUsed = 0
    while L >= 0:
        solved = 0
        while R < len(arr) and arr[L] + arr[R] <= limit:
            R += 1
            solved += 1
        if solved == 0:
            lessUnUsed += 1
            L -= 1
        else:
            # L在有配对的情况下不需要一个一个往左跳，可以直接根据R动了几次来移动L，如果R部分比较长，那么L有可能不够，这时L最终位置是-1代表arr最左侧的左侧
            L = max(-1, L - solved)

    lessAll = lessR + 1
    lessUsed = lessAll - lessUnUsed
    moreUnsolved = len(arr) - lessR - 1- lessUsed
    # 可以双指针头尾往中间走(复杂度在O(N)的级别稍微高一些，但整体复杂度还是O(NLOGN))
    # 也可以用下面的方法：
    # 找小于等于limit/2最右位置L，若L是arr最左边再往左，则每个元素一条船，若返回最右的右边为末尾，则返回N/2条船
    # 一般情况：两个指针L和R=L+1,看arr[L]+arr[R]和limit的关系，可以分成三部分：a.配好对的，b.右侧太大只能一个人一个船的，和c.左侧剩下的
    # 最后结果为a+(b+1)//2+c
    return lessUsed + ((lessUnUsed + 1) >> 1) + moreUnsolved


def palindromParts(strs):
    def valid(strs, i, j):
        # 正常实现判断回文是O(N)复杂度
        # 或者直接预处理一张表
        pass

    # str[i...]最少能分割成多少回文的部分，返回最少部分数
    # 
    def process(strs, i):
        if i == len(strs)-1:
            return 0
        ans = float("inf")
        # i作为起点，尝试后续每一个终点，如果str[i...end]部分是回文，就去尝试这个部分作为第一块回文的整体回文部分数答案
        for end in range(i, len(strs)):
            if valid(strs, i, end):
                # str[i...]拆成第一块为str[i...end],后续是str[end+1...]怎么拆最省的问题
                ans = min(ans, 1 + process(strs, end + 1))
        return ans

    if strs is None or len(strs) == 0:
        return 0
    
    return process(strs, 0)
    

def palindromCuts(strs):
    def record(strs):
        n = len(strs)
        records = [[None for _ in range(n)] for __ in range(n)]
        records[n-1][n-1] = True
        for i in range(n):
            records[i][i] = True
            records[i][i+1] = strs[i] == strs[i+1]
        for row in range(n-3, -1, -1):
            for col in range(row+2,n):
                records[row][col] = strs[row] == strs[col] and records[row+1][col-1]
        return records

    if strs is None or len(strs) == 0:
        return 0
    if len(strs) == 1:
        return 0
    n = len(strs)
    dp = [None for _ in range(n + 1)]
    # str[i...]最少切几次可以让每个部分都是回文
    dp[n] = 0
    dp[n-1] = 1
    p = record(strs)
    for i in range(n-2, -1, -1):
        # 初始化最大cut数为元素数量
        dp[i] = n - i
        for j in range(i, n):
            if p[i][j]:
                dp[i] = min(dp[i], dp[j+1]+1)
    return dp[0]


def getMinKthByBTPRT(arr, k):
    def swap(arr, i, j):
        arr[i], arr[j] = arr[j], arr[i]
        return

    def getMedian(arr, s, e):
        # 每次median只有5个元素或更少，排序复杂度认为是常数 #这里有额外的空间复杂度
        sArr = sorted(arr[s:e+1])
        return sArr[len(sArr) // 2]
        
    def medianofMedians(arr, begin, end):
        num = end - begin + 1
        offset = 0 if num % 5 == 0 else 1
        mArr = [None for _ in range(num // 5 + offset)]
        for i in range(len(mArr)):
            beginI = begin + i * 5
            endI = beginI + 4
            mArr[i] = getMedian(arr, beginI, min(end, endI))
        return select(mArr, 0, len(mArr)-1, len(mArr) // 2)
    
    def partition(arr, begin, end, pivot):
        less, more = begin - 1, end+1
        curr = begin
        while curr != more:
            if arr[curr] < pivot:
                swap(arr, less + 1, curr)
                curr += 1
                less += 1
            elif arr[curr] > pivot:
                swap(arr, more - 1, curr)
                more -= 1
            else:
                curr += 1
        return less+1, more-1
    


    def select(arr, begin, end, i):
        # 在arr[begin...end]范围上假设排序的话,i位置的数是谁
        if begin == end:
            return arr[begin]
        # 有讲究地选一个pivot
        pivot = medianofMedians(arr, begin, end)
        lU, gL = partition(arr, begin, end, pivot)
        if i >= lU and i <= gL:
            return arr[i]
        elif i < lU:
            return select(arr, begin, lU-1, i)
        else:
            return select(arr, gL+1, end, i)
    
    

    return select(arr, 0, len(arr)-1, k-1)
    

def wayToSplitN(N):
    # 给定一个正数N，返回裂开这个数的方法数，裂开的每个数最小为1，假设N裂开成三个不同的数，要求第二个数不小于第一个数且第三个数不小于第二个数
    # dp[pre][rest]: 可以分析出从下往上，从左往右地填表
    # 可斜率优化：临近位置可以代替第三个循环的枚举行为
    def process(pre, rest):
        if rest == 0:
            return 1
        if pre > rest:
            return 0
        ways = 0
        for i in range(pre, rest + 1, 1):
            ways += process(i, rest - 1)
        return ways

    def dp(N):
        if N < 1:
            return 0
        dp = [[0 for _ in range(N+1)] for __ in range(N+1)]
        # dp[pre][rest]
        for pre in range(1, N+1):
            dp[pre][0] = 1

        for pre in range(N, 0, -1):
            for rest in range(pre, N+1, 1):
                for i in range(pre, rest+1, 1):
                    dp[pre][rest] += dp[i][rest-i]
        return dp[1][N]

    def dp2(N):
        # 2 是对 1 的斜率优化，通过观察所有枚举的值和现在表中存在值的依赖关系来把枚举的和换成了表中已有的值
        if N < 1:
            return 0
        dp = [[0 for _ in range(N+1)] for __ in range(N+1)]
        # dp[pre][rest]
        for pre in range(1, N+1):
            dp[pre][0] = 1

        # 注意这里边界发生了微小的变化
        for pre in range(N-1, 0, -1):
            for rest in range(pre+1, N+1, 1):
                dp[pre][rest] + dp[pre+1][rest] + dp[pre][rest-pre]
        return dp[1][N]

    if N < 1:
        return 0
    if N == 1:
        return 1

    return process(1, N)


def maxConnectedBSTTopoSize(head: Node):
    class Record:
        def __init__(self, l, r):
            self.l = l
            self.r = r

    def maxTopo(h:Node,n:Node):
        if h and n and isBSTNode(h, n, n.value):
            return maxTopo(h, n.left) + maxTopo(h, n.right) + 1
        return 0
    def isBSTNode(h:Node, n:Node, val):
        if h is None:
            return False
        if h == n:
            return True

        return isBSTNode(h.left if h.value > val else h.right, n, val)

    def bstTopoSize(head:Node):
        mp = {}
        return posOrder(head, mp)

    def posOrder(head:Node, mp:Dict):
        if not head:
            return 0

        ls = posOrder(head.left, mp)
        rs = posOrder(head.right, mp)
        modifyMap(head.left, head.value, mp, True)
        modifyMap(head.right, head.value, mp, False)

        lr = mp[head.left] if head.left in mp else None
        rr = mp[head.right] if head.right in mp else None

        lbst = 0 if lr is None else lr.l + lr.r + 1
        rbst = 0 if rr is None else rr.l + rr.r + 1
        mp[head] = Record(lbst, rbst)
        return max(lbst + rbst + 1, max(ls, rs))
    def modifyMap(n: Node, v, mp, tf):
        if n is None or n not in mp:
            return 0
        r = mp[n]
        if (tf and n.value > v) or (not tf and n.value < v):
            del mp[n]
            return r.l + r.r + 1
        else:
            minus = modifyMap(n.right if tf else n.left, v, mp, tf)
            if tf:
                r.r = r.r - minus
            else:
                r.l = r.l - minus
            mp[n, r]
            return minus

    if not head:
        return 0
    mx = maxTopo(head, head)
    mx = max(mx, bstTopoSize(head.left))
    mx = max(mx, bstTopoSize(head.right))
    return mx

def minimumSubStringCoveringPattern(s, t):
    # 最小覆盖子串问题，双指针可解
    need_pattern_map = collections.Counter(t)

    l,r = 0,0
    # pattern里unique的值有几个
    unique_pattern = len(need_pattern_map)
    res = float("inf")
    res_str = ""
    while r < len(s):
        # 双指针r往右走，如果r当前位置的字符在pattern里，那么pattern map对应的值频率减1
        if s[r] in need_pattern_map:
            need_pattern_map[s[r]] -= 1
        if s[r] in need_pattern_map and need_pattern_map[s[r]] == 0:
            # 如果某个字符在pattern map里频率降为0了，那么说明l...r这个范围内这个字符已经满足覆盖的要求了
            # 把独一无二字符数减1
            unique_pattern -= 1
        
        while unique_pattern == 0:
            # 当独一无二字符数变为0时，说明当前l...r是一个满足要求的覆盖，记录一下长度和更新一下最短结果
            if r - l + 1 < res:
                res_str = s[l: r+1]
                res = r - l + 1
            # 接下来我们试图让这个覆盖变短
            if s[l] in need_pattern_map:
                # 如果当前左侧字符在pattern map里面，那么会从滑动窗口中滑出去，
                # 我们在需求pattern map中+1，表明需要让r向右动去找到这个字符以形成覆盖
                need_pattern_map[s[l]] += 1
            if need_pattern_map[s[l]] == 1:
                # 当需求map中某个字符需求变为1时，我们增加独一无二字符数的count
                unique_pattern += 1
            l += 1
        r += 1
    return res_str


def perfectShuffle(arr):
    # 完美洗牌问题，len(arr)为偶数
    pass

def strPatternMatchRecursive(s, p):
    def isValid(s, p):
        # 先确认s和p字符串本身没有问题，*前面必须是非*,s里只能有[a-z]
        for item in s:
            if item == "*" or item == "?":
                return False
        for i,item in enumerate(p):
            if item == "*" and (i == 0 or p[i-1] == "*"):
                return False

    def process(s, p, si, pi):
        # s[si...]能不能被p[pi...]匹配，这里必须保证pi压中的不是*这样可以保证当前位置不受前面位置东西的影响
        # 0位置在之前的isValid函数已经查过了
        if pi == len(p):
            return si == len(s)
        # ei+1位置不是*
        if (pi+1 == len(p)) and (p[pi+1] != "*"):
            # 没有pi+1位置时以及有pi+1位置但是下一个位置不是*
            return si != len(s) and (p[pi] == s[si] or p[pi] == "?") and process(s, p, si+1, pi+1)
        # ei+1位置是*
        while si!=len(s) and (s[si]==p[pi] or p[pi]=="?"):
            # 如果没hit到while循环，说明带*的这个pattern没法match这部分s，那么直接跳过这个pattern，等同于让*这部分变为“”，然后继续下面的
            # match
            if process(s, p, si, pi+2):
                # 再尝试一次0字符配后续
                return True
            #零字符走不通那么至少当前的s[si]可以匹配上一个p[pi],由于pi+1是*，后续可能有更多的si和p[pi]匹配，这里直接移动si
            si += 1

        # 如果没hit到while循环，说明带*的这个pattern没法match si当前的东西，那么直接跳过这个pattern，等同于让*这部分变为“”，然后继续下面的
        # match
        return process(s, p, si, pi+2)


    
    if s is None or p is None:
        return False
    return process(s, p, 0, 0) if isValid(s, p) else False

def strPatternMatchDp(s, p):
    def isValid(s, p):
        # 先确认s和p字符串本身没有问题，*前面必须是非*,s里只能有[a-z]
        for item in s:
            if item == "*" or item == "?":
                return False
        for i,item in enumerate(p):
            if item == "*" and (i == 0 or p[i-1] == "*"):
                return False

    if s is None or p is None:
        return False
    if not isValid(s, p):
        return False
    sl, pl = len(s), len(p)
    dp = [[False for col in range(pl+1)] for row in range(sl+1)]

    # 填最后一列
    for row in range(sl):
        dp[row][-1] = True if row == sl - 1 else False

        # 这里倒数第二列和最后一行要根据题目定义来填而不能直接用递归式来填
        # 填倒数第二列

    # 填最后一行: “{anychar}*"可以变为空串，否则全是false
    dp[-1][-1] = True
    dp[-1][-2] = False
    for col in range(pl-2, -1, -2):
        dp[-1][col] = True if p[col] != "*" and p[col+1] == "*" else False
    
    if sl > 0 and pl > 0:
        if p[pl-1] == "?" or s[sl-1] == p[pl-1]:
            dp[sl-1][pl-1] = True

    for row in range(sl, -1, -1):
        for col in range(pl-2, -1 -1):
            if p[col+1] != "*":
                dp[row][col] = (p[col] == s[row] or s[row] == "?") and dp[row+1][col+1]
            else:
                si = row
                while si != sl and (p[col] == s[si] or p[col] == "?") :
                    if dp[si][col+2]:
                        dp[row][col] = True
                        break
                    si += 1
                if not dp[row][col]:
                    dp[row][col] = dp[si][col+2]


def maxXORSubArray(arr):
    def process1(arr):
        if arr is None or len(arr) == 0:
            return 0

        ans = float("-inf")
        for i in range(len(arr)):
            for j in range(0, i):
                cur = 0
                for ele in arr[j, i+1]:
                    cur ^= ele
                ans = min(cur, ans)
        return ans
    def process2(arr):
        if arr is None or len(arr) == 0:
            return 0
        preSum = [0 for ele in len(arr)]
        preSum[0] = arr[0]
        for i in range(1, len(arr)):
            preSum[i] = preSum[i-1] ^ arr[i]

        ans = float("-inf")
        for i in range(len(arr)):
            for j in range(0, i):
                cur = preSum[j] ^ (preSum[i] if j-1 == -1 else 0)
                ans = min(cur, ans)
        return ans

    def process3(arr):
        # 前缀树方法，可以去掉内层循环
        class NodeTrie:
            def __init__(self):
                self.nexts = [None, None]
        class NumTrie:
            def __init__(self):
                self.head = NodeTrie()
            
            def add(self, num):
                cur = self.head
                for mv in range(31, -1, -1):
                    path = (num >> mv) & 1 # 取出第mv位的0或1状态
                    cur.nexts[path] = NodeTrie() if cur.nexts[path] is None else cur.nexts[path]
                    cur = cur.nexts[path]

            def maxXor(self, sm):
                # 沿着前缀树从高位到低位走一遍，试图让sm与前缀树产生的数的异或和最大，就等同于试图让最高位符号位变成正且剩下的位凑出1
                # 符号位：1=>负数，要尽量凑成0即走1的路
                cur = self.head
                res = 0
                for mv in range(31, -1, -1):
                    path = (sm >> mv) & 1 # 取出第mv位的0或1状态
                    best = path if mv == 31 else path ^ 1 # best为凑成最大异或应该走的路
                    best = best if cur.nexts[best] is not None else (best ^ 1) # 应该走的路在用来组合的前缀和中不一定有，那么这里更新成实际可以走的路

                    # 给最终结果加上这一位实际二进制位 （path来自要组合XOR的数字）,best为当前位在当前前缀树中最合理的二进制组合位，
                    # 异或后为存在于当前前缀树的最佳选择，左移mv位还愿当前二进制在哪一位，然后和res取或表明加在res上面
                    res |= (path ^ best) << mv
                    cur = cur.nexts[best]


        if arr is None or len(arr) == 0:
            return 0
        mx = float("inf")
        sm = 0
        nTri = NumTrie()
        nTri.add(0)

        for ele in arr:
            sm ^= ele
            mx = max(mx, nTri.maxXor(sm))
            nTri.add(sm)
        return mx


    return process3(arr)


def balloonShoot(arr):
    def process(ar, l, r):
        if l == r:
            return ar[l] * ar[l-1] * ar[r+1]

        mx = max(
            ar[l-1] * ar[l] * ar[r+1] + process(ar, l+1,r),
            ar[l-1] * ar[r] * ar[r+1] + process(ar, l, r-1)
        )
        for loc in (l+1, r):
            mx = max(mx, ar[loc] * ar[r+1] * ar[l-1] + process(ar, l+1, loc-1) + process(ar, loc+1, r-1))
        return mx

    if arr is None:
        return 0
    N = len(arr)
    if N== 0:
        return 0
    if N == 1:
        return arr[1]
    help = [None for ele in range(N+2)]
    help[0], help[-1] = 1
    for i,ele in enumerate(arr):
        help[i+1] = ele

    return process(help, 1, N)
    

def HanoiStatus(arr):
    def process(arr, i, from_, other, to_):
        # 把[0...i]的圆盘从from挪到to上去
        # 分成三步：1.把[0...i-1]圆盘从from挪到other上， 2.把i盘从from挪到to上 3.把[0...i-1]盘从other挪到to上

        if i == -1: # base case代表递归完所有的盘了
            return 0
        if arr[i] == other: # 三步没有任何一步中i盘到了other，所以这种状态不是最优状态中的一步
            return -1

        if arr[i] == from_: # 第一步没走完
            return process(arr, i - 1, from_, to_, other)
        else:
            # 第三步
            rest = process(arr, i - 1, other, from_, to_)
            if rest == -1:
                return -1
            return (1 << i) + rest # 这里已经加了挪最大圆盘的那一步

    if arr is None or len(arr) == 0:
        return -1
    return process(arr, len(arr)-1, 1, 3, 2)

def isScrambleStr(str1, str2):
    # 给两个字符串str1和str2，请返回这两个串是否互为旋变串
    def sameFreq(str1, str2):
        from collections import Counter
        c1 = Counter(str1)
        c2 = Counter(str2)
        for k, v in c1.items():
            if k not in c2 or c2[k] != v:
                return False
        return True

    def method1(str1, str2):
        def process(str1, str2, L1, L2, K):
            if K == 1:
                return str1[L1] == str2[L2]
            for k in range(1, K):
                # 第一个是左对左右对右，第二个是左对右右对左
                if (process(str1, str2, L1, L2, k) and process(str1, str2, L1+k, L2+k, K-k)) or \
                        (process(str1, str2, L1, L2+(K-k), k) and process(str1, str2, L1+k, L2, K-k)):
                    return True
            return False
        if not str1 or not str2:
            return False
        if len(str1) != len(str2):
            return False
        if not sameFreq(str1, str2):
            return False
        if str1 == str2:
            return True

        return process(str1, str2, 0, 0, len(str1))
    def method2(str1, str2):
        if not str1 or not str2:
            return False
        if len(str1) != len(str2):
            return False
        if not sameFreq(str1, str2):
            return False
        if str1 == str2:
            return True
        N = len(str1)
        dp = [[[False for _ in range(N)] for __ in range(N)] for ___ in range(N + 1)]
        # dp[l1][l2][k]: str1从下标l1开始，str2从下标l2开始，长度为k的串是否互为旋变串

        # 填好k第一层
        for l1 in range(N):
            for l2 in range(N):
                dp[l1][l2][1] = str1[l1] == str2[l2]
        # 根据递归关系填好上面每一层
        for k in range(2, N+1):
            for l1 in range(N-k+1):
                for l2 in range(N-k+1):
                    for runk in range(1, k):
                        if (dp[l1][l2][runk] and dp[l1+runk][l2+runk][k-runk]) or (dp[l1+runk][l2][k-runk] and dp[l1][l2+k-runk][k]):
                            dp[l1][l2][k] = True
                            break
        return dp[0][0][N]


def cycleGasStation(oil, distance):

    # 用油数组来放net油数组，元素为加的油去掉从这个加油站出发到下一个油站的距离
    # 小于0的位置直接是非法出发点
    for i in range(len(oil)):
        oil[i] -= distance[i]



###### 每日练手
def lowestCommonAncestorCheck(root, n1, n2):
    if root == n1 or root == n2 or (not root):
        return root
    left = lowestCommonAncestorCheck(root.left, n1, n2)
    right = lowestCommonAncestorCheck(root.right, n1, n2)

    if left is not None and right is not None:
        return root
    
    return left if left else right

def treeLevelOrderIterativeCheck(root: BinaryTree):
    if not root:
        return

    from collections import deque
    q = deque([root])
    while q:
        cur = q.popleft()
        print(cur.value)
        if cur.left:
            q.append(cur.left)
        if cur.right:
            q.append(cur.right)
    return

def treePreOrderIterativeCheck(root: BinaryTree):
    if not root:
        return

    from collections import deque
    s = deque([root])
    while s:
        cur = s.pop()
        print(cur.value)
        if cur.right:
            s.append(cur.right)
        if cur.left:
            s.append(cur.left)

    return

def treeInOrderIterativeCheck(root:BinaryTree):
    if not root:
        return

    from collections import deque
    s = deque([])
    while root:
        s.append(root)
        root = root.left

    while s:
        cur = s.pop()
        print(cur.value)
        cur_R = cur.right
        while cur_R:
            s.append(cur_R)
            cur_R = cur_R.left
    return

def treePostOrderIterativeCheck(root: BinaryTree):
    if not root:
        return
    
    from collections import deque
    c = deque([root])
    p = deque([])
    while c:
        cur = c.pop()
        p.append(cur)
        if cur.left:
            c.append(cur.left)
        if cur.right:
            c.append(cur.right)
    while p:
        print(p.pop().value)
    return

def reverseListCheck(head:ListNode):
    if not head:
        return

    prev_node = None
    curr_node = head
    while curr_node:
        next_node = curr_node.next
        curr_node.next = prev_node
        prev_node = curr_node
        curr_node = next_node
    return prev_node

def dfsCheck(node: Node):
    def helper(nd: Node, st:set):
        print(nd.value)
        st.add(nd)
        for neighbor in nd.nexts:
            if neighbor not in st:
                helper(neighbor, st)
        return
    
    if not node:
        return

    visited = set()
    helper(node, visited)

def bfsCheck(node: Node):
    if not node:
        return

    from collections import deque
    q = deque([node])

    visited = set()
    while q:
        cur = q.popleft()
        print(cur.value)
        visited.add(cur)
        for nb in cur.nexts:
            if nb not in visited:
                q.append(nb)
    return

def qsortCheck(arr):
    return arr if len(arr) < 2 else qsortCheck([x for x in arr[1:] if x < arr[0]]) + [arr[0]] + qsortCheck([x for x in arr[1:] if x > arr[0]])

def quickSortCheck(arr):
    def swap(ar, i, j):
        ar[i], ar[j] = ar[j], ar[i]
        return

    def partition(arr, l, r):
        if l == r:
            return l, r
        if l > r:
            return -1, -1

        less, curr, more = l-1, l, r
        pivot = arr[r]

        while curr < more:
            if arr[curr] < pivot:
                swap(arr, less+1, curr)
                less += 1
                curr += 1
            elif arr[curr] > pivot:
                swap(arr, more - 1, curr)
                more -= 1
            else:
                curr+=1
        
        swap(arr, r, more)
        return less+1, more

    def process(arr, l, r):
        if l >=r:
            return
        rand_ind = random.randint(l, r-1)
        swap(arr, rand_ind, r)
        L, U = partition(arr, l, r)
        process(arr, l, L-1)
        process(arr, U+1, r)
        return
    if len(arr) <2:
        return
    process(arr, 0, len(arr)-1)
    return

def mergeSortCheck(arr):
    def merge(arr, l, m, r):
        n = r-l+1
        temp = [None for _ in range(n)]
        p1,p2,i = l, m+1, 0
        while p1 <= m and p2 <= r:
            if arr[p1] < arr[p2]:
                temp[i] = arr[p1]
                p1 += 1
            else:
                temp[i] = arr[p2]
                p2 += 1
            i += 1
        while p1 <= m:
            temp[i] = arr[p1]
            i+=1
            p1+=1
        while p2 <= r:
            temp[i] = arr[p2]
            i+=1
            p2+=1
        # 放回原数组
        for i in range(n):
            arr[l+i] = temp[i]
        return

    def process(arr, l , r):
        if l == r:
            return

        m = l + ((r-l) >> 1)
        process(arr, l, m)
        process(arr, m+1, r)
        merge(arr, l, m, r)
        return
    
    if len(arr) < 2:
        return
    process(arr, 0, len(arr) - 1)
    return

def equalizeTeamSize(teamSize, k):
    # [1,2,2,3,4], k=2 -> change 2 times, 3,4 -> 2,2
    # Write your code here
    n = len(teamSize)
    if n == 1:
        return 1
    if k > len(teamSize)-1:
        return n
    # # dp[i][j] allow j reduction at most, can change teamSize[0...i]
    # dp = [[None for _ in range(k)] for __ in range(kn)]
    # teamSize.sort()
    import heapq
    import collections 
    freq = collections.Counter(teamSize)
    team_freq = [-v for k, v in freq.items()]
    heapq.heapify(team_freq)
    num_of_team,remK = 0, k
    
    temp = -heapq.heappop(team_freq)
    while team_freq:
        curr = heapq.heappop(team_freq)
        if remK + curr < 0:
            return num_of_team+remK+temp
        else:
            num_of_team += -curr
            remK += curr
    return num_of_team+temp

def getMinimumCost(cost, k):
    # Write your code here
    # [4,3,9,3,1], k=2 => 3+3+1=7
    # dp 超时
    dp = [float("inf") for _ in range(len(cost))]
    dp[0] = cost[0]
    for i in range(1, len(cost)):
        for j in range(1, k+1):
            if i - j < 0:
                dp[i] = cost[i]
            else:
                dp[i] = min(dp[i], dp[i-j]+cost[i])
    return dp[-1]

    # minCost = float("inf")
    # for i in range(1, k+1):
    #     curCost = 0
    #     curr = -1
    #     while curr + i < len(cost):
    #         curr += i
    #         curCost += cost[curr]
    #     if curr != len(cost)-1:
    #         curCost += cost[-1]
        
    #     minCost = min(curCost, minCost)
    return minCost

def getTransformedLength(word, t):
    pass
    # a = "abcdedfhijklmnopqrstuvwxy"
    # b = "bcdedfhijklmnopqrstuvwxyz"
    # mp = {k:v for k,v in zip(a,b)}
    # mp['z'] = 'ab'
    # # Write your code here
    # new_word = word
    # for i in range(t):
    #     new_word = "".join([mp[item] for item in new_word])
    # return len(new_word) % (100000007)



if __name__ == "__main__":
    grid = [[1,1,1,1,0],[1,1,0,1,0],[1,1,0,0,0],[0,0,0,0,0]]
    print(countIsland(grid))
    # print(getNextArr(""))
    # print("Hi")
    # print(equalizeTeamSize([1,2,2,3,4], 3))

