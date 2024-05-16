class Node:
    def __init__(self, key=None, val=None, next=None, pre=None) -> None:
        self._key = key
        self._val = val
        self._next = next
        self._pre = pre


class LinkedList:
    def __init__(self) -> None:
        self._head = Node()
        self._tail = Node()
        self._head._next = self._tail
        self._tail._pre = self._head

    def insert_to_head(self, node: Node):
        node._next = self._head._next
        node._pre = self._head
        self._head._next._pre = node
        self._head._next = node

    def remove_node(self, node: Node):
        node._pre._next = node._next
        node._next._pre = node._pre
        node._pre = None
        node._next = None

    def swap_to_head(self, node: Node):
        self.remove_node(node)
        self.insert_to_head(node)

    def get_last_node(self):
        return self._tail._pre

    def erase_tail(self):
        ptr = self._tail._pre
        if ptr != self._head:
            self.remove_node(ptr)
        return ptr

    def travel(self):
        res = []
        ptr = self._head._next
        while ptr != self._tail:
            res.append((ptr._key, ptr._val))
            ptr = ptr._next
        return res


class LRURecorder:
    def __init__(self, size=1000) -> None:
        self._list = LinkedList()
        self._key2ptr = {}
        self._size = size

    def insert_record(self, key, val):
        node = Node(key, val)
        self._key2ptr[key] = node
        self._list.insert_to_head(node)
        while len(self._key2ptr) > self._size:
            self.remove_expired_record()

    def remove_expired_record(self):
        ptr = self._list.erase_tail()
        del self._key2ptr[ptr._key]

    def find(self, key):
        ptr = self._key2ptr.get(key, None)
        res = None
        if not (ptr is None):
            res = ptr._val
            self._list.swap_to_head(ptr)
        return res

    def get_cached_datas(self):
        res1 = []
        for key in self._key2ptr:
            res1.append((key, self._key2ptr[key]._val))
        res2 = self._list.travel()
        return res1, res2


if __name__ == "__main__":
    record = LRURecorder(size=3)
    record.insert_record(1, 1)
    record.insert_record(2, 2)
    record.insert_record(3, 3)
    print(record.get_cached_datas())
    record.find(1)
    record.find(2)
    print(record.get_cached_datas())
    record.insert_record(8, 8)
    print(record.get_cached_datas())
