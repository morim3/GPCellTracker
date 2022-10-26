import collections


class CellNode:
    def __init__(self, id: str, pos, hidden: int, prev_cell, ):
        self.id = id
        self.pos = pos
        self.hidden = hidden
        self.prev_cell = prev_cell
        self.next = None

        if prev_cell is not None:
            prev_cell.next = self


class Tracklet:
    def __init__(self, time_len):
        self.track = [collections.deque() for _ in range(time_len)]
        pass

    def push_cell(self, cell: CellNode, time):
        self.track[time].append(cell)

    def __getitem__(self, item):
        return self.track[item]


class CellDataSet:
    def __init__(self, data, time_len):
        self.data = data
        self.time_len = time_len
        pass

    def __getitem__(self, item):
        if item > self.time_len:
            raise IndexError

        else:
            data = self.data[item]
            return data
