
class Tracklet:
    def __init__(self):
        pass

    pass


class Cell:
    pass


class CellDataSet:
    def __init__(self, data, time_len):
        self.data = data
        self.time_len = time_len
        pass

    def get_time_slice(self, t):
        if t > self.time_len:
            raise IndexError

        else:
            return self.data[t]
