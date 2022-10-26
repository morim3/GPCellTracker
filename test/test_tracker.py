from unittest import TestCase
from models.tracker import Tracker
import numpy as np
from attrdict import AttrDict

from tools.datasets import CellDataSet, Tracklet, CellNode


class TestTracker(TestCase):
    def test_step(self):
        dataset = np.random.randn(5, 20, 2)

        motion_estimation = lambda x: x
        config = AttrDict({"split_pair_max_dist": 0.1,
                           "split_max_dist": 0.1,
                           "data_match_max_dist": 0.1,
                           "data_weight": 1,
                           "hidden_weight": 1,
                           "death_weight": 1,
                           "split_weight": 1})
        tracker = Tracker(CellDataSet(dataset, 5), motion_estimation, config)
        tracklet = Tracklet(5)
        for i in range(20):
            tracklet.push_cell(CellNode("0"+str(i), np.random.randn(2), 0, None), 0)
        result = tracker.match(tracklet[0], dataset[0])
        print(result)


    def test_step_time(self):
        self.fail()

    def test_match(self):
        self.fail()
