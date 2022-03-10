from tools.datasets import Tracklet, Cell
import pulp


class Tracker:
    def __init__(self, dataset, motion_estimator, tracker_config):
        self.dataset = dataset
        self.motion_estimator = motion_estimator
        self.config = tracker_config
        self.solver = pulp.GUROBI()

        pass

    def step(self):
        """
        return optimal tracklet given current motion estimation
        :return:
        """
        tracklet_ = Tracklet()
        for t in range(self.dataset.time_len - 1):
            self.step_time(t, tracklet_)

        return tracklet_

    def step_time(self, time, tracklet):
        return self.match(tracklet[time], self.dataset[time])

    def match(self, tracklet_t, data_t_):
        problem = pulp.LpProblem()
        x = self.get_data_link(tracklet_t, data_t_)
        h = self.get_hidden_link(tracklet_t, data_t_)
        d = self.get_death_link(tracklet_t, data_t_)

        s, p = self.get_split_link(tracklet_t, data_t_)

        problem +=

        pass

    def get_data_link(self, tracklet_t, data_t_):
        return [[pulp.LpVariable('x' + str(i) + str(j), lowBound=0, upBound=1)
                 for j in data_t_.len()]
                for i in range(tracklet_t.len())]

    def get_hidden_link(self, tracklet_t, data_t_):
        return [pulp.LpVariable('h' + str(i), lowBound=0) for i in range(tracklet_t.len())]

    def get_death_link(self, tracklet_t, data_t_):
        pass

    def get_split_link(self, tracklet_t, data_t):
        pass
