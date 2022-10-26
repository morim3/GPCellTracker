from typing import Optional, List, Deque, Tuple
import numpy as np
from pulp.pulp import lpSum

from tools.datasets import Tracklet, CellNode
from enum import Enum
import pulp
import scipy.spatial as ss
import collections


class EdgeType(Enum):
    DataMatch = 0
    Hidden = 1
    MitosisTo = 2
    MitosisFrom = 3
    Death = 4


class Edge:
    def __init__(self, prev_node, next_node, lp_val: pulp.LpVariable, kind: EdgeType, weight: float):
        self.prev_node = prev_node
        self.next_node = next_node
        self.lp_val = lp_val
        self.kind = kind
        self.weight = weight


class MitosisNode:
    def __init__(self, to_edges: List[Edge], from_edge: Edge):
        self.to_edges = to_edges
        self.from_edge = from_edge


class Tracker:
    def __init__(self, dataset, motion_estimator, tracker_config):
        self.dataset = dataset
        self.time_len = dataset.time_len
        self.motion_estimator = motion_estimator
        self.config = tracker_config
        self.solver = pulp.GUROBI()

    def init_tracklet(self, tracklet):
        for i, d in enumerate(self.dataset[-1]):
            tracklet.push_cell(CellNode(str(i), d, 0, None,), self.time_len - 1)

    def step(self):
        """
        return optimal tracklet given current motion estimation
        :return:
        """
        tracklet_ = Tracklet(self.time_len)
        self.init_tracklet(tracklet_)

        for t in range(self.time_len - 2, -1, -1):
            self.step_time(t, tracklet_)

        return tracklet_

    def step_time(self, time, tracklet):
        return self.match(tracklet[time + 1], self.dataset[time])

    def match(self, tracklet_t, data_t_):

        problem = pulp.LpProblem()
        data_edges = self.get_data_edges(tracklet_t, data_t_)
        hidden_edges = self.get_hidden_edges(tracklet_t)
        death_edges = self.get_death_edges(data_t_)

        split_into, split_from, mitosis_nodes = self.get_split_edges(tracklet_t, data_t_)

        for i in range(len(tracklet_t)):
            problem += lpSum([e.lp_val for e in data_edges[i] if e is not None]) \
                       + hidden_edges[i].lp_val + lpSum([e.lp_val for d in split_into for e in d]) == 1

        for j in range(len(data_t_)):
            problem += lpSum([data_edges[i][j].lp_val for i in range(len(tracklet_t)) if data_edges[i][j] is not None]) \
                       + death_edges[j].lp_val \
                       + lpSum([e.lp_val for d in split_from for e in d]) == 1

        for e in mitosis_nodes:
            for to in e.to_edges:
                problem += to.lp_val == e.from_edge.lp_val

        problem += lpSum(e.lp_val * e.weight for l in data_edges for e in l if e is not None) \
            + lpSum(e.lp_val * e.weight for e in hidden_edges) \
            + lpSum(e.lp_val * e.weight for e in death_edges) \
            + lpSum(e.lp_val * e.weight for l in split_into for e in l) \
            + lpSum(e.lp_val * e.weight for l in split_from for e in l)

        result = problem.solve(self.solver)
        return result

    def get_data_edges(self, tracklet_t, data_t_) -> List[List[Optional[Edge]]]:

        return [[self.get_edge(tracklet_t[i], data_t_[j], EdgeType.DataMatch, j) for j in range(len(data_t_))]
                for i in range(len(tracklet_t))]

    def get_hidden_edges(self, tracklet_t, ):

        return [self.get_edge(tracklet_t[i], None, EdgeType.Hidden, None) for i in range(len(tracklet_t))]

    def get_death_edges(self, data_t_):
        return [self.get_edge(None, data_t_[j], EdgeType.Death, j) for j in range(len(data_t_))]

    def get_split_edges(self, tracklet_t, data_t) -> Tuple[List[Deque[Edge]], List[Deque[Edge]], List[MitosisNode]]:
        split_into = [collections.deque() for _ in range(len(tracklet_t))]
        split_from = [collections.deque() for _ in range(len(data_t))]
        mitosis_nodes = []
        for i in range(len(tracklet_t)):
            for j in range(i + 1, len(tracklet_t)):
                dist_bet_into = self.dist_cell_to_cell(tracklet_t[i], tracklet_t[j])
                if dist_bet_into < self.config.split_pair_max_dist:
                    for k in range(len(data_t)):
                        dist_pair_to_into = self.dist_mitosis_to_data(self.cell_mean(tracklet_t[i], tracklet_t[j]),
                                                                      data_t[k])
                        if dist_pair_to_into < self.config.split_max_dist:

                            into_edge1 = Edge(tracklet_t[i],
                                              None,
                                              pulp.LpVariable(f'split_into_{i}_{k}', lowBound=0, upBound=1),
                                              EdgeType.MitosisTo,
                                              0.
                                              )
                            split_into[i].append(into_edge1)

                            into_edge2 = Edge(tracklet_t[j],
                                              None,
                                              pulp.LpVariable(f'split_into_{j}_{k}', lowBound=0, upBound=1),
                                              EdgeType.MitosisTo,
                                              0.
                                              )
                            split_into[j].append(into_edge2)

                            from_edge = Edge(None,
                                             tracklet_t[k],
                                             pulp.LpVariable(f'split_from_{k}_into{i}_{j}', lowBound=0,
                                                             upBound=1),
                                             EdgeType.MitosisFrom,
                                             self.split_weight(dist_pair_to_into)
                                             )
                            split_from[k].append(from_edge)

                            nodes = MitosisNode([into_edge1, into_edge2], from_edge)
                            mitosis_nodes.append(nodes)

        return split_into, split_from, mitosis_nodes

    def get_edge(self, cell, cell_data, kind: EdgeType, data_id) -> Optional[Edge]:
        if kind == EdgeType.DataMatch:
            dist = self.dist_cell_to_data(cell, cell_data)
            if dist < self.config.data_match_max_dist:
                return Edge(cell, cell_data,
                            pulp.LpVariable(f'data{cell.id}_{data_id}', lowBound=0, upBound=1),
                            EdgeType.DataMatch,
                            dist * self.config.data_weight)
            else:
                return None

        if kind == EdgeType.Hidden:
            return Edge(cell, None, pulp.LpVariable(f'hidden_{cell.id}', lowBound=0),
                        EdgeType.Hidden, self.config.hidden_weight)

        if kind == EdgeType.Death:
            return Edge(None, cell_data, pulp.LpVariable(f'death_{data_id}', lowBound=0),
                        EdgeType.Death, self.config.death_weight)

        raise NotImplementedError

    def dist_cell_to_data(self, cell: CellNode, data):
        return ((self.motion_estimator(cell.pos) - data) ** 2).sum()

    def dist_mitosis_to_data(self, mito, data):
        return ((self.motion_estimator(mito) - data) ** 2).sum()

    def dist_cell_to_cell(self, cell_1: CellNode, cell_2: CellNode):
        return ((cell_1.pos - cell_2.pos) ** 2).sum()

    def split_weight(self, dist):
        return dist * self.config.split_weight * 2

    def cell_mean(self, cell1, cell2):
        return (cell1.pos + cell2.pos) / 2
