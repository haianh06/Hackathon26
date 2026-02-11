import heapq

class NavEngine:
    def __init__(self, graph, turn_table):
        self.graph = graph
        self.turn_table = turn_table

    def get_shortest_path(self, start, end):
        if start not in self.graph or end not in self.graph:
            return None
        queue = [(0, start, [])]
        seen = set()
        mins = {start: 0}
        while queue:
            (cost, v1, path) = heapq.heappop(queue)
            if v1 not in seen:
                seen.add(v1)
                path = path + [v1]
                if v1 == end: return path
                for v2, weight in self.graph.get(v1, {}).items():
                    next_cost = cost + weight
                    if next_cost < mins.get(v2, float('inf')):
                        mins[v2] = next_cost
                        heapq.heappush(queue, (next_cost, v2, path))
        return None

    def get_action(self, from_node, to_node):
        return self.turn_table.get((from_node, to_node), "STRAIGHT")