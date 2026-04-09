import heapq
import math

class NavEngine:
    def __init__(self, graph_data, turn_table):
        if isinstance(graph_data, dict) and 'edges' in graph_data:
            self.graph = graph_data['edges']
            self.nodes = graph_data.get('nodes', {})
        else:
            self.graph = graph_data
            self.nodes = {}
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

    def get_action(self, prev_node, curr_node, next_node):
        # Fallback to straight if node coordinates are missing
        if not self.nodes or prev_node not in self.nodes or curr_node not in self.nodes or next_node not in self.nodes:
            return "STRAIGHT"

        dx1 = self.nodes[curr_node]['x'] - self.nodes[prev_node]['x']
        dy1 = self.nodes[curr_node]['y'] - self.nodes[prev_node]['y']
        a1 = math.degrees(math.atan2(dy1, dx1))

        dx2 = self.nodes[next_node]['x'] - self.nodes[curr_node]['x']
        dy2 = self.nodes[next_node]['y'] - self.nodes[curr_node]['y']
        a2 = math.degrees(math.atan2(dy2, dx2))

        diff = (a2 - a1) % 360
        if diff > 180: diff -= 360

        if -25 <= diff <= 25: return "STRAIGHT"
        elif 25 < diff <= 135: return "LEFT"
        elif -135 <= diff < -25: return "RIGHT"
        else: return "TURN_AROUND"