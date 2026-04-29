import os
from dataclasses import dataclass
from typing import Any

import networkx as nx
import pandas as pd
import pyomo.environ as pyo
from pyomo.core import ConcreteModel


@dataclass
class Ptn:
    G: nx.Graph
    edge_mapping: dict
    lines: dict
    od_pairs: dict
    line_costs: dict


def load_ptn(dataset_path: str) -> Ptn:
    stop_file = os.path.join(dataset_path, 'Stop.giv')
    edge_file = os.path.join(dataset_path, 'Edge.giv')

    stops_df = pd.read_csv(stop_file, sep=';', comment='#', skipinitialspace=True)
    stops_df.columns = [col.strip() for col in stops_df.columns]

    edges_df = pd.read_csv(edge_file, sep=';', comment='#', skipinitialspace=True)
    edges_df.columns = [col.strip() for col in edges_df.columns]

    G = nx.Graph()

    for _, row in stops_df.iterrows():
        G.add_node(
            int(row['stop-id']),
            x=float(row['x-coordinate']),
            y=float(row['y-coordinate']),
            name=str(row['short-name']).strip()
        )

    for _, row in edges_df.iterrows():
        G.add_edge(
            int(row['left-stop-id']),
            int(row['right-stop-id']),
            edge_id=int(row['edge-id']),
            # Expected value of uniform distribution over [lower-bound, upper-bound]
            weight=0.5 * (float(row['lower-bound']) + float(row['upper-bound']))
        )

    od_file = os.path.join(dataset_path, 'OD.giv')
    df = pd.read_csv(od_file, sep=';', comment='#', skipinitialspace=True)
    df.columns = [col.strip() for col in df.columns]
    od_pairs = {}
    col_origin = df.columns[0]
    col_dest = df.columns[1]
    col_demand = df.columns[2]
    for _, row in df.iterrows():
        origin = int(row[col_origin])
        dest = int(row[col_dest])
        demand = float(row[col_demand])
        if demand > 0 and origin != dest:
            #scale demand by 5 to make it more realistic
            od_pairs[(origin, dest)] = 5 * demand

    cost_file = os.path.join(dataset_path, 'Pool-Cost.giv')
    df = pd.read_csv(cost_file, sep=';', comment='#', skipinitialspace=True)
    df.columns = [col.strip() for col in df.columns]
    line_costs = {}
    col_line = df.columns[0]
    col_cost = 'cost' if 'cost' in df.columns else df.columns[-1]
    for _, row in df.iterrows():
        line_id = int(row[col_line])
        cost = float(row[col_cost])
        line_costs[line_id] = cost

    edge_mapping = {
        int(row['edge-id']): (int(row['left-stop-id']), int(row['right-stop-id']), row['length'])
        for _, row in edges_df.iterrows()}
    pool = pd.read_csv(dataset_path + '/Pool.giv', sep=';', comment='#',
                       skipinitialspace=True)
    pool_dict = pool.sort_values('edge-order').groupby('line-id')['edge-id'].apply(list).to_dict()
    return Ptn(G, edge_mapping, pool_dict, od_pairs, line_costs)


def generate_subpaths(pool: dict, edge_mapping: dict, stops) -> dict:
    subpaths = []

    def map_edges(edges):
        mapped_edges = []
        for e in edges:
            mapped_edges.append(edge_mapping[e][0])
            mapped_edges.append(edge_mapping[e][1])
        return set(mapped_edges)

    line_stops = {id: map_edges(e) for id, e in pool.items()}
    lines_at_stop = {stop: set([line for line, stops in line_stops.items() if stop in stops]) for stop in stops}

    for line_id, edge_ids in pool.items():
        nodes = []
        weights = []
        # Determine the correct directional flow of the first two edges
        if len(edge_ids) >= 2:
            e1_u, e1_v, e1_w = edge_mapping[edge_ids[0]]
            e2_u, e2_v, _ = edge_mapping[edge_ids[1]]
            if e1_v in (e2_u, e2_v):
                current_node, next_node = e1_u, e1_v
            else:
                current_node, next_node = e1_v, e1_u
        else:
            current_node, next_node, e1_w = edge_mapping[edge_ids[0]]

        nodes.extend([current_node, next_node])
        weights.append(e1_w)

        # Chain the remaining edges to complete the line path
        for i in range(1, len(edge_ids)):
            u, v, w = edge_mapping[edge_ids[i]]
            next_node = v if u == next_node else u
            nodes.append(next_node)
            weights.append(w)

        n = len(nodes)
        for start_idx in range(n - 1):
            t_L = 0
            origin = nodes[start_idx]
            skipping = True
            for end_idx in range(start_idx + 1, n):
                t_L += weights[end_idx - 1]

                destination = nodes[end_idx]
                if skipping and end_idx < (n - 1) and lines_at_stop[origin] <= lines_at_stop[destination]:
                    continue

                skipping = False

                subpaths.append({
                    'line_id': line_id,
                    'entry_node': origin,
                    'exit_node': destination,
                    'line_travel_time': t_L
                })
                subpaths.append({
                    'line_id': line_id,
                    'entry_node': destination,
                    'exit_node': origin,
                    'line_travel_time': t_L
                })

    subpaths_df = pd.DataFrame(subpaths).rename(columns={
        'line_id': 'line',
        'entry_node': 'entry',
        'exit_node': 'exit',
        'line_travel_time': 'time'
    })
    return {
        (row.line, row.entry, row.exit): row.time
        for row in subpaths_df.itertuples(index=False)
    }


def build_model(ptn: Ptn, subpaths: dict,
                shortest_paths: dict, budget: float,
                beta: float, rho: float, capacity: int = 100):
    G, od_pairs, line_costs = ptn.G, ptn.od_pairs, ptn.line_costs
    lines = ptn.lines.keys()
    edges = list(ptn.edge_mapping.values())
    od_time = {(i, j): k for i, dest in shortest_paths.items() for (j, k) in dest.items()
              if (i, j) in od_pairs.keys()}
    model = pyo.ConcreteModel(name="Simultaneous_Line_Planning")

    print("Sets...")
    model.E = pyo.Set(initialize=edges, dimen=3)
    model.S = pyo.Set(initialize=list(G.nodes()), doc="Transit Nodes")
    model.L = pyo.Set(initialize=list(lines), doc="Line Pool")
    model.PathIndex = pyo.Set(initialize=subpaths.keys(), dimen=3, doc="Line Pool")
    model.R = pyo.Set(initialize=list(od_pairs.keys()),
                      dimen=2,
                      doc="OD Pairs (Origin, Destination)")

    print("Variable params...")
    model.beta = pyo.Param(initialize=beta, mutable=True, doc="Change Penalty")
    model.rho = pyo.Param(initialize=rho, mutable=True, doc="Detour Factor Tolerance")
    model.B = pyo.Param(initialize=budget, mutable=True, doc="Budget")
    print("Params...")
    model.N = pyo.Param(initialize=capacity, doc="Vehicle Capacity")
    model.t_star = pyo.Param(model.R, initialize=od_time,
                             doc="Theoretical Baseline Shortest Path")
    model.demand = pyo.Param(model.R, initialize=od_pairs, default=0, doc="Demand")
    model.time = pyo.Param(model.PathIndex, initialize=subpaths, doc="Subpath Lengths")
    model.line_costs = pyo.Param(model.L, initialize=line_costs, doc="Line Costs")


    def weight_rule(m, l, a, b, s, t):
        if a == s and b != t:
            return -m.time[l, a, b]
        elif a != s and b != t:
            return -m.time[l, a, b] - m.beta
        elif a != s and b == t:
            return m.rho * m.t_star[s, t] - m.time[l, a, b] - m.beta
        elif a == s and b == t:
            return m.rho * m.t_star[s, t] - m.time[l, a, b]

    print("Weights...")
    model.omega = pyo.Expression(model.PathIndex, model.R, rule=weight_rule)

    print("Variables...")
    model.f = pyo.Var(model.L, domain=pyo.NonNegativeIntegers)
    model.x = pyo.Var(model.PathIndex, model.R, domain=pyo.NonNegativeReals)


    def objective_rule(m):
        return sum(m.x[i, w] * m.omega[i, w] for i in m.PathIndex for w in m.R)

    print("Objective...")
    model.Objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)


    def flow_conservation_rule(m, node, origin, dest):
        w = (origin, dest)

        if origin == dest:
            return pyo.Constraint.Skip

        flow_out = sum(m.x[i, w] for i in m.PathIndex if i[1] == node)
        flow_in = sum(m.x[i, w] for i in m.PathIndex if i[2] == node)

        if node == origin:
            expr = flow_out <= m.demand[w]
        elif node == dest:
            expr = flow_in <= m.demand[w]
        else:
            expr = flow_out - flow_in == 0

        if isinstance(expr, bool):
            return pyo.Constraint.Feasible if expr else pyo.Constraint.Infeasible
        return expr

    print("Flow Conservation...")
    model.FlowCons = pyo.Constraint(model.S, model.R, rule=flow_conservation_rule)


    def line_capacity_rule(m, l, a, b):
        i = (l, a, b)
        total_passengers_on_subpath = sum(m.x[i, w] for w in m.R)
        return total_passengers_on_subpath <= m.N * m.f[l]

    print("Line Capacity...")
    model.LineCap = pyo.Constraint(model.PathIndex, rule=line_capacity_rule)


    def budget_rule(m):
        return sum(m.line_costs[l] * m.f[l] for l in m.L) <= m.B

    print("Budget...")
    model.BudgetCons = pyo.Constraint(rule=budget_rule)

    return model


def run_solver(model: ConcreteModel) -> Any:
    try:
        solver = pyo.SolverFactory('appsi_gurobi')
    except ValueError:
        print("gurobi appsi failed")
        solver = pyo.SolverFactory('gurobi')

    solver.options['Threads'] = 0  # 0 allows Gurobi to use all available CPU cores
    solver.options['TimeLimit'] = 300  # 5 minute absolute cutoff
    solver.options['MIPGap'] = 0.01  # Stop when within 1% of the theoretical minimum

    print("Sending model to solver...")
    results = solver.solve(model, tee=False, load_solutions=False)

    if results.solver.termination_condition == pyo.TerminationCondition.optimal:
        model.solutions.load_from(results)
    elif results.solver.termination_condition == pyo.TerminationCondition.feasible:
        input("Suboptimal solution found. Press Enter to continue...")

    return results


def print_frequencies(model, lines):
    print([(i, model.f[i].value) for i in lines if model.f[i].value >= 0.01])


def main():
    dataset_dir = './grid/joint'
    ptn = load_ptn(dataset_dir)
    shortest_paths = dict(nx.all_pairs_dijkstra_path_length(ptn.G, weight='weight'))
    subpaths = generate_subpaths(
        pool=ptn.lines,
        edge_mapping=ptn.edge_mapping,
        stops=list(ptn.G.nodes())
    )
    model = build_model(ptn, subpaths, shortest_paths, 900.0, 5, 1.2, 50)
    print("Beta = 5; Rho = 1.2")
    run_solver(model)
    print_frequencies(model, ptn.lines.keys())
    print("Beta = 0; Rho = 1.2")
    model.beta = 0
    model.rho = 1.2
    run_solver(model)
    print_frequencies(model, ptn.lines.keys())
    print("Beta = 5; Rho = 1.1")
    model.beta = 5.0
    model.rho = 1.1
    run_solver(model)
    print_frequencies(model, ptn.lines.keys())
    print("Beta = 20; Rho = 1.3")
    model.beta = 20.0
    model.rho = 1.3
    run_solver(model)
    print_frequencies(model, ptn.lines.keys())


if __name__ == '__main__':
    main()
