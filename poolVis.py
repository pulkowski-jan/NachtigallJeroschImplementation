import os
from dataclasses import dataclass

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def visualize_line_pool(dataset_path: str, lines_to_plot: list):
    """
    Visualizes specific transit lines from a LinTim line pool over the base topology.
    """
    # 1. File Paths
    stop_file = os.path.join(dataset_path, 'Stop.giv')
    edge_file = os.path.join(dataset_path, 'Edge.giv')
    pool_file = os.path.join(dataset_path, 'Pool.giv')

    # 2. Parse Base Topology
    stops_df = pd.read_csv(stop_file, sep=';', comment='#', skipinitialspace=True)
    stops_df.columns = [col.strip() for col in stops_df.columns]

    edges_df = pd.read_csv(edge_file, sep=';', comment='#', skipinitialspace=True)
    edges_df.columns = [col.strip() for col in edges_df.columns]

    G = nx.Graph()
    pos = {}
    for _, row in stops_df.iterrows():
        node_id = int(row['stop-id'])
        G.add_node(node_id)
        pos[node_id] = (float(row['x-coordinate']), float(row['y-coordinate']))

    edge_mapping = {}
    for _, row in edges_df.iterrows():
        u, v = int(row['left-stop-id']), int(row['right-stop-id'])
        G.add_edge(u, v)
        edge_mapping[int(row['edge-id'])] = (u, v)

    # High-contrast palette for the transit lines
    colors = ['#78E2A1', '#3E92CC', '#003153', '#D1345B', "#FFCF56"]

    for idx, line_id in enumerate(lines_to_plot):
        # 3. Configure the Canvas
        plt.figure(figsize=(4.95, 4.11))

        # Draw the faint background topology
        nx.draw_networkx_edges(G, pos, edge_color='#C0C0C0', width=2.0)
        nx.draw_networkx_nodes(G, pos, node_color='white', edgecolors='#003153', node_size=500)
        nx.draw_networkx_labels(G, pos, font_size=14, font_color='#003153')

        # 4. Parse Pool and Overlay Lines
        pool_df = pd.read_csv(pool_file, sep=';', comment='#', skipinitialspace=True)
        pool_df.columns = [col.strip() for col in pool_df.columns]
        line_data = pool_df[pool_df['line-id'] == line_id]
        if line_data.empty:
            continue

        # Extract the node pairs for this specific line
        line_edges = []
        for _, row in line_data.iterrows():
            e_id = int(row['edge-id'])
            if e_id in edge_mapping:
                line_edges.append(edge_mapping[e_id])

        nx.draw_networkx_edges(
            G, pos,
            edgelist=line_edges,
            edge_color=colors[idx % len(colors)],
            width=4.0,
            label=f'Line {chr(idx + ord("A"))}'
        )

        # 5. Format and Export
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(f"graphics/line{line_id}.png", dpi=150, bbox_inches='tight')
        plt.close()


@dataclass
class Ptn:
    G: nx.Graph
    edge_mapping: dict
    lines: dict
    od_pairs: dict
    line_costs: dict


def load_lintim_topology(dataset_path: str) -> Ptn:
    """
    Reads LinTim Stop.giv and Edge.giv files and constructs a NetworkX graph.
    """
    stop_file = os.path.join(dataset_path, 'Stop.giv')
    edge_file = os.path.join(dataset_path, 'Edge.giv')
    # 1. Parse Stop.giv
    # Expected columns: stop-id; short-name; long-name; x-coord; y-coord
    stops_df = pd.read_csv(stop_file, sep=';', comment='#', skipinitialspace=True)
    stops_df.columns = [col.strip() for col in stops_df.columns]

    # 2. Parse Edge.giv
    # Expected columns: edge-id; left-stop-id; right-stop-id; length; lower-bound; upper-bound
    edges_df = pd.read_csv(edge_file, sep=';', comment='#', skipinitialspace=True)
    edges_df.columns = [col.strip() for col in edges_df.columns]

    # 3. Construct the Graph
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
            weight=float(row['length'])
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
            od_pairs[(origin, dest)] = demand

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
    poolDict = pool.sort_values('edge-order').groupby('line-id')['edge-id'].apply(list).to_dict()
    return Ptn(G, edge_mapping, poolDict, od_pairs, line_costs)


def visualize_transit_network(G: nx.Graph,
                              output_filename: str = 'sioux_falls_topology.png'):
    """
    Visualizes a spatial NetworkX graph and saves it as a high-resolution image for poster use.
    """
    # 1. Extract spatial coordinates for the layout
    # This creates a dictionary of {node_id: (x, y)} required by NetworkX drawing functions
    pos = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}

    # 2. Configure the Matplotlib figure canvas
    # Setting a large figure size (e.g., 12x10 inches) ensures clarity on a poster
    plt.figure(figsize=(6, 5))

    # 3. Draw Network Elements
    # Draw edges (the transit links)
    nx.draw_networkx_edges(
        G, pos,
        edge_color='#A0A0A0',
        width=2.0,
        alpha=0.7
    )

    # Draw nodes (the transit stops)
    nx.draw_networkx_nodes(
        G, pos,
        node_color='#003153',
        node_size=500,
        edgecolors='white',
        linewidths=1.5
    )

    # Draw labels (stop IDs)
    nx.draw_networkx_labels(
        G, pos,
        font_size=10,
        font_family='sans-serif',
        font_color='white',
        font_weight='bold'
    )

    nx.draw_networkx_nodes(G, pos, node_color='#F0F0F0', edgecolors='#003153', node_size=500)
    nx.draw_networkx_labels(G, pos, font_size=12,
                            font_family='sans-serif',
                            font_weight='bold',
                            font_color='#003153')

    # 4. Format the Plot
    # plt.title("Public Transportation Network", fontsize=18, fontweight='bold', pad=20)
    plt.axis('off')  # Hide the default matplotlib axes and grid

    # 5. Export for Poster
    # dpi=300 is the standard minimum for academic print materials
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"Visualization successfully saved to {output_filename}")

    # Close the plot to free memory
    plt.close()



def draw_parallel_transit_paths(ptn: Ptn, lines_frequencies,  offset_spacing=0.1, filename: str = "graphics/result.png"):
    plt.figure(figsize=(7.5, 5.6))
    fig, ax = plt.subplots()
    G = ptn.G
    edge_mapping = ptn.edge_mapping
    # Process each line individually
    paths = []
    lines = [line for line, _ in lines_frequencies]
    pos = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}
    for line_id in lines:
        edge_ids = ptn.lines[line_id]
        # Reconstruct the ordered sequence of nodes and travel times
        nodes = []

        # Determine the correct directional flow of the first two edges
        if len(edge_ids) >= 2:
            e1_u, e1_v, _ = edge_mapping[edge_ids[0]]
            e2_u, e2_v, _ = edge_mapping[edge_ids[1]]

            if e1_v in (e2_u, e2_v):
                current_node, next_node = e1_u, e1_v
            else:
                current_node, next_node = e1_v, e1_u
        else:
            current_node, next_node, _ = edge_mapping[edge_ids[0]]

        nodes.extend([current_node, next_node])

        # Chain the remaining edges to complete the line path
        for i in range(1, len(edge_ids)):
            u, v, _ = edge_mapping[edge_ids[i]]
            next_node = v if u == next_node else u
            nodes.append(next_node)
        paths.append(nodes)

    # Dictionary to track which paths traverse which edge
    # Key: canonical edge tuple (u, v), Value: list of path indices
    edge_occupancy = {}
    for path_idx, path in enumerate(paths):
        for u, v in zip(path[:-1], path[1:]):
            edge = tuple(sorted((u, v)))
            if edge not in edge_occupancy:
                edge_occupancy[edge] = []
            edge_occupancy[edge].append(path_idx)

    nx.draw_networkx_edges(G, pos, edge_color='#E0E0E0', width=1.2)
    colors = [
        '#E3000F', '#007A33', '#0054A6', '#F39200', '#6A215B',
        '#FFD200', '#00A1DE', '#C0007A', '#97D700', '#663300',
        '#777777', '#2E3192', '#E11C52'
    ]
    # Initialize names if not provided
    path_names = [f"Linia {i}" for i in lines]

    # 1. Create Legend Handles (Proxy Artists)
    legend_elements = [
        Line2D([0], [0], color=colors[i % len(colors)], lw=4, label=path_names[i])
        for i in range(len(paths))
    ]

    # 2. Draw offset edges
    for (u, v), path_indices in edge_occupancy.items():
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        # Calculate direction vector
        dx = x2 - x1
        dy = y2 - y1
        length = np.hypot(dx, dy)

        if length == 0: continue

        # Calculate normalized perpendicular vector
        nx_vec = -dy / length
        ny_vec = dx / length

        total_paths = len(path_indices)

        for i, path_idx in enumerate(path_indices):
            # Calculate offset distance centered around the original line
            # E.g., for 3 paths: -1 * spacing, 0, 1 * spacing
            offset_multiplier = i - (total_paths - 1) / 2.0
            current_offset = offset_multiplier * offset_spacing

            # Apply offset to start and end coordinates
            start_x = x1 + nx_vec * current_offset
            start_y = y1 + ny_vec * current_offset
            end_x = x2 + nx_vec * current_offset
            end_y = y2 + ny_vec * current_offset

            color = colors[path_idx % len(colors)]
            ax.plot([start_x, end_x], [start_y, end_y], color=color, linewidth=2, solid_capstyle='round')

    # Draw nodes (the transit stops)
    nx.draw_networkx_nodes(
        G, pos,
        node_color='white',
        node_size=500,
        edgecolors='#003153',
        linewidths=1.5
    )

    # Draw labels (stop IDs)
    nx.draw_networkx_labels(
        G, pos,
        font_size=10,
        font_family='sans-serif',
        font_color='#003153',
        font_weight='bold'
    )

    plt.axis('off')
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), title="Schemat sieci")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

# Execution
if __name__ == '__main__':
    visualize_line_pool("./grid/joint", [1, 4])
    ptn = load_lintim_topology("./grid/joint")
    results = [
        [(1, 6.0), (3, 1.0), (4, 2.0), (5, 1.0), (13, 2.0), (19, 1.0), (20, 1.0), (23, 1.0), (26, 1.0), (29, 1.0)],
        [(1, 5.0), (3, 1.0), (4, 3.0), (6, 1.0), (13, 2.0), (19, 1.0), (20, 1.0), (26, 1.0), (29, 1.0)],
        [(1, 3.0), (3, 2.0), (4, 2.0), (5, 1.0), (6, 1.0), (10, 1.0), (13, 2.0), (19, 1.0), (20, 1.0), (29, 1.0), (33, 1.0), (34, 1.0)],
        [(1, 2.0), (2, 1.0), (3, 3.0), (5, 1.0), (8, 1.0), (10, 1.0), (13, 2.0), (19, 1.0), (20, 1.0), (22, 1.0), (29, 1.0), (33, 1.0), (34, 1.0)]
    ]
    i = 1
    for result in results:
        draw_parallel_transit_paths(ptn, result, 3.5, f"graphics/result{i}.png")
        i += 1
