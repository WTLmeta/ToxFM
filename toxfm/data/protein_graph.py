import networkx as nx
import requests


def create_protein_graph(gene_list, required_score=400):
    """Create a protein-protein interaction network using the STRING API, using directed edges and ensuring connectivity."""
    string_api_url = "https://version-11-5.string-db.org/api"
    output_format = "tsv-no-header"
    method = "network"

    params = {
        "identifiers": "%0d".join(gene_list),  # URL encode the newline character
        "species": 9606,  # human
        "required_score": required_score,
        "caller_identity": "pretrain_gnns"
    }

    response = requests.post(f"{string_api_url}/{output_format}/{method}", data=params)

    if response.status_code != 200:
        print(f"STRING API请求失败: {response.status_code}")
        # Create a connected graph as a fallback, using directed edges
        G = nx.DiGraph()  # Use a directed graph
        for gene in gene_list:
            G.add_node(gene)

        # Ensure the graph is connected: connect all nodes to the first node, forming a star structure
        if len(gene_list) > 1:
            center_node = gene_list[0]
            for gene in gene_list[1:]:
                G.add_edge(center_node, gene, weight=0.5)  # Directed edge, from the center node to other nodes

        return G

    lines = response.text.strip().split("\n")

    # Build a NetworkX directed graph
    G = nx.DiGraph()  # Use a directed graph

    # Add all genes as nodes
    for gene in gene_list:
        G.add_node(gene)

    # Add directed edges (protein interactions)
    for line in lines:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        protein_a = parts[2]
        protein_b = parts[3]
        score = float(parts[4]) / 1000  # STRING score range is 0-1000
        if protein_a in gene_list and protein_b in gene_list and protein_a != protein_b:
            # Only add a directed edge (protein_a -> protein_b)
            G.add_edge(protein_a, protein_b, weight=score)

    # Ensure the graph is connected
    # Check weakly connected components (treat the directed graph as undirected to check connectivity)
    if not nx.is_weakly_connected(G):
        # Get all weakly connected components
        components = list(nx.weakly_connected_components(G))

        # Sort by size; the largest component is the main component
        components.sort(key=len, reverse=True)
        main_component = components[0]

        # For each other component, connect one of its nodes to the main component
        for component in components[1:]:
            # Pick a node from the current component
            component_node = next(iter(component))
            # Pick a node from the main component
            main_node = next(iter(main_component))
            # Add an edge to ensure connectivity (main_node -> component_node)
            G.add_edge(main_node, component_node, weight=0.3)  # Use a lower weight to represent a weaker connection

    return G


def load_gene_list(filepath):
    """Load a gene list from a file."""
    try:
        with open(filepath, 'r') as f:
            gene_list = [line.strip() for line in f.readlines() if line.strip()]
        return gene_list
    except Exception as e:
        print(f"加载基因列表时出错: {e}")
        return []


def create_protein_data_from_graph(gene_list, required_score=400):
    """Create protein graph data from a gene list."""
    G = create_protein_graph(gene_list, required_score)

    # Convert the NetworkX graph to PyTorch Geometric format
    import torch
    from torch_geometric.data import Data

    # Create a gene-to-index mapping
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_list)}

    # Extract edges
    edge_list = []
    edge_attr_list = []
    for edge in G.edges(data=True):
        src_gene, dst_gene, attr = edge
        if src_gene in gene_to_idx and dst_gene in gene_to_idx:
            src_idx = gene_to_idx[src_gene]
            dst_idx = gene_to_idx[dst_gene]
            edge_list.append([src_idx, dst_idx])
            # Use the edge weight as the edge feature
            edge_attr_list.append([attr.get('weight', 0.5)])  # Default weight 0.5

    # Here only the edge structure is created; node features will be set based on the specific LINCS data at training time
    num_nodes = len(gene_list)

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0),
                                                                                                          dtype=torch.long)

    # Normalize the edge features
    if edge_attr_list:
        edge_attr_tensor = torch.tensor(edge_attr_list, dtype=torch.float)
        # Use min-max normalization to scale the edge features to the [0, 1] range
        min_val = edge_attr_tensor.min()
        max_val = edge_attr_tensor.max()
        if max_val != min_val:  # Avoid division by zero
            edge_attr_tensor = (edge_attr_tensor - min_val) / (max_val - min_val)
        edge_attr = edge_attr_tensor
    else:
        edge_attr = None

    # Return the graph structure; node features will be set dynamically at training time
    return edge_index, edge_attr


def load_protein_graph_data(gene_list_file, required_score=400):
    """Load precomputed protein graph data."""
    gene_list = load_gene_list(gene_list_file)
    if gene_list:
        print(f"Loaded {len(gene_list)} genes from {gene_list_file}")
        # Create the protein graph data structure
        edge_index, edge_attr = create_protein_data_from_graph(gene_list, required_score)
        print(f"Created protein graph with {len(gene_list)} nodes and {edge_index.size(1)} edges")
        return edge_index, edge_attr, gene_list
    else:
        print("Failed to load gene list")
        return None, None, []