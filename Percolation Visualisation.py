import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# Load data
data = pd.read_csv("updated25ly3.csv")
data = pd.read_csv("new_distances.csv")
print(data.size)

def data_frame_creation(data):
    df = data
    
    colour_selection = ['Red', 'Orange', 'Blue']
    
    x, y, z, colour, sizes, name = [0], [0], [0], ['Orange'], [5], ['Sun']
    
    for i in range(1, len(df)):
        ra_rad = float(df.iloc[i, 2]) * np.pi / 180  # Right Ascension in radians
        dec_rad = float(df.iloc[i, 3]) * np.pi / 180  # Declination in radians
        distance = float(df.iloc[i, 4])  # Distance in parsecs

        # Convert to Cartesian coordinates
        x.append((1000 * distance) * 3.26 * np.sin(ra_rad) * np.cos(dec_rad))
        y.append((1000 * distance) * 3.26 * np.sin(ra_rad) * np.sin(dec_rad))
        z.append((1000 * distance) * 3.26 * np.cos(ra_rad))

    points = np.stack((x, y, z), axis=1)
    return name, x, y, z, colour, sizes, points

name, x, y, z, colour, sizes, points = data_frame_creation(data)

hop_lengths = np.arange(0, 20.5, 0.5)  # Fix: Proper step size

def compute_connected_fraction(hop_length):
    """Compute the fraction of connected stars for a given hop length."""
    G = nx.Graph()
    
    # Add nodes
    for i in range(len(points)):
        G.add_node(i)  # Using index instead of tuple

    # Add edges if within hop distance
    for i in range(len(points)):
        for j in range(0, len(points)):  # Avoid duplicate edges
            distance = np.linalg.norm(points[i] - points[j])
            if distance <= hop_length:
                G.add_edge(i, j)

    # Find largest connected component
    if nx.is_empty(G):
        return 0  # No connections

    largest_cc_size = max(len(c) for c in nx.connected_components(G))
    
    return largest_cc_size / len(G.nodes)  # Fraction of connected nodes

# Compute percolation data
connected_fractions = [compute_connected_fraction(h) for h in hop_lengths]

# Plot percolation behavior
plt.plot(hop_lengths, connected_fractions, marker='o')
plt.xlabel("Hop Length (Light Years)")
plt.ylabel("Fraction of Connected Stars")
plt.title("Percolation of Stars within 25 Light Years")
plt.grid()
plt.savefig("percolation_plot.png", dpi=300, bbox_inches='tight', format='png')
plt.show()


