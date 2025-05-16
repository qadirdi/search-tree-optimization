import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Rectangle
import os

@dataclass
class Node:
    name: str
    cost: float
    gains: List[float]

@dataclass
class SearchState:
    x: float
    y: float
    path: List[Tuple[str, int]]  # (node_name, step_count)

class SearchTree:
    def __init__(self, initial_x: float, initial_y: float):
        self.initial_x = initial_x
        self.initial_y = initial_y
        self.nodes: Dict[str, Node] = {}
        self.solutions: List[SearchState] = []
        self.paths_checked = 0
        self.last_update_time = time.time()
        self.best_x = float('inf')
        
        # Track explored paths and their results
        self.explored_paths: Set[str] = set()
        self.path_results: Dict[str, Tuple[float, float]] = {}  # path -> (X, Y)
        self.path_tree: Dict[str, List[str]] = defaultdict(list)  # parent -> children
        
        # Track solutions found
        self.solutions_count = 0
        
        # Create snapshots directory
        self.snapshot_dir = "search_snapshots"
        if not os.path.exists(self.snapshot_dir):
            os.makedirs(self.snapshot_dir)
        
        # Define correct node costs
        self.node_costs = {
            'power': 1.18,   # 100/1.18 ≈ 84.7 steps possible
            'long': 10,      # 100/10 = 10 steps possible
            'lateral': 20,   # 100/20 = 5 steps possible
            'weight': 1.2,   # 100/1.2 ≈ 83.3 steps possible
            'aero': 5        # 100/5 = 20 steps possible
        }
        
        # Define colors for each node type
        self.node_colors = {
            'power': '#FF9999',    # Light red
            'long': '#99FF99',     # Light green
            'lateral': '#9999FF',  # Light blue
            'weight': '#FFFF99',   # Light yellow
            'aero': '#FF99FF'      # Light purple
        }
        
        self.load_gains()

    def load_gains(self):
        # Read gains from CSV file
        df = pd.read_csv('gain.csv', index_col=0)
        
        # Process each node's gains
        for node_name, row in df.iterrows():
            if node_name.lower() in self.node_costs:  # Case-insensitive comparison
                # Convert row to list, dropping any NaN values
                gains = row.dropna().tolist()
                self.nodes[node_name.lower()] = Node(
                    name=node_name.lower(),
                    cost=self.node_costs[node_name.lower()],
                    gains=gains
                )
        print("Loaded gain values for nodes:", ", ".join(self.nodes.keys()))

    def path_to_string(self, path: List[Tuple[str, int]]) -> str:
        """Convert a path to a string representation"""
        return " -> ".join([f"{node}_{step}" for node, step in path]) or "root"

    def print_path_summary(self, state: SearchState) -> str:
        """Returns a short summary of the current path"""
        node_counts = {}
        for node, _ in state.path:
            node_counts[node] = node_counts.get(node, 0) + 1
        return " + ".join([f"{count}{node[:2]}" for node, count in node_counts.items()])

    def print_tree_visualization(self, max_depth: int = 4):
        """Print a tree visualization of explored paths up to max_depth"""
        print("\nSearch Tree Visualization (limited depth):")
        print("Format: node_stepnumber (X value, Y value)")
        print("root (953.844691, 100.000000)")
        
        def print_node(path: str, depth: int, prefix: str):
            if depth >= max_depth:
                return
            children = sorted(self.path_tree[path])
            for i, child in enumerate(children):
                is_last = i == len(children) - 1
                x, y = self.path_results[child]
                
                # Print current node
                print(f"{prefix}{'└── ' if is_last else '├── '}{child.split(' -> ')[-1]} ({x:.6f}, {y:.6f})")
                
                # Print children
                new_prefix = prefix + ('    ' if is_last else '│   ')
                print_node(child, depth + 1, new_prefix)
        
        print_node("root", 0, "")

    def visualize_search_tree(self, max_depth: int = 4, filename: str = 'search_tree.png'):
        """Create a graphical visualization of the search tree"""
        G = nx.DiGraph()
        
        # Add root node
        root = "root"
        G.add_node(root, x=self.initial_x, y=self.initial_y)
        
        def add_children(parent: str, depth: int):
            if depth >= max_depth:
                return
            children = self.path_tree[parent]
            for child in children:
                x, y = self.path_results[child]
                node_type = child.split('_')[0] if '_' in child else 'root'
                G.add_node(child, x=x, y=y, node_type=node_type)
                G.add_edge(parent, child)
                add_children(child, depth + 1)
        
        add_children(root, 0)
        
        # Create the plot
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes
        for node in G.nodes():
            node_type = G.nodes[node].get('node_type', 'root')
            color = self.node_colors.get(node_type, 'gray')
            nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=color, node_size=1000, alpha=0.7)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)
        
        # Add labels
        labels = {node: f"{node}\nX={G.nodes[node]['x']:.2f}\nY={G.nodes[node]['y']:.2f}" for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title(f"Search Tree Visualization\nPaths: {self.paths_checked:,}, Solutions: {self.solutions_count}")
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()

    def create_search_snapshot(self):
        """Create a snapshot of the current search state"""
        snapshot_file = os.path.join(self.snapshot_dir, f"search_tree_{self.paths_checked:08d}.png")
        self.visualize_search_tree(max_depth=4, filename=snapshot_file)
        print(f"\nCreated search snapshot at {self.paths_checked:,} paths")

    def visualize_optimal_path(self, optimal_path: List[Tuple[str, int]]):
        """Create a visualization of the optimal path progression"""
        plt.figure(figsize=(12, 8))
        
        # Calculate points for the plot
        points = [(self.initial_x, self.initial_y)]
        current_x = self.initial_x
        current_y = self.initial_y
        
        for node_name, step in optimal_path:
            node = self.nodes[node_name]
            gain = node.gains[step]
            current_x += gain
            current_y -= node.cost
            points.append((current_x, current_y))
        
        # Convert points to arrays for plotting
        points = np.array(points)
        
        # Plot the path
        plt.plot(points[:, 0], points[:, 1], 'b-', linewidth=2, alpha=0.6)
        plt.plot(points[:, 0], points[:, 1], 'ko', markersize=8)
        
        # Add labels for each point
        for i, (x, y) in enumerate(points):
            if i == 0:
                label = "Start"
            else:
                node_name, step = optimal_path[i-1]
                label = f"{node_name}_{step+1}"
            plt.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('X Value')
        plt.ylabel('Y Value')
        plt.title('Optimal Path Progression')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('optimal_path.png', bbox_inches='tight', dpi=300)
        plt.close()

    def search(self, state: SearchState):
        self.paths_checked += 1
        current_time = time.time()
        
        # Create snapshot every 500,000 paths
        if self.paths_checked % 500000 == 0:
            self.create_search_snapshot()
        
        # Convert current path to string for tracking
        path_str = self.path_to_string(state.path)
        if path_str in self.explored_paths:
            return
        
        # Track this path
        self.explored_paths.add(path_str)
        self.path_results[path_str] = (state.x, state.y)
        if len(state.path) > 0:
            parent_path = self.path_to_string(state.path[:-1])
            self.path_tree[parent_path].append(path_str)
        
        # Show progress update every 5 seconds
        if current_time - self.last_update_time > 5:
            print(f"\rPaths checked: {self.paths_checked:,} | Solutions found: {self.solutions_count} | Current path: {self.print_path_summary(state)} | Best X: {self.best_x:.6f}", end="")
            self.last_update_time = current_time

        # Check if this is a solution (Y = 0)
        if abs(state.y) < 1e-10:  # Using small epsilon for floating-point comparison
            self.solutions_count += 1
            if state.x < self.best_x:
                self.best_x = state.x
                print(f"\nNew best solution found! X = {state.x:.6f} | Path: {self.print_path_summary(state)}")
            self.solutions.append(state)
            return  # Stop this path but continue exploring others
        
        # Stop this path if Y < 0 (invalid)
        if state.y < 0:
            return

        # Calculate remaining cost capacity for each node type
        remaining_capacity = {}
        for node_name, node in self.nodes.items():
            current_usage = sum(node.cost for n, _ in state.path if n == node_name)
            remaining_capacity[node_name] = 100 - current_usage

        # Try each node that still has capacity
        for node_name, node in self.nodes.items():
            if remaining_capacity[node_name] < node.cost:
                continue  # Skip if not enough capacity left for this node

            # Calculate the next step number for this node
            current_steps = sum(1 for n, _ in state.path if n == node_name)
            
            # Check if we've used all available steps for this node
            max_steps = min(len(node.gains), int(100 / node.cost))
            if current_steps >= max_steps:
                continue  # Skip if we've used all available gains or reached max steps

            # Calculate new X and Y
            gain = node.gains[current_steps]
            new_x = state.x + gain
            new_y = state.y - node.cost

            # Create new path
            new_path = state.path + [(node_name, current_steps)]

            # Create new state and continue search
            new_state = SearchState(new_x, new_y, new_path)
            self.search(new_state)

    def find_optimal_path(self) -> Tuple[float, List[Tuple[str, int]]]:
        print("Starting search for optimal path...")
        print("This will explore ALL possible paths that reach Y=0")
        print(f"Search snapshots will be saved in the '{self.snapshot_dir}' directory every 500,000 paths")
        
        initial_state = SearchState(self.initial_x, self.initial_y, [])
        self.search(initial_state)
        
        # Create final snapshot
        final_snapshot = os.path.join(self.snapshot_dir, "search_tree_final.png")
        self.visualize_search_tree(max_depth=4, filename=final_snapshot)
        
        print(f"\nSearch completed! Checked {self.paths_checked:,} paths.")
        print(f"Found {self.solutions_count} different paths that reach Y=0")
        print(f"Search snapshots saved in '{self.snapshot_dir}'")

        # Print exploration statistics
        print(f"\nExploration Statistics:")
        print(f"Total unique paths explored: {len(self.explored_paths)}")
        print(f"Total solutions found (Y=0): {len(self.solutions)}")
        
        # Print tree visualization
        self.print_tree_visualization(max_depth=4)
        
        if not self.solutions:
            raise ValueError("No solutions found!")

        # Find the solution with minimum X
        optimal_solution = min(self.solutions, key=lambda s: s.x)
        
        # Create visualizations
        print("\nGenerating visualizations...")
        self.visualize_search_tree(max_depth=4)
        self.visualize_optimal_path(optimal_solution.path)
        print("Visualizations saved as 'search_tree.png' and 'optimal_path.png'")
        
        # Print all solutions sorted by X value
        print("\nAll solutions found (sorted by X value):")
        sorted_solutions = sorted(self.solutions, key=lambda s: s.x)
        for i, solution in enumerate(sorted_solutions[:10], 1):  # Show top 10
            print(f"\nSolution {i}:")
            print(f"X value: {solution.x:.6f}")
            print(f"Path: {self.print_path_summary(solution)}")
        
        if len(sorted_solutions) > 10:
            print(f"\n... and {len(sorted_solutions) - 10} more solutions")
        
        return optimal_solution.x, optimal_solution.path

def main():
    # Initial parameters
    X0 = 953.844691416525
    Y0 = 100

    print(f"Starting with X0 = {X0:.6f}, Y0 = {Y0}")
    
    # Create search tree
    tree = SearchTree(X0, Y0)
    
    # Find optimal path
    try:
        min_x, optimal_path = tree.find_optimal_path()
        print(f"\n{'='*50}")
        print(f"FINAL SOLUTION:")
        print(f"Minimum X value found: {min_x:.6f}")
        print(f"Improvement: {X0 - min_x:.6f}")
        
        print("\nOptimal path:")
        node_counts = {}
        for node_name, _ in optimal_path:
            node_counts[node_name] = node_counts.get(node_name, 0) + 1
        for node, count in node_counts.items():
            print(f"  {node}: {count} steps")
            
        # Print detailed solution information
        print("\nStep-by-step progression:")
        current_x = X0
        current_y = Y0
        print(f"Start: X = {current_x:.6f}, Y = {current_y}")
        
        total_cost = 0
        for node_name, step in optimal_path:
            node = tree.nodes[node_name]
            gain = node.gains[step]
            current_x += gain
            current_y -= node.cost
            total_cost += node.cost
            # Only print steps where there's a significant change
            if abs(gain) > 0.3 or abs(current_y) < 1:  # Print big changes or final steps
                print(f"After {node_name} (Step {step + 1}): X = {current_x:.6f}, Y = {current_y:.6f}")
                print(f"  Cost: -{node.cost}, Gain: {gain:.6f}")
        print(f"\nTotal cost used: {total_cost:.6f}")
            
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 