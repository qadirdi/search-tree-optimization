# Search Tree Optimization Project

This project implements two different approaches to find optimal paths in a search tree:
1. Recursive search with visualization
2. Dynamic programming approach for specific cost ranges

## Project Structure
```
.
├── README.md
├── requirements.txt
├── search_tree.py      # Recursive search implementation
├── dp_search.py        # Dynamic programming implementation
├── gain.csv           # Input data file
└── search_snapshots/  # Directory for search visualization snapshots
```

## Setup
1. Clone the repository:
```bash
git clone <your-repo-url>
cd search-tree-optimization
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Recursive Search
```bash
python search_tree.py
```
This will:
- Find all paths that reach Y=0
- Create visualizations of the search process
- Save snapshots every 500,000 paths
- Generate final visualizations

### Dynamic Programming Search
```bash
python dp_search.py
```
This will:
- Find paths with total costs between 98-100
- Show top 10 solutions sorted by X value
- Generate visualization of the best solution

## Output Files
- `search_tree.png`: Visualization of the search tree
- `optimal_path.png`: Visualization of the optimal path
- `dp_solution.png`: Visualization of the best solution from DP approach
- `search_snapshots/`: Directory containing periodic snapshots of the search process

## Requirements
- Python 3.6+
- pandas
- numpy
- matplotlib
- networkx 