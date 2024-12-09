# Trust & Fraud Detection in SIoT Using GCN 

This project implements a **Graph Convolutional Network (GCN)** to detect fraudulent activities in Social Internet of Things (SIoT). The solution leverages **PyTorch Geometric (PyG)** to build a Relational Graph Convolutional Network (RGCN), visualizes the graph structure, and predicts binary outcomes (fraud or trust) for nodes in the graph.

---

## Project Structure

### Data Files
The project uses three main data files:
- `userid-table 1.csv`: Contains user-related features.
- `userid-table 221104.csv`: Includes additional user feature data.
- `users_relation 3.csv`: Contains user-to-user relationship data.

### Graph Construction
1. **Node Features**: User-related features (e.g., `lam1000`, `lam3000`, `lam5000`, `efficiency_x`) are normalized and used as node attributes.
2. **Edge Features**:
   - **User-to-User Relationships**: Represented by edges and weighted by `relation` scores.
   - **User-to-Service Relationships**: Generated based on service interactions (e.g., `serviceid` values).
3. **Graph Object**: Combines nodes, edges, and attributes using PyTorch Geometric's `Data` object.

---

## Key Components

### Dependencies
The project requires the following libraries:
- Python (3.8+)
- [PyTorch](https://pytorch.org/) (with CUDA for GPU support)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)
- Pandas, NumPy
- Plotly (for visualization)
- Matplotlib (for loss and accuracy plots)

Install dependencies:
```bash
pip install torch torch-geometric pandas numpy plotly matplotlib
