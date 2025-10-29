"""
FIXED: SOTA-Informed GNN Architecture for Node Ranking

Key Fixes Applied:
1. Spectral filter now uses set-to-set relationships (not scalar-to-scalar)
2. Parallel pathways (spectral, spatial, temporal) - no information loss
3. Temporal encoder uses spectral features (global context)
4. Multi-scale temporal encoding by default
5. Proper fusion of all three pathways

Based on SSGNN principles and your project goals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from temporal_encoder import MultiScaleTemporalEncoder
from typing import List, Dict, Optional, Tuple



class ImprovedSpectralFilter(nn.Module):
    """
    FIXED: Set-to-set spectral filtering.
    
    Key improvement: Each eigenvalue's filter depends on ALL other eigenvalues,
    capturing spectral gaps and relative positions.
    """
    
    def __init__(self, in_dim: int, out_dim: int, k: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k = k
        
        # Stage 1: Encode each eigenvalue independently
        self.eig_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 32),
            nn.LayerNorm(32)
        )
        
        # Stage 2: Cross-eigenvalue context (set-to-set)
        # This is the KEY difference from your original implementation
        self.eig_attention = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Stage 3: Generate filter coefficients using full eigenvalue context
        self.filter_generator = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Tanh()  # Bound filter magnitudes [-1, 1]
        )
        
        # Stage 4: Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, eigenvectors: torch.Tensor,
                eigenvalues: torch.Tensor, eig_mask: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor: # <-- ADD batch
        N, D = x.shape
        k_target = self.k

        # Determine batch size
        if batch is None: # Handle single graph case
            batch = torch.zeros(N, dtype=torch.long, device=x.device)
            bsz = 1
        else:
            bsz = batch.max().item() + 1 # Number of graphs in batch

        # --- Process Eigenvalues Graph-by-Graph for Attention & Filters ---
        filters_list = []
        for i in range(bsz):
            # Extract eigenvalues and mask for the current graph
            # Assuming eigenvalues/mask are concatenated [B*k]
            start_idx = i * k_target
            end_idx = (i + 1) * k_target
            
            eigvals_graph = eigenvalues[start_idx:end_idx]
            eig_mask_graph = eig_mask[start_idx:end_idx]
            
            # --- Handle potential padding/truncation within the BATCH slice ---
            # (Dataset loader should already handle this, but add safety)
            current_k_in_slice = len(eigvals_graph)
            if current_k_in_slice < k_target:
                 pad_k = k_target - current_k_in_slice
                 eigvals_graph = F.pad(eigvals_graph, (0, pad_k))
                 eig_mask_graph = F.pad(eig_mask_graph, (0, pad_k), value=False)
            elif current_k_in_slice > k_target:
                 eigvals_graph = eigvals_graph[:k_target]
                 eig_mask_graph = eig_mask_graph[:k_target]
            # --- End safety check ---

            if k_target == 0: # Handle k=0 case
                 filters_list.append(torch.empty(0, device=x.device))
                 continue

            # Stage 1: Encode eigenvalues (for this graph)
            eig_features = self.eig_encoder(eigvals_graph.unsqueeze(-1))  # [k, 32]

            # Stage 2: Cross-eigenvalue attention (for this graph)
            padding_mask = ~eig_mask_graph.bool() # [k]

            # Attention expects (SeqLen, Batch, Emb) or (Batch, SeqLen, Emb) if batch_first=True
            # Here, Batch=1 for the single graph processing
            eig_context, _ = self.eig_attention(
                eig_features.unsqueeze(0), # Query [1, k, 32]
                eig_features.unsqueeze(0), # Key   [1, k, 32]
                eig_features.unsqueeze(0), # Value [1, k, 32]
                key_padding_mask=padding_mask.unsqueeze(0) # [1, k]
            ) # Output eig_context shape [1, k, 32]
            eig_context = eig_context.squeeze(0) # [k, 32]

            # Stage 3: Generate filters (for this graph)
            filters_graph = self.filter_generator(eig_context).squeeze(-1)  # [k]

            # Zero-out filters for padded dimensions
            filters_graph = filters_graph.masked_fill(padding_mask, 0.0) # [k]
            filters_list.append(filters_graph)
            
        # filters tensor now shape [B, k] after stacking
        filters = torch.stack(filters_list, dim=0)

        # Stage 4: Apply spectral filtering (Graph by Graph using batch index)
        output_list = []
        for i in range(bsz):
            graph_node_mask = (batch == i)
            x_graph = x[graph_node_mask] # [N_i, D]
            eigvecs_graph = eigenvectors[graph_node_mask] # [N_i, k] - Assumes eigenvectors follow node batching
            filters_graph = filters[i] # [k]

            if x_graph.numel() == 0 or eigvecs_graph.numel() == 0 or k_target == 0:
                # Handle empty graph or k=0 case - project directly
                output_list.append(self.output_proj(x_graph)) # Apply projection even if empty
                continue
                
            x_freq_graph = eigvecs_graph.t() @ x_graph # [k, N_i] @ [N_i, D] -> [k, D]
            
            # Apply filters
            filtered_freq_graph = filters_graph.unsqueeze(-1) * x_freq_graph # [k, D] (Broadcasting)
            
            # Transform back to spatial domain
            x_spatial_graph = eigvecs_graph @ filtered_freq_graph # [N_i, k] @ [k, D] -> [N_i, D]
            
            # Project and add to list
            output_list.append(self.output_proj(x_spatial_graph)) # [N_i, out_dim]

        # Concatenate results for the whole batch
        x_spectral_out = torch.cat(output_list, dim=0) # [TotalNodes, out_dim]

        return x_spectral_out


class TypeAwareGATv2Conv(nn.Module):
    """
    GATv2Conv with type-awareness for fact vs rule nodes.
    (Unchanged from original - this part was good)
    """
    
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.3, edge_dim=None):
        super().__init__()
        
        self.gat_fact_to_rule = GATv2Conv(
            in_channels, out_channels // heads, heads=heads,
            dropout=dropout, edge_dim=edge_dim, concat=True
        )
        self.gat_rule_to_fact = GATv2Conv(
            in_channels, out_channels // heads, heads=heads,
            dropout=dropout, edge_dim=edge_dim, concat=True
        )
        
        self.type_proj = nn.Linear(2, out_channels // 4)
        self.combine = nn.Linear(out_channels + out_channels // 4, out_channels)
        
    def forward(self, x, edge_index, node_types, edge_attr=None):
        h1 = self.gat_fact_to_rule(x, edge_index, edge_attr=edge_attr)
        h2 = self.gat_rule_to_fact(x, edge_index, edge_attr=edge_attr)
        type_emb = self.type_proj(node_types)
        h = torch.cat([h1 + h2, type_emb], dim=-1)
        return self.combine(h)


class NodeRankingGNN(nn.Module):
    """
    Structural pathway using type-aware message passing.
    (Unchanged from original - this part was good)
    """
    
    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 3, 
                 num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.edge_encoder = nn.Embedding(3, hidden_dim // num_heads)
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(
                TypeAwareGATv2Conv(
                    hidden_dim, hidden_dim, heads=num_heads,
                    dropout=dropout, edge_dim=hidden_dim // num_heads
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        self.graph_summary = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.node_scorer = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.dropout = dropout
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        node_types = x[:, :2]
        h = self.input_proj(x)
        
        if edge_attr is not None and len(edge_attr) > 0:
            edge_emb = self.edge_encoder(edge_attr)
        else:
            edge_emb = None
        
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_res = h
            h = conv(h, edge_index, node_types, edge_attr=edge_emb)
            h = norm(h + h_res)
            
            if i < self.num_layers - 1:
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        if batch is not None:
            graph_mean = global_mean_pool(h, batch)
            graph_max = global_max_pool(h, batch)
            graph_summary = self.graph_summary(torch.cat([graph_mean, graph_max], dim=-1))
            node_graph_summary = graph_summary[batch]
        else:
            graph_mean = h.mean(dim=0, keepdim=True)
            graph_max = h.max(dim=0, keepdim=True)[0]
            graph_summary = self.graph_summary(torch.cat([graph_mean, graph_max], dim=-1))
            node_graph_summary = graph_summary.expand(h.size(0), -1)
        
        node_input = torch.cat([h, node_graph_summary], dim=-1)
        scores = self.node_scorer(node_input).squeeze(-1)
        
        return scores, h


class FixedTemporalSpectralGNN(nn.Module):
    """
    FIXED: Properly combines spectral, spatial, and temporal pathways.
    
    Key fixes:
    1. Parallel pathways (no sequential information loss)
    2. Temporal encoder uses spectral features (global context)
    3. Multi-scale temporal encoding by default
    4. Late fusion of all three pathways
    """
    
    def __init__(self, in_dim, hidden_dim=256, num_layers=3, num_heads=4, 
                 dropout=0.3, k=16):
        super().__init__()
        
        self.base_dim = in_dim
        self.k = k
        self.hidden_dim = hidden_dim
        # Takes the fused graph embedding as input
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), # Input is 3*hidden_dim
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() # Output value between 0 (end) and 1 (start)
        )
        
        # Pathway 1: Spectral (global structure)
        self.spectral_filter = ImprovedSpectralFilter(
            in_dim=self.base_dim,
            out_dim=hidden_dim,
            k=self.k
        )
        
        # Pathway 2: Structural (local message passing)
        # FIXED: Takes base features, not spectral-enriched features
        self.struct_gnn = NodeRankingGNN(
            in_dim=self.base_dim,  # Direct from base features
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Pathway 3: Temporal (proof state evolution)
        # FIXED: Now uses MultiScaleTemporalEncoder
        self.temporal_encoder = MultiScaleTemporalEncoder(
            hidden_dim=hidden_dim,
            num_scales=3,  # Fine, medium, coarse
            max_steps=100,
            dropout=dropout
        )
        
        # FIXED: Fusion layer takes ALL THREE pathways
        # Input: hidden_dim (spectral) + hidden_dim (spatial) + hidden_dim (temporal)
        self.fusion_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, edge_index, derived_mask, step_numbers, eigvecs, eigvals, eig_mask,  
                edge_attr=None, batch=None):
        """
        FIXED forward pass with parallel pathways.
        
        Returns:
            scores: [N] node scores for ranking
            embeddings: [N, hidden_dim*3] combined embeddings (for loss computation)
        """
        # FIXED: All three pathways operate independently (parallel)
        
        # Pathway 1: Spectral - captures global proof structure
        h_spectral = self.spectral_filter(x, eigvecs, eigvals, eig_mask, batch)  # [N, hidden_dim]
        
        # Pathway 2: Structural - captures local rule dependencies
        # FIXED: Uses base features x, not spectral-enriched
        _, h_spatial = self.struct_gnn(x, edge_index, edge_attr, batch)  # [N, hidden_dim]
        
        # Pathway 3: Temporal - captures proof evolution
        # FIXED: Uses spectral features (global context), not spatial
        h_temporal = self.temporal_encoder(
            derived_mask, 
            step_numbers, 
            h_spectral  # Use spectral, not spatial!
        )  # [N, hidden_dim]
        
        # FIXED: Late fusion - concatenate all three pathways
        h_combined = torch.cat([h_spectral, h_spatial, h_temporal], dim=-1)  # [N, 3*hidden_dim]
        
        # Final scoring
        scores = self.fusion_scorer(h_combined).squeeze(-1)  # [N]

        if batch is None:
             # Handle single graph case
             graph_embedding = torch.mean(h_combined, dim=0, keepdim=True)
        else:
             graph_embedding = global_mean_pool(h_combined, batch) # [batch_size, 3*hidden_dim]
        # --- END MOVE ---
        value = self.value_head(graph_embedding)  # [batch_size, 1]

        if batch is None:
            value = value.squeeze(0)  # Scalar
        else:
            value = value.squeeze(-1)  # [batch_size]
        
        return scores, h_combined, value


class TacticClassifier(nn.Module):
    """Predicts high-level tactic from graph embedding."""
    def __init__(self, hidden_dim, num_tactics=6, dropout=0.3):
        super().__init__()
        self.tactic_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), # Takes 3-pathway embedding
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tactics)
        )
    
    def forward(self, graph_embedding):
        """Predict tactic distribution."""
        return self.tactic_head(graph_embedding) # Return logits

class TacticGuidedGNN(nn.Module):
    """
    FIXED: SOTA-Informed GNN Architecture
    Wraps the Phase 2 GNN and adds Phase 1's Tactic Abstraction.
    """
    def __init__(self, in_dim, hidden_dim=256, num_layers=3, num_heads=4, 
                 dropout=0.3, k=16, num_tactics=6, num_rules=5000): # num_rules is approx
        super().__init__()
        
        # Part 1: The core 3-pathway GNN (Phase 2)
        self.base_model = FixedTemporalSpectralGNN(
            in_dim, hidden_dim, num_layers, num_heads, dropout, k
        )
        
        # Part 2: Tactic Classifier (from Phase 1)
        self.tactic_classifier = TacticClassifier(
            hidden_dim, num_tactics, dropout
        )
        
        # Part 3: Tactic-to-Rule Modulation (from Phase 1)
        # This layer learns to map a tactic (e.g., "forward_chain")
        # to a bias vector over all rules.
        self.tactic_modulation = nn.Linear(num_tactics, hidden_dim // 2)
        
        # Final scorer needs to incorporate this bias
        self.final_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3 + hidden_dim // 2, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Re-use the value head from the base model
        self.value_head = self.base_model.value_head

    def forward(self, x, edge_index, derived_mask, step_numbers, eigvecs, eigvals, eig_mask,
                edge_attr=None, batch=None):
        
        # 1. Get base scores, embeddings, and value
        # We'll re-implement the base forward pass to get embeddings
        
        # --- Base Model Forward Pass ---
        h_spectral = self.base_model.spectral_filter(x, eigvecs, eigvals, eig_mask, batch)
        _, h_spatial = self.base_model.struct_gnn(x, edge_index, edge_attr, batch)
        h_temporal = self.base_model.temporal_encoder(derived_mask, step_numbers, h_spectral)
        h_combined = torch.cat([h_spectral, h_spatial, h_temporal], dim=-1)
        
        
        if batch is None:
             graph_embedding = torch.mean(h_combined, dim=0, keepdim=True)
             if x.shape[0] > 0:
                 batch_indices = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
             else:
                 batch_indices = torch.empty(0, dtype=torch.long, device=x.device)
        else:
             graph_embedding = global_mean_pool(h_combined, batch)
             batch_indices = batch
            
        value = self.value_head(graph_embedding)

        if batch is None:
            value = value.view(1) # Scalar for single graph
        else:
            value = value.squeeze(-1) # [batch_size] for batched graph
        
        # --- End Base Model Pass ---
        
        # 2. Predict Tactic
        tactic_logits = self.tactic_classifier(graph_embedding) # [batch_size, num_tactics]
        tactic_probs = F.softmax(tactic_logits, dim=-1)
        
        # 3. Modulate Scores by Tactic
        tactic_bias_features = self.tactic_modulation(tactic_probs) # [batch_size, hidden/2]
        
        # Broadcast tactic bias to all nodes in the batch
        tactic_bias_per_node = tactic_bias_features[batch_indices] # [N, hidden/2]
        
        # 4. Final Scoring
        # Concatenate node features with the tactic bias
        scoring_input = torch.cat([h_combined, tactic_bias_per_node], dim=-1)
        
        # The original model.fusion_scorer is replaced by this
        modulated_scores = self.final_scorer(scoring_input).squeeze(-1) # [N]
        # print("value_head type:", type(self.value_head))
        # print("graph_embedding shape:", graph_embedding.shape)
        return modulated_scores, h_combined, value, tactic_logits
# --- END NEW ---



    """
    Factory function to get the fixed spectral-temporal model.
    
    Args:
        in_dim: Base feature dimension (e.g., 22)
        hidden_dim: Hidden dimension for all pathways
        num_layers: Number of GNN layers in structural pathway
        dropout: Dropout rate
        use_type_aware: Whether to use type-aware GNN (always True now)
        k: Spectral dimension
    
    Returns:
        FixedTemporalSpectralGNN model
    """

    
class PremiseSelector(nn.Module):
    """
    Selects relevant facts (premises) for a given goal (GPT-f style).
    This module is trained separately or jointly.
    For now, we define it for use by the search agent.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # Bilinear scorer: (fact_embedding, rule_embedding) -> score
        self.relevance_scorer = nn.Bilinear(hidden_dim * 3, hidden_dim * 3, 1)
    
    def forward(self, 
                fact_embeddings: torch.Tensor, 
                rule_embeddings: torch.Tensor,
                fact_batch_indices: torch.Tensor,
                rule_batch_indices: torch.Tensor):
        """
        Scores the relevance of each fact to each rule in the batch.
        
        Args:
            fact_embeddings: [N_facts, 3*hidden_dim]
            rule_embeddings: [N_rules, 3*hidden_dim]
            fact_batch_indices: [N_facts]
            rule_batch_indices: [N_rules]
            
        Returns:
            relevance_scores: [N_facts, N_rules]
        """
        # This is a complex operation. A simpler version:
        # Score relevance of each fact to the *graph* (goal).
        
        # Simple scorer: (fact_embedding, graph_embedding) -> score
        # self.relevance_scorer = nn.Bilinear(hidden_dim * 3, hidden_dim * 3, 1)
        
    def score_fact_relevance(self, 
                             fact_embeddings: torch.Tensor, 
                             graph_embedding: torch.Tensor):
        """
        Args:
            fact_embeddings: [N, 3*hidden_dim]
            graph_embedding: [1, 3*hidden_dim]
        
        Returns:
            relevance_scores: [N]
        """
        graph_expanded = graph_embedding.expand_as(fact_embeddings)
        relevance = self.relevance_scorer(fact_embeddings, graph_expanded).squeeze(-1)
        return torch.sigmoid(relevance)
# --- END NEW ---
def get_model(in_dim, hidden_dim=256, num_layers=4, dropout=0.3, 
              use_type_aware=True, k=16, num_tactics=6, num_rules=5000): # <-- NEW ARGS
    
    """
    Factory function to get the fixed Tactic-guid model.
    
    Args:
        in_dim: Base feature dimension (e.g., 22)
        hidden_dim: Hidden dimension for all pathways
        num_layers: Number of GNN layers in structural pathway
        dropout: Dropout rate
        use_type_aware: Whether to use type-aware GNN (always True now)
        k: Spectral dimension
    
    Returns:
        FixedTemporalSpectralGNN model
    """

    num_heads = 4
    
    print(f"Initializing FIXED TacticGuidedGNN:")
    print(f"  Base features: {in_dim}D")
    print(f"  Hidden dim: {hidden_dim}D per pathway")
    print(f"  Spectral dim: k={k}")
    print(f"  Structural layers: {num_layers}")
    print(f"  Temporal scales: 3 (fine, medium, coarse)")
    print(f"  Tactics: {num_tactics}") # <-- NEW
    
    # --- MODIFIED: Return the new wrapper model ---
    return TacticGuidedGNN(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        k=k,
        num_tactics=num_tactics,
        num_rules=num_rules
    )