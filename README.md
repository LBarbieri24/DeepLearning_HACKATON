# HACKATON
<svg viewBox="0 0 1200 800" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#4F46E5;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#7C3AED;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="grad2" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#059669;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#0891B2;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="grad3" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#DC2626;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#EA580C;stop-opacity:1" />
    </linearGradient>
  </defs>
  
  <!-- Title -->
  <text x="600" y="30" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" font-weight="bold" fill="#1F2937">
    Robust Graph Classification with Noisy Labels
  </text>
  
  <!-- Input Graphs Section -->
  <rect x="20" y="60" width="200" height="140" rx="10" fill="#F3F4F6" stroke="#D1D5DB" stroke-width="2"/>
  <text x="120" y="85" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#374151">Input Graphs</text>
  
  <!-- Graph visualization -->
  <g transform="translate(40, 100)">
    <!-- Graph 1 -->
    <circle cx="30" cy="20" r="4" fill="#4F46E5"/>
    <circle cx="50" cy="10" r="4" fill="#4F46E5"/>
    <circle cx="60" cy="30" r="4" fill="#4F46E5"/>
    <circle cx="40" cy="40" r="4" fill="#4F46E5"/>
    <line x1="30" y1="20" x2="50" y2="10" stroke="#6B7280" stroke-width="2"/>
    <line x1="50" y1="10" x2="60" y2="30" stroke="#6B7280" stroke-width="2"/>
    <line x1="30" y1="20" x2="40" y2="40" stroke="#6B7280" stroke-width="2"/>
    <line x1="40" y1="40" x2="60" y2="30" stroke="#6B7280" stroke-width="2"/>
    
    <!-- Graph 2 -->
    <circle cx="110" cy="15" r="4" fill="#7C3AED"/>
    <circle cx="130" cy="25" r="4" fill="#7C3AED"/>
    <circle cx="120" cy="45" r="4" fill="#7C3AED"/>
    <line x1="110" y1="15" x2="130" y2="25" stroke="#6B7280" stroke-width="2"/>
    <line x1="130" y1="25" x2="120" y2="45" stroke="#6B7280" stroke-width="2"/>
    <line x1="110" y1="15" x2="120" y2="45" stroke="#6B7280" stroke-width="2"/>
  </g>
  
  <text x="120" y="175" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#6B7280">
    + Edge Dropping (augmentation)
  </text>
  
  <!-- Arrow 1 -->
  <path d="M 240 130 L 280 130" stroke="#374151" stroke-width="3" fill="none" marker-end="url(#arrowhead)"/>
  
  <!-- GNN Architecture Section -->
  <rect x="300" y="60" width="300" height="480" rx="10" fill="url(#grad1)" fill-opacity="0.1" stroke="#4F46E5" stroke-width="2"/>
  <text x="450" y="85" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="#1F2937">GNN Architecture</text>
  
  <!-- Virtual Node -->
  <rect x="320" y="100" width="260" height="30" rx="5" fill="#EDE9FE" stroke="#7C3AED" stroke-width="1"/>
  <text x="450" y="120" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#5B21B6">Virtual Node (global information)</text>
  
  <!-- GIN Layers -->
  <g transform="translate(320, 140)">
    <rect x="0" y="0" width="260" height="40" rx="5" fill="#DBEAFE" stroke="#3B82F6" stroke-width="1"/>
    <text x="130" y="25" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#1E40AF">GIN Layer 1 + BatchNorm + Dropout</text>
    
    <rect x="0" y="50" width="260" height="40" rx="5" fill="#DBEAFE" stroke="#3B82F6" stroke-width="1"/>
    <text x="130" y="75" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#1E40AF">GIN Layer 2 + BatchNorm + Dropout</text>
    
    <rect x="0" y="100" width="260" height="40" rx="5" fill="#DBEAFE" stroke="#3B82F6" stroke-width="1"/>
    <text x="130" y="125" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#1E40AF">GIN Layer 3 + BatchNorm + Dropout</text>
    
    <!-- Residual connections -->
    <path d="M -10 20 Q -30 20 -30 70 Q -30 120 -10 120" stroke="#EF4444" stroke-width="2" fill="none" stroke-dasharray="3,3"/>
    <text x="-45" y="75" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#DC2626" transform="rotate(-90, -45, 75)">Residual</text>
  </g>
  
  <!-- Jumping Knowledge -->
  <rect x="320" y="290" width="260" height="30" rx="5" fill="#FEF3C7" stroke="#F59E0B" stroke-width="1"/>
  <text x="450" y="310" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#92400E">Jumping Knowledge (JK): "last" or "sum"</text>
  
  <!-- Graph Pooling -->
  <rect x="320" y="330" width="260" height="40" rx="5" fill="#D1FAE5" stroke="#059669" stroke-width="1"/>
  <text x="450" y="355" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#047857">Graph Pooling: Mean/Max/Attention</text>
  
  <!-- Final Linear Layer -->
  <rect x="320" y="380" width="260" height="40" rx="5" fill="#F3E8FF" stroke="#8B5CF6" stroke-width="1"/>
  <text x="450" y="405" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#6D28D9">Linear Layer → Predictions</text>
  
  <!-- Architecture details -->
  <text x="450" y="450" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#4B5563">
    • Embedding Dim: 128-218
  </text>
  <text x="450" y="465" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#4B5563">
    • Batch Size: 64
  </text>
  <text x="450" y="480" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#4B5563">
    • Adam Optimizer (lr=5e-3)
  </text>
  <text x="450" y="495" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#4B5563">
    • ReduceLROnPlateau Scheduler
  </text>
  
  <!-- Arrow 2 -->
  <path d="M 620 270 L 660 270" stroke="#374151" stroke-width="3" fill="none" marker-end="url(#arrowhead)"/>
  
  <!-- Loss Functions Section -->
  <rect x="680" y="60" width="280" height="320" rx="10" fill="url(#grad2)" fill-opacity="0.1" stroke="#059669" stroke-width="2"/>
  <text x="820" y="85" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="#1F2937">Noise-Robust Loss Functions</text>
  
  <!-- GCE Loss -->
  <rect x="700" y="110" width="240" height="80" rx="5" fill="#ECFDF5" stroke="#10B981" stroke-width="2"/>
  <text x="820" y="130" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#065F46">Generalized Cross-Entropy (GCE)</text>
  <text x="820" y="150" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#047857">For Datasets A & B</text>
  <text x="820" y="165" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#065F46">q ∈ [0.7, 0.9]</text>
  <text x="820" y="180" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#6B7280">Down-weights noisy samples</text>
  
  <!-- GCOD Loss -->
  <rect x="700" y="210" width="240" height="80" rx="5" fill="#FEF2F2" stroke="#EF4444" stroke-width="2"/>
  <text x="820" y="230" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#991B1B">Graph Centroid Outlier</text>
  <text x="820" y="245" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#991B1B">Discounting (GCOD)</text>
  <text x="820" y="265" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#DC2626">For Datasets C & D</text>
  <text x="820" y="280" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#6B7280">Dynamic confidence scoring</text>
  
  <!-- Dataset Strategy -->
  <rect x="700" y="310" width="240" height="50" rx="5" fill="#F9FAFB" stroke="#9CA3AF" stroke-width="1"/>
  <text x="820" y="330" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#374151">Dataset-Specific Strategy</text>
  <text x="820" y="345" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#6B7280">Tailored loss selection</text>
  <text x="820" y="355" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#6B7280">based on noise characteristics</text>
  
  <!-- Key Innovations Box -->
  <rect x="20" y="600" width="1160" height="180" rx="10" fill="url(#grad3)" fill-opacity="0.1" stroke="#DC2626" stroke-width="2"/>
  <text x="600" y="625" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="#1F2937">Key Design Choices for Robustness</text>
  
  <g transform="translate(40, 640)">
    <!-- Column 1 -->
    <text x="0" y="20" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#DC2626">Regularization:</text>
    <text x="0" y="35" font-family="Arial, sans-serif" font-size="11" fill="#374151">• High dropout rates</text>
    <text x="0" y="50" font-family="Arial, sans-serif" font-size="11" fill="#374151">• Batch normalization</text>
    <text x="0" y="65" font-family="Arial, sans-serif" font-size="11" fill="#374151">• Edge dropping augmentation</text>
    
    <!-- Column 2 -->
    <text x="280" y="20" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#DC2626">Architecture:</text>
    <text x="280" y="35" font-family="Arial, sans-serif" font-size="11" fill="#374151">• Virtual node for global info</text>
    <text x="280" y="50" font-family="Arial, sans-serif" font-size="11" fill="#374151">• Residual connections</text>
    <text x="280" y="65" font-family="Arial, sans-serif" font-size="11" fill="#374151">• Jumping knowledge networks</text>
    
    <!-- Column 3 -->
    <text x="560" y="20" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#DC2626">Noise Handling:</text>
    <text x="560" y="35" font-family="Arial, sans-serif" font-size="11" fill="#374151">• GCE loss (A, B datasets)</text>
    <text x="560" y="50" font-family="Arial, sans-serif" font-size="11" fill="#374151">• GCOD loss (C, D datasets)</text>
    <text x="560" y="65" font-family="Arial, sans-serif" font-size="11" fill="#374151">• Adaptive learning rate</text>
    
    <!-- Column 4 -->
    <text x="840" y="20" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#DC2626">Training Strategy:</text>
    <text x="840" y="35" font-family="Arial, sans-serif" font-size="11" fill="#374151">• 2-3 layer GIN networks</text>
    <text x="840" y="50" font-family="Arial, sans-serif" font-size="11" fill="#374151">• Dataset-specific tuning</text>
    <text x="840" y="65" font-family="Arial, sans-serif" font-size="11" fill="#374151">• Robust to label noise</text>
  </g>
  
  <!-- Result -->
  <rect x="1000" y="200" width="160" height="80" rx="10" fill="#F0FDF4" stroke="#16A34A" stroke-width="2"/>
  <text x="1080" y="225" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#15803D">Final</text>
  <text x="1080" y="245" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#15803D">Predictions</text>
  <text x="1080" y="265" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#16A34A">Robust to</text>
  <text x="1080" y="275" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#16A34A">noisy labels</text>
  
  <!-- Arrow 3 -->
  <path d="M 980 270 L 1020 240" stroke="#374151" stroke-width="3" fill="none" marker-end="url(#arrowhead)"/>
  
  <!-- Arrow markers -->
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#374151"/>
    </marker>
  </defs>
</svg>
This repository contains the final submission for the Deep Learning Hackathon (Sapienza University, MSc in Artificial Intelligence and Robotics). It details our approach to graph classification in the presence of noisy labels.

Starting from the baseline structure, our solution builds upon a GNN architecture centered around Graph Isomorphism Network (GIN) convolutions. 
To enhance model performance, stability, and robustness to the noisy dataset, we incorporated several techniques:
- Batch Normalization: Applied after GIN layers and optionally before graph pooling to stabilize training dynamics.
Dropout: Utilized with relatively high probabilities on node embeddings to mitigate overfitting, especially given the noisy labels.
- Residual Connections: To facilitate easier training of deeper GNNs by improving gradient flow.
- Jumping Knowledge (JK) Connections: Configured as "last" or "sum" to effectively aggregate information from different depths of the GNN.
- Virtual Node: Integrated to enhance global information propagation across each graph, allowing the model to learn more comprehensive graph representations.
- Edge Dropping: Employed during training as a form of data augmentation and regularization, encouraging the network to learn more robust features by randomly removing edges.

A critical aspect of our approach was the selection of loss functions tailored to handle label noise. Our empirical evaluations led to the following strategy:
- For Datasets A and B, Generalized Cross-Entropy (GCE) loss (with q values typically around 0.7-0.9) yielded superior performance. GCE is known for its robustness to noise by down-weighting the contribution of potentially mislabeled samples with high loss.
- For Datasets C and D, we implemented and utilized the Graph Centroid Outlier Discounting (GCOD) loss, as proposed by Wani et al. (2023) in "Robustness of Graph Classification: failure modes, causes, and noise-resistant loss in Graph Neural Networks". GCOD dynamically estimates per-sample confidence scores to re-weight the training objective, proving more effective for the noise characteristics observed in these datasets.

While specific hyperparameter configurations varied slightly per dataset to optimize performance, several common settings emerged:
- Network Depth: 2 to 3 GIN layers.
- Embedding Dimension: Ranged from 128 to 218.
- Batch Size: Consistently set to 64.
- Optimizer & Learning Rate: Adam optimizer with an initial learning rate of 5e-3, managed by a ReduceLROnPlateau learning rate scheduler to adjust learning based on validation performance.

Repository structure:
- main.py: Main entry point for training and testing.
- source/baselinedeep_updated.py: Implements GCE and other standard baselines.
- source/gcod_optimized_updated.py: Implements the GCOD baseline.
- source/models_EDandBatch_norm.py: Defines the core GNN architecture.
- source/conv.py: Contains GIN/GCN convolution layer implementations.