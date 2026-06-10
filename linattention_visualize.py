# /// script
# dependencies = [
#   "torch",
#   "matplotlib",
#   "numpy",
# ]
# ///

import os
import time
import math
import copy
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Ensure reports directory exists
os.makedirs("reports", exist_ok=True)

# Import core simulation from altprop_linattention
from altprop_linattention import (
    NUM_LAYERS,
    LOSS_REDUCTION_FACTOR,
    MultiLayerSelfAttention,
    generate_orthogonal_rows_batch,
    step_classic,
    step_fixed,
    run_step_with_lr_tuning
)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def compute_average_angle(pred, target):
    """Computes the average angle in radians normalized by pi (ranges from 0 to 1) between predicted and target vectors."""
    cos_sim = F.cosine_similarity(pred, target, dim=-1)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    angles = torch.acos(cos_sim)
    angles_normalized = angles / math.pi
    return torch.mean(angles_normalized).item()

def run_initial_lr_search(student_initial_state, teacher, dim, batch_size, num_rows):
    """Computes and plots learning rate sensitivity for both backprop and altprop, and returns half of the max stable LRs."""
    print("\nRunning Initial Learning Rate Search (1-step average angle sensitivity)...")
    student_init = MultiLayerSelfAttention(dim, num_layers=len(list(teacher)))
    student_init.load_state_dict(copy.deepcopy(student_initial_state))
    
    torch.manual_seed(123)
    batch_x_a = generate_orthogonal_rows_batch(batch_size, num_rows, dim)
    with torch.no_grad():
        batch_y_a = teacher(batch_x_a)

    batch_x_b = generate_orthogonal_rows_batch(batch_size, num_rows, dim)
    with torch.no_grad():
        batch_y_b = teacher(batch_x_b)

    with torch.no_grad():
        init_angle_a = compute_average_angle(student_init(batch_x_a), batch_y_a)

    lrs = np.logspace(-2, 4, 300)
    
    results_c = {"same": [], "diff": []}
    results_f = {"same": [], "diff": []}

    for lr in lrs:
        # backprop
        model_c = MultiLayerSelfAttention(dim, num_layers=len(list(teacher)))
        model_c.load_state_dict(copy.deepcopy(student_initial_state))
        step_classic(model_c, batch_x_a, batch_y_a, lr)
        with torch.no_grad():
            angle_same_c = compute_average_angle(model_c(batch_x_a), batch_y_a)
            angle_diff_c = compute_average_angle(model_c(batch_x_b), batch_y_b)
            if not math.isfinite(angle_same_c): angle_same_c = 1.0
            if not math.isfinite(angle_diff_c): angle_diff_c = 1.0
        results_c["same"].append(angle_same_c)
        results_c["diff"].append(angle_diff_c)

        # altprop
        model_f = MultiLayerSelfAttention(dim, num_layers=len(list(teacher)))
        model_f.load_state_dict(copy.deepcopy(student_initial_state))
        step_fixed(model_f, batch_x_a, batch_y_a, lr)
        with torch.no_grad():
            angle_same_f = compute_average_angle(model_f(batch_x_a), batch_y_a)
            angle_diff_f = compute_average_angle(model_f(batch_x_b), batch_y_b)
            if not math.isfinite(angle_same_f): angle_same_f = 1.0
            if not math.isfinite(angle_diff_f): angle_diff_f = 1.0
        results_f["same"].append(angle_same_f)
        results_f["diff"].append(angle_diff_f)

    # Find the largest learning rate that causes a decrease in angle after 1 step
    stable_lrs_c = [lr for lr, angle in zip(lrs, results_c["same"]) if angle < init_angle_a]
    stable_lrs_f = [lr for lr, angle in zip(lrs, results_f["same"]) if angle < init_angle_a]
    
    max_lr_c = max(stable_lrs_c) if stable_lrs_c else 1.0
    max_lr_f = max(stable_lrs_f) if stable_lrs_f else 1.0

    tuned_lr_c = max_lr_c / 2.0
    tuned_lr_f = max_lr_f / 2.0

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot backprop (blue tones, slightly transparent)
    ax.plot(lrs, results_c["same"], label='backprop - Same Batch (A)', color='#1f77b4', linewidth=1.5, alpha=0.7)
    ax.plot(lrs, results_c["diff"], label='backprop - Different Batch (B)', color='#1f77b4', linestyle='--', linewidth=1.0, alpha=0.6)
    
    # Plot altprop (orange/red tones, more solid)
    ax.plot(lrs, results_f["same"], label='altprop - Same Batch (A)', color='#ff7f0e', linewidth=1.5, alpha=0.9)
    ax.plot(lrs, results_f["diff"], label='altprop - Different Batch (B)', color='#ff7f0e', linestyle='--', linewidth=1.0, alpha=0.8)

    # Mark chosen learning rates with prominent dots on the same-batch curves
    y_tuned_c = np.interp(np.log10(tuned_lr_c), np.log10(lrs), results_c["same"])
    y_tuned_f = np.interp(np.log10(tuned_lr_f), np.log10(lrs), results_f["same"])
    
    ax.plot(tuned_lr_c, y_tuned_c, 'o', color='#004c8c', markersize=9, markeredgecolor='white', markeredgewidth=1.5, zorder=5, label=f'Chosen backprop LR ({tuned_lr_c:.3f})')
    ax.plot(tuned_lr_f, y_tuned_f, 'o', color='#b35900', markersize=9, markeredgecolor='white', markeredgewidth=1.5, zorder=5, label=f'Chosen altprop LR ({tuned_lr_f:.3f})')

    ax.set_xscale('log')
    ax.set_yscale('linear')
    ax.set_xlim(lrs[0], lrs[-1])
    
    # Set Y-axis to show the full range of normalized average angles (0.0 to 1.0)
    ax.set_ylim(0.0, 1.0)

    ax.set_xlabel('Learning Rate', fontsize=11)
    ax.set_ylabel(r'Average Angle (normalized by $\pi$ rad)', fontsize=11)
    ax.set_title('Initial LR Search: Angle Sensitivity & Selection', fontsize=12, fontweight='bold')
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=8, ncol=2)

    plt.tight_layout()
    plot_path = "reports/initial_lr_search.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved initial learning rate search plot to {plot_path}")
    print(f"-> Chosen LRs: backprop={tuned_lr_c:.3f}, altprop={tuned_lr_f:.3f}")
    return tuned_lr_c, tuned_lr_f

def generate_html_dashboard(num_layers, dim, seq_len, backprop_steps, altprop_steps, final_backprop_lr, final_altprop_lr, execution_time):
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Linear Transformer Reconstruction Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0b0f19;
            --card-bg: rgba(22, 30, 49, 0.7);
            --card-border: rgba(255, 255, 255, 0.08);
            --text-color: #f3f4f6;
            --text-muted: #9ca3af;
            --accent-primary: #3b82f6;
            --accent-secondary: #ff7f0e;
            --accent-success: #10b981;
            --glow-color: rgba(59, 130, 246, 0.15);
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            background-color: var(--bg-color);
            background-image: 
                radial-gradient(at 10% 20%, rgba(59, 130, 246, 0.1) 0px, transparent 50%),
                radial-gradient(at 90% 80%, rgba(16, 185, 129, 0.05) 0px, transparent 50%);
            background-attachment: fixed;
            color: var(--text-color);
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            min-height: 100vh;
            padding: 2rem;
            line-height: 1.5;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        header {{
            margin-bottom: 3rem;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 2rem;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            flex-wrap: wrap;
            gap: 1.5rem;
        }}

        .header-title h1 {{
            font-family: 'Outfit', sans-serif;
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #fff 30%, #a5b4fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.02em;
            margin-bottom: 0.5rem;
        }}

        .header-title p {{
            color: var(--text-muted);
            font-size: 1.1rem;
            font-weight: 300;
        }}

        .meta-badges {{
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
        }}

        .badge {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 0.75rem 1.25rem;
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        }}

        .badge-label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.25rem;
        }}

        .badge-value {{
            font-size: 1.2rem;
            font-weight: 700;
            color: var(--text-color);
            font-family: 'Outfit', sans-serif;
        }}

        .badge-value.backprop {{
            color: var(--accent-primary);
        }}

        .badge-value.altprop {{
            color: var(--accent-secondary);
        }}

        /* Dashboard Grid Layout */
        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(12, 1fr);
            gap: 2rem;
        }}

        .card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 1.75rem;
            backdrop-filter: blur(12px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.3s ease;
            position: relative;
            overflow: hidden;
        }}

        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 35px var(--glow-color);
            border-color: rgba(59, 130, 246, 0.2);
        }}

        .card-header {{
            margin-bottom: 1.25rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .card-title {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.3rem;
            font-weight: 600;
            letter-spacing: -0.01em;
        }}

        .card-description {{
            font-size: 0.875rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }}

        /* Dynamic Sizes for Cards */
        .card-full {{
            grid-column: span 12;
        }}

        .card-large {{
            grid-column: span 6;
        }}

        .card-third {{
            grid-column: span 4;
        }}

        @media (max-width: 1024px) {{
            .card-large, .card-third, .card-full {{
                grid-column: span 12;
            }}
        }}

        /* Image Display styling */
        .image-container {{
            width: 100%;
            border-radius: 12px;
            overflow: hidden;
            background: rgba(0,0,0,0.2);
            border: 1px solid rgba(255,255,255,0.05);
            display: flex;
            justify-content: center;
            align-items: center;
        }}

        .image-container img {{
            width: 100%;
            height: auto;
            display: block;
            object-fit: contain;
            transition: transform 0.5s ease;
        }}

        .image-container:hover img {{
            transform: scale(1.02);
        }}

        footer {{
            margin-top: 4rem;
            text-align: center;
            color: var(--text-muted);
            font-size: 0.9rem;
            border-top: 1px solid var(--card-border);
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-title">
                <h1>Linear Transformer Reconstruction Dashboard</h1>
                <p>Visualization and analysis of MultiLayerSelfAttention training runs with LinearSelfAttention and Least Squares Loss</p>
            </div>
            <div class="meta-badges">
                <div class="badge">
                    <span class="badge-label">Layers</span>
                    <span class="badge-value">{num_layers}</span>
                </div>
                <div class="badge">
                    <span class="badge-label">Dimension</span>
                    <span class="badge-value">{dim}</span>
                </div>
                <div class="badge">
                    <span class="badge-label">Seq Length</span>
                    <span class="badge-value">{seq_len}</span>
                </div>
                <div class="badge">
                    <span class="badge-label">backprop Convergence</span>
                    <span class="badge-value backprop">{backprop_steps}</span>
                </div>
                <div class="badge">
                    <span class="badge-label">altprop Convergence</span>
                    <span class="badge-value altprop">{altprop_steps}</span>
                </div>
                <div class="badge">
                    <span class="badge-label">Execution Time</span>
                    <span class="badge-value">{execution_time:.2f}s</span>
                </div>
            </div>
        </header>

        <main class="dashboard-grid">
            <!-- 1. Initial Learning Rate Search -->
            <section class="card card-full">
                <div class="card-header">
                    <div>
                        <h2 class="card-title">Initial Learning Rate Search</h2>
                        <p class="card-description">Shows the 1-step and 10-step MSE reconstruction loss over a grid of learning rates on a same batch vs a different batch.</p>
                    </div>
                </div>
                <div class="image-container">
                    <img src="initial_lr_search.png" alt="Initial Learning Rate Search Plot">
                </div>
            </section>

            <!-- 2. Reconstruction Error -->
            <section class="card card-large">
                <div class="card-header">
                    <div>
                        <h2 class="card-title">Reconstruction Error on New Batch</h2>
                        <p class="card-description">Evaluation MSE loss on a fixed independent validation batch across optimization steps.</p>
                    </div>
                </div>
                <div class="image-container">
                    <img src="reconstruction_error.png" alt="Reconstruction Error Plot">
                </div>
            </section>

            <!-- 3. Target Angles over Time -->
            <section class="card card-large">
                <div class="card-header">
                    <div>
                        <h2 class="card-title">Target Angles over Time</h2>
                        <p class="card-description">Average angle (normalized by &pi; radians) between the produced outputs and the desired targets on the evaluation batch.</p>
                    </div>
                </div>
                <div class="image-container">
                    <img src="target_angles_over_time.png" alt="Target Angles Plot">
                </div>
            </section>

            <!-- Layer 0 Weight Changes -->
            <div class="card-full" style="margin-top: 1.5rem; margin-bottom: -0.5rem;">
                <h2 style="font-family: 'Outfit', sans-serif; font-size: 1.6rem; font-weight: 700; border-bottom: 2px solid var(--accent-primary); padding-bottom: 0.4rem; display: inline-block;">Layer 0 Weight Changes</h2>
            </div>
            <section class="card card-third">
                <div class="card-header">
                    <div>
                        <h3 class="card-title">L0 Query (W<sub>q</sub>)</h3>
                        <p class="card-description">Frobenius norm distance and path length for layer 0 query weights.</p>
                    </div>
                </div>
                <div class="image-container">
                    <img src="weight_changes_L0_W_q.png" alt="Layer 0 Query Changes">
                </div>
            </section>
            <section class="card card-third">
                <div class="card-header">
                    <div>
                        <h3 class="card-title">L0 Key (W<sub>k</sub>)</h3>
                        <p class="card-description">Frobenius norm distance and path length for layer 0 key weights.</p>
                    </div>
                </div>
                <div class="image-container">
                    <img src="weight_changes_L0_W_k.png" alt="Layer 0 Key Changes">
                </div>
            </section>
            <section class="card card-third">
                <div class="card-header">
                    <div>
                        <h3 class="card-title">L0 Value (W<sub>v</sub>)</h3>
                        <p class="card-description">Frobenius norm distance and path length for layer 0 value weights.</p>
                    </div>
                </div>
                <div class="image-container">
                    <img src="weight_changes_L0_W_v.png" alt="Layer 0 Value Changes">
                </div>
            </section>

            <!-- Layer 1 Weight Changes -->
            <div class="card-full" style="margin-top: 1.5rem; margin-bottom: -0.5rem;">
                <h2 style="font-family: 'Outfit', sans-serif; font-size: 1.6rem; font-weight: 700; border-bottom: 2px solid var(--accent-primary); padding-bottom: 0.4rem; display: inline-block;">Layer 1 Weight Changes</h2>
            </div>
            <section class="card card-third">
                <div class="card-header">
                    <div>
                        <h3 class="card-title">L1 Query (W<sub>q</sub>)</h3>
                        <p class="card-description">Frobenius norm distance and path length for layer 1 query weights.</p>
                    </div>
                </div>
                <div class="image-container">
                    <img src="weight_changes_L1_W_q.png" alt="Layer 1 Query Changes">
                </div>
            </section>
            <section class="card card-third">
                <div class="card-header">
                    <div>
                        <h3 class="card-title">L1 Key (W<sub>k</sub>)</h3>
                        <p class="card-description">Frobenius norm distance and path length for layer 1 key weights.</p>
                    </div>
                </div>
                <div class="image-container">
                    <img src="weight_changes_L1_W_k.png" alt="Layer 1 Key Changes">
                </div>
            </section>
            <section class="card card-third">
                <div class="card-header">
                    <div>
                        <h3 class="card-title">L1 Value (W<sub>v</sub>)</h3>
                        <p class="card-description">Frobenius norm distance and path length for layer 1 value weights.</p>
                    </div>
                </div>
                <div class="image-container">
                    <img src="weight_changes_L1_W_v.png" alt="Layer 1 Value Changes">
                </div>
            </section>

            <!-- Layer 2 Weight Changes -->
            <div class="card-full" style="margin-top: 1.5rem; margin-bottom: -0.5rem;">
                <h2 style="font-family: 'Outfit', sans-serif; font-size: 1.6rem; font-weight: 700; border-bottom: 2px solid var(--accent-primary); padding-bottom: 0.4rem; display: inline-block;">Layer 2 Weight Changes</h2>
            </div>
            <section class="card card-third">
                <div class="card-header">
                    <div>
                        <h3 class="card-title">L2 Query (W<sub>q</sub>)</h3>
                        <p class="card-description">Frobenius norm distance and path length for layer 2 query weights.</p>
                    </div>
                </div>
                <div class="image-container">
                    <img src="weight_changes_L2_W_q.png" alt="Layer 2 Query Changes">
                </div>
            </section>
            <section class="card card-third">
                <div class="card-header">
                    <div>
                        <h3 class="card-title">L2 Key (W<sub>k</sub>)</h3>
                        <p class="card-description">Frobenius norm distance and path length for layer 2 key weights.</p>
                    </div>
                </div>
                <div class="image-container">
                    <img src="weight_changes_L2_W_k.png" alt="Layer 2 Key Changes">
                </div>
            </section>
            <section class="card card-third">
                <div class="card-header">
                    <div>
                        <h3 class="card-title">L2 Value (W<sub>v</sub>)</h3>
                        <p class="card-description">Frobenius norm distance and path length for layer 2 value weights.</p>
                    </div>
                </div>
                <div class="image-container">
                    <img src="weight_changes_L2_W_v.png" alt="Layer 2 Value Changes">
                </div>
            </section>

            <!-- 5. Dynamic Learning Rate over Time -->
            <section class="card card-large">
                <div class="card-header">
                    <div>
                        <h2 class="card-title">Dynamic Learning Rate over Time</h2>
                        <p class="card-description">Adjustment of the learning rate schedule over steps based on line-search / backtracking criteria.</p>
                    </div>
                </div>
                <div class="image-container">
                    <img src="dynamic_lr_over_time.png" alt="Dynamic Learning Rate Schedule Plot">
                </div>
            </section>
        </main>

        <footer>
            <p>Generated by Antigravity AI Code Assistant • Core Stack: PyTorch, NumPy, Matplotlib • {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </footer>
    </div>
</body>
</html>
"""
    output_path = "reports/linattention_report.html"
    with open(output_path, "w") as f:
        f.write(html_content)
    print(f"HTML dashboard generated successfully as {output_path}")

def train_and_visualize():
    start_time = time.time()
    dim, num_rows, batch_size, max_steps = 32, 8, 32, 10000

    print(f"Initializing Teacher Model (layers={NUM_LAYERS}, dim={dim}, seq_len={num_rows})...")
    teacher = MultiLayerSelfAttention(dim).requires_grad_(False)

    print("Initializing Student Models...")
    student_init = MultiLayerSelfAttention(dim)
    with torch.no_grad():
        for p in student_init.parameters():
            p.copy_(torch.eye(dim))
    student_initial_state = copy.deepcopy(student_init.state_dict())

    # Run initial learning rate search to determine training rates
    lr_c, lr_f = run_initial_lr_search(student_initial_state, teacher, dim, batch_size, num_rows)

    # Setup training models
    model_c = MultiLayerSelfAttention(dim)
    model_c.load_state_dict(copy.deepcopy(student_initial_state))
    model_f = MultiLayerSelfAttention(dim)
    model_f.load_state_dict(copy.deepcopy(student_initial_state))

    torch.manual_seed(100)
    eval_x = generate_orthogonal_rows_batch(batch_size, num_rows, dim)
    with torch.no_grad():
        eval_y = teacher(eval_x)
        start_eval_loss = torch.mean((model_c(eval_x) - eval_y) ** 2).item()
        
    target_eval_loss = start_eval_loss / LOSS_REDUCTION_FACTOR
    classic_reached = fixed_reached = None
    print(f"\nStopping criterion: both methods reach <= {target_eval_loss:.2e} "
          f"({LOSS_REDUCTION_FACTOR:.0f}x below start {start_eval_loss:.2e}).")

    # Track metrics for plotting
    class_loss_ind = [start_eval_loss]
    fixed_loss_ind = [start_eval_loss]
    classic_lr_history = [lr_c]
    fixed_lr_history = [lr_f]

    with torch.no_grad():
        init_angle_c = compute_average_angle(model_c(eval_x), eval_y)
        init_angle_f = compute_average_angle(model_f(eval_x), eval_y)
    class_angles_history = [init_angle_c]
    fixed_angles_history = [init_angle_f]

    # Track weight changes
    classic_init_params = {name: p.clone().detach() for name, p in model_c.named_parameters()}
    classic_prev_params = {name: p.clone().detach() for name, p in model_c.named_parameters()}
    classic_dist_to_start = {name: [0.0] for name in classic_init_params}
    classic_path_length = {name: [0.0] for name in classic_init_params}

    fixed_init_params = {name: p.clone().detach() for name, p in model_f.named_parameters()}
    fixed_prev_params = {name: p.clone().detach() for name, p in model_f.named_parameters()}
    fixed_dist_to_start = {name: [0.0] for name in fixed_init_params}
    fixed_path_length = {name: [0.0] for name in fixed_init_params}

    torch.manual_seed(42)
    final_step = max_steps
    
    for step in range(1, max_steps + 1):
        X = generate_orthogonal_rows_batch(batch_size, num_rows, dim)
        with torch.no_grad(): Y = teacher(X)

        lr_c_used = lr_c
        lr_c, loss_c = run_step_with_lr_tuning(model_c, step_classic, X, Y, lr_c, eval_x, eval_y, revert_on_fail=True)
        class_loss_ind.append(loss_c)
        classic_lr_history.append(lr_c_used)

        lr_f_used = lr_f
        lr_f, loss_f = run_step_with_lr_tuning(model_f, step_fixed, X, Y, lr_f, eval_x, eval_y, revert_on_fail=True)
        fixed_loss_ind.append(loss_f)
        fixed_lr_history.append(lr_f_used)

        with torch.no_grad():
            angle_c = compute_average_angle(model_c(eval_x), eval_y)
            angle_f = compute_average_angle(model_f(eval_x), eval_y)
        class_angles_history.append(angle_c)
        fixed_angles_history.append(angle_f)

        # Track weight updates
        with torch.no_grad():
            for name, p in model_c.named_parameters():
                dist_start = torch.norm(p - classic_init_params[name], p='fro').item()
                classic_dist_to_start[name].append(dist_start)
                step_dist = torch.norm(p - classic_prev_params[name], p='fro').item()
                classic_path_length[name].append(classic_path_length[name][-1] + step_dist)
                classic_prev_params[name].copy_(p)

            for name, p in model_f.named_parameters():
                dist_start = torch.norm(p - fixed_init_params[name], p='fro').item()
                fixed_dist_to_start[name].append(dist_start)
                step_dist = torch.norm(p - fixed_prev_params[name], p='fro').item()
                fixed_path_length[name].append(fixed_path_length[name][-1] + step_dist)
                fixed_prev_params[name].copy_(p)

        if classic_reached is None and loss_c <= target_eval_loss: classic_reached = step
        if fixed_reached is None and loss_f <= target_eval_loss: fixed_reached = step

        if step % 10 == 0 or step == 1:
            print(f"Step {step:3d} | backprop eval loss: {loss_c:.2e} (LR: {lr_c_used:.1f}) | altprop eval loss: {loss_f:.2e} (LR: {lr_f_used:.1f})")

        if classic_reached is not None and fixed_reached is not None:
            final_step = step
            print(f"Reached {LOSS_REDUCTION_FACTOR:.0f}x reduction: backprop at step {classic_reached}, altprop at step {fixed_reached}; stopping at step {step}.")
            break
    else:
        print(f"Reached max_steps={max_steps} before both hit target. backprop={classic_reached}, altprop={fixed_reached}")

    # Plot 1: Reconstruction Error
    fig1, ax1 = plt.subplots(figsize=(8, 5.5), dpi=150)
    plt.style.use('seaborn-v0_8-whitegrid')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.plot(class_loss_ind, label='backprop - Fixed Eval Batch', color='#1f77b4', linewidth=3.0)
    ax1.plot(fixed_loss_ind, label='altprop - Fixed Eval Batch', color='#ff7f0e', linewidth=1.5)
    ax1.axhline(target_eval_loss, color='#333333', linewidth=1.0, linestyle='--', label=f'{LOSS_REDUCTION_FACTOR:.0f}x target ({target_eval_loss:.1e})')
    ax1.set_yscale('log')
    ax1.set_title(f'Reconstruction Loss (MSE) on New Batch over Steps\nStopped at step {final_step} (backprop: {classic_reached or "N/A"}, altprop: {fixed_reached or "N/A"})', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Step', fontsize=11)
    ax1.set_ylabel('Mean Squared Error (MSE) Loss', fontsize=11)
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plt.savefig('reports/reconstruction_error.png')
    plt.close()

    # Plot 2: Dynamic Learning Rate
    fig2, ax2 = plt.subplots(figsize=(8, 5.5), dpi=150)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.plot(classic_lr_history, label='backprop Learning Rate', color='#1f77b4', linewidth=3.0)
    ax2.plot(fixed_lr_history, label='altprop Learning Rate', color='#ff7f0e', linewidth=1.5, linestyle='--')
    ax2.set_title(f'Dynamic Learning Rate Schedule over Steps\nFinal backprop LR={lr_c:.1f} | Final altprop LR={lr_f:.1f}', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Step', fontsize=11)
    ax2.set_ylabel('Learning Rate (LR)', fontsize=11)
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plt.savefig('reports/dynamic_lr_over_time.png')
    plt.close()

    # Plot 3: Target Angles over Time
    fig3, ax3 = plt.subplots(figsize=(8, 5.5), dpi=150)
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax3.plot(class_angles_history, label='backprop - Fixed Eval Batch', color='#1f77b4', linewidth=3.0)
    ax3.plot(fixed_angles_history, label='altprop - Fixed Eval Batch', color='#ff7f0e', linewidth=1.5)
    ax3.set_title(f'Average Angle Between Produced and Desired Targets\nStopped at step {final_step} (backprop: {classic_reached or "N/A"}, altprop: {fixed_reached or "N/A"})', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Step', fontsize=11)
    ax3.set_ylabel(r'Average Angle (normalized by $\pi$ rad)', fontsize=11)
    ax3.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plt.savefig('reports/target_angles_over_time.png')
    plt.close()

    # Plot 4: Split into 9 separate graphs grouped by layer
    for layer_idx in range(NUM_LAYERS):
        for w_name, w_label in [('W_q', 'Query Weight Matrix ($W_q$)'),
                                 ('W_k', 'Key Weight Matrix ($W_k$)'),
                                 ('W_v', 'Value Weight Matrix ($W_v$)')]:
            fig_w, ax_w = plt.subplots(figsize=(8, 5.5), dpi=150)
            plt.style.use('seaborn-v0_8-whitegrid')
            ax_w.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            param_name = f"{layer_idx}.{w_name}"
            
            # Plot backprop (blue tones)
            ax_w.plot(classic_dist_to_start[param_name], label='backprop - Distance to Start', color='#1f77b4', linestyle='-', linewidth=2.5)
            ax_w.plot(classic_path_length[param_name], label='backprop - Cumulative Path', color='#1f77b4', linestyle='--', linewidth=1.5)
            
            # Plot altprop (orange tones)
            ax_w.plot(fixed_dist_to_start[param_name], label='altprop - Distance to Start', color='#ff7f0e', linestyle='-', linewidth=2.5)
            ax_w.plot(fixed_path_length[param_name], label='altprop - Cumulative Path', color='#ff7f0e', linestyle='--', linewidth=1.5)
            
            ax_w.set_title(f'Layer {layer_idx} {w_label} Changes over Steps', fontsize=12, fontweight='bold')
            ax_w.set_xlabel('Step', fontsize=11)
            ax_w.set_ylabel('Frobenius Norm', fontsize=11)
            ax_w.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
            
            fig_w.tight_layout()
            plot_path = f"reports/weight_changes_L{layer_idx}_{w_name}.png"
            fig_w.savefig(plot_path)
            plt.close(fig_w)

    # Print stats
    print("\nWeight Matrix Statistics:")
    for name in classic_dist_to_start:
        print(f"  backprop {name} | Dist to Start: {classic_dist_to_start[name][-1]:.6f} | Path Length: {classic_path_length[name][-1]:.6f}")
    for name in fixed_dist_to_start:
        print(f"  altprop {name}   | Dist to Start: {fixed_dist_to_start[name][-1]:.6f} | Path Length: {fixed_path_length[name][-1]:.6f}")

    execution_time = time.time() - start_time
    print(f"\nEnd-to-end execution time: {execution_time:.3f} seconds")

    classic_steps_str = f"{classic_reached} steps" if classic_reached is not None else "Failed"
    fixed_steps_str = f"{fixed_reached} steps" if fixed_reached is not None else "Failed"
    
    generate_html_dashboard(
        num_layers=NUM_LAYERS,
        dim=dim,
        seq_len=num_rows,
        backprop_steps=classic_steps_str,
        altprop_steps=fixed_steps_str,
        final_backprop_lr=lr_c,
        final_altprop_lr=lr_f,
        execution_time=execution_time
    )

if __name__ == '__main__':
    train_and_visualize()
