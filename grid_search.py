import graphweave, reconstruct, stats
import numpy as np
import torch
import pandas as pd
import os
from itertools import product

def run_experiment(c=3, alpha=0.9, k=10, seed=0, num_graphs=200, test_frac=0.5, num_generated_graphs=2, num_nodes=100, use_train_degrees=False, lr=1e-2, epochs=100, graph_type='SBM'):
    torch.manual_seed(seed)
    deg_powers = [0, 1.0]

    df, all_M, all_L, all_A = graphweave.create_graphs_df(
        num_nodes=num_nodes,
        num_graphs=num_graphs,
        num_steps=k,
        seed=seed,
        use_norm_lap=True,
        norm_lap_alpha=alpha,
        deg_powers=deg_powers,
        graph_type=graph_type,
        norm_start_vec=True
    )

    Q, Qc, min_val, m, s = graphweave.get_Q_from_df(df, cat_mult=c)

    Q_train, Q_test, Qc_train, Qc_test, all_A_train, all_A_test = graphweave.train_test_split(Q, Qc, all_A, test_frac=test_frac)

    model, num_categories = graphweave.do_train_vals(Q_train, Qc_train, learning_rate=lr, num_epochs=epochs)

    all_generated_graphs = []
    for i in range(len(all_A_test)):
        degrees = all_A_test[i].todense().sum(axis=1) if not use_train_degrees else all_A_train[i].todense().sum(axis=1)
        V_t, V_tplus1 = graphweave.generate_graph_from_degrees(
            model=model,
            degrees=degrees,
            deg_powers=deg_powers,
            alpha=alpha,
            m=m,
            s=s,
            cat_mult=c,
            min_val=min_val,
            num_categories=num_categories,
            num_steps=k,
            norm_start_vec=True
        )
        try:
            A_pred = reconstruct.reconstruct_G_using_D(V_t, V_tplus1, alpha=alpha, degrees=degrees, max_time=240)
            A_pred = reconstruct.pick_threshold5(A_pred, degrees)
        except:
            A_pred = None
        all_generated_graphs.append(A_pred)

    res_df, all_stats_true, all_stats_methods = stats.compare_all(all_A_test, all_generated_graphs, None)

    return res_df, all_stats_true, all_stats_methods

def grid_search_for_graph_type(graph_type='SBM', c_values=[1, 3, 5], alpha_values=[0.5, 0.7, 0.9, 0.99], k_values=[5, 10, 20], **run_kwargs):
    results = []
    for c, alpha, k in product(c_values, alpha_values, k_values):
        print(f"[{graph_type}] Running c={c}, alpha={alpha}, k={k}")
        try:
            res_df, _, _ = run_experiment(c=c, alpha=alpha, k=k, graph_type=graph_type, **run_kwargs)
            # score = res_df.loc['degree', 'inter GraphWeave']
            score = res_df['ratio GraphWeave'].values.tolist()
        except Exception as e:
            score = None
            print(f"Failed for c={c}, alpha={alpha}, k={k}: {e}")
        results.append({"c": c, "alpha": alpha, "k": k, "score": score})

    df = pd.DataFrame(results)
    df.to_csv(f"grid_search_results_{graph_type}.csv", index=False)
    print(df.sort_values(by="score", ascending=False))
    return df

def run_and_save_best_for_graph_type(graph_type='SBM', num_generated_graphs=40, use_train_degrees=True, **run_kwargs):
    df_results = grid_search_for_graph_type(graph_type=graph_type, **run_kwargs)
    best_row = df_results.sort_values(by="score", ascending=False).iloc[0]
    best_c = int(best_row['c'])
    best_alpha = float(best_row['alpha'])
    best_k = int(best_row['k'])

    print(f"[{graph_type}] Running best config: c={best_c}, alpha={best_alpha}, k={best_k}")
    try:
        res_df, all_stats_true, all_stats_methods = run_experiment(
            c=best_c, alpha=best_alpha, k=best_k,
            num_generated_graphs=num_generated_graphs,
            use_train_degrees=use_train_degrees,
            graph_type=graph_type,
            **run_kwargs
        )
        full_metrics = pd.concat({
            'inter GraphWeave': res_df.iloc[:,0],
            'ratio GraphWeave': res_df.iloc[:,1],
            'true': res_df.iloc[:,2]
        }, axis=1)
        full_metrics.to_csv(f"final_metrics_{graph_type}.csv")
        print(f"Saved final metrics for {graph_type}")
    except Exception as e:
        print(f"Failed final run for {graph_type}: {e}")

# def run_and_save_grid_search_all(methods=['SBM', 'WS', 'BA', 'RGlikeSBM', 'Planar'], **run_kwargs):
run_and_save_best_for_graph_type(
    graph_type = 'SBM',
    num_generated_graphs=40,
    use_train_degrees=True,
    seed=0,
    test_frac=0.5
)
        

