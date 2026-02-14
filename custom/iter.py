import itertools

# Sweep parameters
costs = []
for lambda_1, lambda_pi in tqdm.tqdm(itertools.product(np.logspace(3, 6, base=10, num=5), np.logspace(-5, 6, base=10, num=5)), total=25):
    print(f"running for l1: {lambda_1}, lpi: {lambda_pi}")
    controller_args.controller_costs.regularizer_cost_g_1 = lambda_1
    controller_args.controller_costs.regularizer_cost_g_pi = lambda_pi
    select_deepc = SelectDeePC(
        controller_args,
        LkSelector(
            1,
            controller_args.deepc_dims,
            custom_callback=selector_cb,
            forgetting_factor=0.8
        ),
        num_hankel_cols=controller_args.deepc_dims.T_fut*controller_args.deepc_dims.m + 30,
        n_iter=1,
        debug=False,
    )
    traj_select_deepc, preds_select_deepc, g_vals, cost, _ = run_experiment(
        select_deepc,
        f"select-deepc-sweep-{int(lambda_1)}-{int(lambda_pi)}",
        DeePCCostAccumulator(controller_args.controller_costs)
    )
    print(cost)
    costs.append(cost)

costs_val = [val["cost"] for val in costs]
costs_val = np.array(costs_val)
# costs_val = np.log(costs_val)
fig, ax = plt.subplots(1, 1)
y, x = np.meshgrid(
    np.logspace(-5, 6, base=10, num=5), np.logspace(3, 6, base=10, num=5), indexing="ij"
)
im = ax.pcolormesh(
    x,
    y,
    costs_val.reshape(5, 5),
    # shading="gouraud",
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e3, 1e6)
ax.set_ylim(1e-5, 1e6)
plt.colorbar(im, label=r"cost")
plt.ylabel(r"$\lambda_\Pi$")
plt.xlabel(r"$\lambda_1$")
