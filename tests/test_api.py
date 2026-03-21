from slurptuna import optimize
from slurptuna.example_losses import toy_conditions


def test_optimize_toy_loss_runs():
    result = optimize(toy_conditions, n_trials=2, seeds=[0, 1, 2], random_seed=7)
    assert result.loss_name == "toy_conditions"
    assert result.n_trials == 2
    assert isinstance(result.best_value, float)
    assert "alpha" in result.best_params
    assert "beta" in result.best_params
