import optuna

# Load the study
study = optuna.load_study(
    study_name="fruitbit_quick",
    storage="sqlite:///optuna_studies/fruitbit_quick/optuna.db"
)

# Print results
print(f"Best trial: {study.best_trial.number}")
print(f"Best value: {study.best_value}")
print(f"Best params: {study.best_params}")

# Create visualization plots (saves as HTML files)
from optuna.visualization import plot_optimization_history, plot_param_importances
import plotly

fig1 = plot_optimization_history(study)
fig1.write_html("optimization_history.html")

fig2 = plot_param_importances(study)
fig2.write_html("param_importances.html")

print("Plots saved! Open .html files in browser.")