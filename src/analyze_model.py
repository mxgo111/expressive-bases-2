from util import *


def analyze_model(
    hyp, model, x_train, y_train, trained_epochs=None, model_init_id=None
):
    """
    Analyzes a model and adds a data point to a dataframe
    """

    # create experiments folder if not there
    experiment_path = "../experiments/" + hyp["experiment_name"]
    models_path = experiment_path + "/models/"

    os.makedirs(experiment_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)

    data_path = experiment_path + "/data.pkl"
    analyze_notebook_path = experiment_path + "/analyze_data.ipynb"

    # copy analysis notebook if it doesn't exist
    if not os.path.exists(analyze_notebook_path):
        shutil.copyfile("analyze_data_template.ipynb", analyze_notebook_path)

    # if the data.pkl file exists, add to it
    # otherwise create a new file
    if len(glob.glob(data_path)) == 0:
        data = {}
        existing_data = 0
    else:
        try:
            data = pd.read_pickle(data_path)
            data = data.to_dict()  # convert to dictionary
            for key in data:
                data[key] = list(data[key].values())
            existing_data = len(data[key])
        except:
            data = {}
            existing_data = 0

    # trained_epochs (note: different from total_epochs!)
    if trained_epochs == None:
        trained_epochs = pd.NA

    # DO STUFF LIKE UNCERTAINTY GAP + EFFECTIVE DIMENSIONALITY HERE
    uncertainty_area = get_uncertainty_in_gap(
        model.model,
        model.basis,
        x_train,
        y_train,
        n_points=hyp["posterior_prior_predictive_samples"],
    )
    posterior_contraction = model.model.posterior_contraction()
    try:
        uncertainty_area_var = var_of_posterior_predictive_var(
            model.model, model.basis, x_train, n_points=1000
        )
    except:
        uncertainty_area_var = None

    x_viz = np.linspace(
        hyp["dataset_min_range"],
        hyp["dataset_max_range"],
        hyp["num_points_linspace_visualize"],
    )
    basis_vals = model.basis(torch.tensor(x_viz.reshape(-1, 1)))
    eff_dim = compute_eff_dim(basis_vals)

    # add data to file (add any new information into cols)
    cols = list(hyp.keys())
    cols.remove("total_epochs")
    cols += [
        "model_init_id",
        "trained_epochs",
        "model_id",
        "eff_dim",
        "uncertainty_area",
        "uncertainty_area_var",
        "posterior_contraction",
    ]

    for col in cols:
        # column doesn't exist in data
        if col not in data:
            if existing_data > 0:
                data[col] = [pd.NA] * existing_data
            else:
                data[col] = []

        # append new data
        if col in hyp and col != "total_epochs":
            data[col].append(hyp[col])

    # if old hyp is no longer in use
    for key in data:
        if key not in cols:
            data[key].append(pd.NA)

    data["trained_epochs"].append(trained_epochs)
    data["eff_dim"].append(eff_dim)
    data["uncertainty_area"].append(uncertainty_area)
    data["posterior_contraction"].append(posterior_contraction)
    try:
        data["uncertainty_area_var"].append(np.var(uncertainty_area_var))
    except:
        data["uncertainty_area_var"].append(None)

    # create model id and make folders
    model_id = get_unique_id()
    this_model_path = models_path + model_id + ".obj"
    visualize_bases_path = models_path + model_id + "-bases"
    visualize_bases_weights_hist_path = models_path + model_id + "-bases-weights_hist"
    visualize_prior_path = models_path + model_id + "-prior"
    visualize_posterior_path = models_path + model_id + "-posterior"

    data["model_init_id"].append(model_init_id)
    data["model_id"].append(model_id)
    model.set_id(model_id)

    # POTENTIALLY DO STUFF LIKE VISUALIZATIONS AND SAVING TO FILES HERE
    # print(hyp["num_bases"])
    # print(model.model.posterior_mu.detach().cpu().numpy()[-2])
    if hyp["visualize_bases"]:
        model.visualize_bases(x_train, y_train, savefig=visualize_bases_path)
    model.visualize_posterior_predictive(
        x_train, y_train, savefig=visualize_posterior_path
    )
    model.visualize_posterior_weights_mean_hist(savefig = visualize_bases_weights_hist_path)
    model.visualize_prior_predictive(x_train, y_train, savefig=visualize_prior_path)

    # plot var of var to check
    visualize_varofvar_path = models_path + model_id + "-varofvar"
    try:
        plt.plot(uncertainty_area_var)
        plt.savefig(visualize_varofvar_path)
        plt.close()
    except:
        pass

    # save data
    df = pd.DataFrame(data)
    df.to_pickle(data_path, protocol=4)

    # save model to folder
    with open(this_model_path, "wb") as file_obj:
        pickle.dump(model, file_obj)


def analyze_gp_model(
    hyp, model, x_train, y_train, trained_epochs=None, model_init_id=None
):
    """
    Analyzes a model and adds a data point to a dataframe
    """

    # create experiments folder if not there
    experiment_path = "../experiments/" + hyp["experiment_name"]
    models_path = experiment_path + "/models/"

    os.makedirs(experiment_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)

    data_path = experiment_path + "/data.pkl"
    analyze_notebook_path = experiment_path + "/analyze_data.ipynb"

    if not os.path.exists(analyze_notebook_path):
        shutil.copyfile("analyze_data_template.ipynb", analyze_notebook_path)

    # if the data.pkl file exists, add to it
    # otherwise create a new file
    if len(glob.glob(data_path)) == 0:
        data = {}
        existing_data = 0
    else:
        try:
            data = pd.read_pickle(data_path)
            data = data.to_dict()  # convert to dictionary
            for key in data:
                data[key] = list(data[key].values())
            existing_data = len(data[key])
        except:
            data = {}
            existing_data = 0

    # trained_epochs (note: different from total_epochs!)
    if trained_epochs == None:
        trained_epochs = pd.NA

    # add data to file (add any new information into cols)
    cols = list(hyp.keys())
    cols.remove("total_epochs")
    cols += [
        "model_init_id",
        "trained_epochs",
        "model_id",
        "uncertainty_area_var",
        "uncertainty_area",
    ]

    for col in cols:
        # column doesn't exist in data
        if col not in data:
            if existing_data > 0:
                data[col] = [pd.NA] * existing_data
            else:
                data[col] = []

        # append new data
        if col in hyp and col != "total_epochs":
            data[col].append(hyp[col])

    # if old hyp is no longer in use
    for key in data:
        if key not in cols:
            data[key].append(pd.NA)

    data["trained_epochs"].append(trained_epochs)

    x_viz = np.linspace(
        hyp["dataset_min_range"],
        hyp["dataset_max_range"],
        hyp["num_points_linspace_visualize"],
    )

    x_gap = np.array(get_epistemic_gap(x_train))

    uncertainty_mean, uncertainty_var = model.get_uncertainty_area(x_gap)
    data["uncertainty_area"].append(uncertainty_mean)
    data["uncertainty_area_var"].append(uncertainty_var)
    data["model_init_id"].append(model_init_id)
    data["model_id"].append(model_init_id)

    visualize_gp_path = models_path + model_init_id + "-gp"
    model.visualize_uncertainty(x_train, y_train, savefig=visualize_gp_path)

    # print(data)
    df = pd.DataFrame(data)
    df.to_pickle(data_path, protocol=4)
