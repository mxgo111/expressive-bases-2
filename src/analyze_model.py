from util import *

def analyze_model(hyp, model, x_train, y_train, trained_epochs=None, model_init_id=None):
    """
    Analyzes a model and adds a data point to a dataframe
    """

    # create experiments folder if not there
    experiment_path = "../experiments/" + hyp["experiment_name"]
    models_path = experiment_path + "/models/"

    os.makedirs(experiment_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)

    data_path = experiment_path + "/data.pkl"

    # if the data.pkl file exists, add to it
    # otherwise create a new file
    if len(glob.glob(data_path)) == 0:
        data = {}
        existing_data = 0
    else:
        try:
            data = pd.read_pickle(data_path)
            data = data.to_dict() # convert to dictionary
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
    uncertainty_area = get_uncertainty_in_gap(model.model, model.basis, x_train, y_train, n_points=hyp["posterior_prior_predictive_samples"])

    x_viz = np.linspace(hyp["dataset_min_range"], hyp["dataset_max_range"], hyp["num_points_linspace_visualize"])
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
        "uncertainty_area"
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

    # create model id and make folders
    model_id = get_unique_id()
    this_model_path = models_path + model_id + ".obj"
    visualize_bases_path = models_path + model_id + "-bases"
    visualize_prior_path = models_path + model_id + "-prior"
    visualize_posterior_path = models_path + model_id + "-posterior"


    data["model_init_id"].append(model_init_id)
    data["model_id"].append(model_id)
    model.set_id(model_id)


    # POTENTIALLY DO STUFF LIKE VISUALIZATIONS AND SAVING TO FILES HERE
    model.visualize_bases(x_train, y_train, savefig=visualize_bases_path)
    model.visualize_posterior_predictive(x_train, y_train, savefig=visualize_posterior_path)
    model.visualize_prior_predictive(x_train, y_train, savefig=visualize_prior_path)

    # save data
    df = pd.DataFrame(data)
    df.to_pickle(data_path, protocol=4)

    # save model to folder
    with open(this_model_path, "wb") as file_obj:
        pickle.dump(model, file_obj)