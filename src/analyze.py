from util import *

def analyze(hyp, model, x_train, y_train, trained_epochs=None):
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
        data = pd.read_pickle(data_path)
        data = data.to_dict() # convert to dictionary
        for key in data:
            data[key] = list(data[key].values())
        existing_data = len(data[key])

    # trained_epochs (note: different from total_epochs!)
    if trained_epochs == None:
        trained_epochs = pd.NA

    # DO STUFF LIKE UNCERTAINTY GAP + EFFECTIVE DIMENSIONALITY HERE


    # add data to file
    cols = list(hyp.keys())
    cols.remove("total_epochs")
    cols += [
        "trained_epochs",
        "model_id"
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

    # create model id and make folders
    model_id = str(int(time.time() * 1e6) % int(1e13)) # get time for id
    this_model_path = models_path + model_id + ".obj"
    visualize_bases_path = models_path + model_id + "-bases"
    visualize_prior_path = models_path + model_id + "-prior"
    visualize_posterior_path = models_path + model_id + "-posterior"


    data["model_id"].append(model_id)
    model.set_id(model_id)


    # POTENTIALLY DO STUFF LIKE VISUALIZATIONS AND SAVING TO FILES HERE
    model.visualize_bases(x_train, y_train, savefig=visualize_bases_path)
    model.visualize_posterior_predictive(x_train, y_train, savefig=visualize_posterior_path)
    model.visualize_prior_predictive(x_train, y_train, savefig=visualize_prior_path)

    pprint(data)

    # save data
    df = pd.DataFrame(data)
    df.to_pickle(data_path, protocol=4)

    # save model to folder
    with open(this_model_path, "wb") as file_obj:
        pickle.dump(model, file_obj)
