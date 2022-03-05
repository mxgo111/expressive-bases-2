from util import *

def analyze(hyp, model, x_train, y_train, trained_epochs=None):
    """
    Analyzes a model and adds a data point to a dataframe
    """

    # create experiments folder if not there
    experiment_path = "../experiments/" + hyp["experiment_name"]
    os.makedirs(experiment_path, exist_ok=True)
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
        existing_data = len(list(data.keys()))

    # trained_epochs (note: different from total_epochs!)
    if trained_epochs == None:
        trained_epochs = pd.NA

    # DO STUFF LIKE UNCERTAINTY GAP + EFFECTIVE DIMENSIONALITY HERE
    #
    #
    #
    #
    #
    #
    #

    # add data to file
    cols = list(hyp.keys())
    cols.remove("total_epochs")
    cols += [
        "trained_epochs"
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

    data["trained_epochs"].append(trained_epochs)

    # save data
    df = pd.DataFrame(data)
    df.to_pickle(data_path, protocol=4)
