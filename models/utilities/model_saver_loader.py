import json


def Load_Model(file_name, path=''):
    """ load a model from a trained model file in json format

    Parameters:

    (Type: string) file_name : ----> name of the modal to load

    Return:
    (Type: model name, features, rows,split ratio, model parameters)

    """

    model_file_full_name = path + '/' + file_name
    with open(model_file_full_name, "r") as model_file:
        data = json.load(model_file)
        # print(data)
        return data


def Save_Model(model_name, training_data_file_name, no_features, no_row, split_ratio, trained_model_value, path=''):
    """  save a trained model in a file (json format)

    Parameters:
    (Type: string) model_name : ----> model file name
    (Type: string) training_data_file_name : ----> name of the data file used for training
    (Type: int) no_features : ----> no fo feature vector ing data used for training
    (Type: integer) no_row -----> len of te=he dataset
    (Type: flot) split_ratio -----> data splitting used to split data
    (Type: dictionary/model parameters) trained_model_value

    Return:
    (Type: NONE) does not return anything

    """

    model_data = {
        "model": model_name,
        "training_file_name": training_data_file_name,
        "no_features": no_features,
        "no_of_rows": no_row,
        "split_ratio": split_ratio,
        "trained_model_value": trained_model_value,
    }
    # print(model_data)
    # path="Saved_models/"

    model_file_name = model_name + str(".json")
    model_file_full_name = path + '/' + model_file_name

    try:
        with open(model_file_full_name, 'x') as modelfile:
            json.dump(model_data, modelfile, indent=4)
            print('''model_saved sucessfully in file named "{}" at location "{}"'''.format(model_file_name, path))
    except:
        with open(model_file_full_name, 'w') as modelfile:
            json.dump(model_data, modelfile, indent=4)
            print('''model_saved sucessfully in file named "{}" at location "{}"'''.format(model_file_name, path))

    return


if __name__ == "__main__":
    Save_Model("decisiontree", "jkl.txt", 14, 14, 0.7, "no values")
