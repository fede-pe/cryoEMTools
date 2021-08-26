
""" Module containing functions to parse and process datasets and results. """


def getTitleFromVariable(variable):
    """ This method returns the title (variable name) from the input variable identifier. """

    if variable == 0:
        title = "Centroid X"
    elif variable == 1:
        title = "Centroid Y"
    elif variable == 2:
        title = "Max distance"
    elif variable == 3:
        title = "Total distance"
    elif variable == 4:
        title = "Hull area"
    elif variable == 5:
        title = "Hull perimeter"
    elif variable == 6:
        title = "PCA X"
    elif variable == 7:
        title = "PCA Y"
    else:
        raise Exception("Variable %d code is out of range" % variable)

    return title

