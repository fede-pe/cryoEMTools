""" Module to compose a numpy array with all the .mrc files contained in a regex """
import argparse


def composeVector(subtomoRegex, outputLocation):
    """ Method to compose a numpy vector saved at outputLocation with all the subtomos
    indicated by  subtomoRegex. """


# ----------------------------------- Main ------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process subtomograms and save composed vector.')

    # Add the command line arguments
    parser.add_argument('--subtomoRegex', '-i', required=True,
                        help='Regex indicating the location of input subtomograms.')
    parser.add_argument('--output', '-o', required=True, help='Location where the composed vector will be saved.')

    # Parse the command line arguments
    args = parser.parse_args()

    # Access the values
    subtomoRegex = args.subtomoRegex
    outputLocation = args.output

    # Now you can use subtomo_regex and output_location in your script
    composeVector(subtomoRegex, outputLocation)
