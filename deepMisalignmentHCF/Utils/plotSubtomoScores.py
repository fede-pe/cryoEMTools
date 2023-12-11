import argparse
import glob
import os
import sqlite3
import matplotlib.pyplot as plt
import numpy as np


class PlotSubtomoScores:
    def __init__(self, input_regex, out_figure):
        self.input_regex = input_regex
        self.out_figure = out_figure
        # Additional initialization code if needed

    def plot_scores(self):
        # Add code for plotting scores based on self.input_regex
        print(f"Plotting scores for SQLite files matching regex: {self.input_regex}")

    def read_scores(self):
        # Use glob to find files matching the input regex
        file_list = glob.glob(self.input_regex)

        Ali_strong_histogram = []
        Ali_weak_histogram = []
        WM_strong_histogram = []
        WM_weak_histogram = []
        SM_strong_histogram = []
        SM_weak_histogram = []

        for file_path in file_list:
            print("Reading sqlite: %s" % file_path)

            # Connect to SQLite database
            conn = sqlite3.connect(file_path)
            cursor = conn.cursor()

            # Get column name for properties '_strongMisaliScore', '_weakMisaliScore'
            label_property_values = ('_strongMisaliScore', '_weakMisaliScore')
            placeholders = ', '.join(['?'] * len(label_property_values))

            cursor.execute(f"SELECT column_name FROM Classes WHERE label_property IN ({placeholders})",
                           label_property_values)

            rows = cursor.fetchall()  # Fetch all rows
            column_names = (rows[0][0], rows[1][0])

            # Use columns names to extract misalignment score values
            cursor.execute(f"SELECT {', '.join(column_names)} FROM Objects")

            rows = cursor.fetchall()  # Fetch all rows from the result set

            weakMisaliScore = []
            strongMisaliScore = []

            for row in rows:
                weakMisaliScore.append(row[1])
                strongMisaliScore.append(row[0])

            # Detect population group
            group = os.path.basename(file_path).split("-")[0]

            if group == "Ali":
                Ali_strong_histogram.extend(strongMisaliScore)
                Ali_weak_histogram.extend(weakMisaliScore)
            elif group == "WM":
                WM_strong_histogram.extend(strongMisaliScore)
                WM_weak_histogram.extend(weakMisaliScore)
            elif group == "SM":
                SM_strong_histogram.extend(strongMisaliScore)
                SM_weak_histogram.extend(weakMisaliScore)
            else:
                raise "ERROR: unrecognized alignment group"

            # Close the connection
            conn.close()

        print("-- Total number of subtomos")
        print("Aligned")
        print("\tStrong misalignment score: %d" % len(Ali_strong_histogram))
        print("\tWeak misalignment score: %d" % len(Ali_weak_histogram))
        print("WM")
        print("\tStrong misalignment score: %d" % len(WM_strong_histogram))
        print("\tWeak misalignment score: %d" % len(WM_weak_histogram))
        print("SM")
        print("\tStrong misalignment score: %d" % len(SM_strong_histogram))
        print("\tWeak misalignment score: %d" % len(SM_weak_histogram))

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # 2 rows, 3 columns

        self.generate_histogram(axes[0, 0], SM_strong_histogram, 50, "Strong misalignment")
        self.generate_histogram(axes[1, 0], SM_weak_histogram, 50, "Weak misalignment")
        self.generate_histogram(axes[0, 1], WM_strong_histogram, 50, "Strong misalignment")
        self.generate_histogram(axes[1, 1], WM_weak_histogram, 50, "Weak misalignment")
        self.generate_histogram(axes[0, 2], Ali_strong_histogram, 50, "Strong misalignment")
        self.generate_histogram(axes[1, 2], Ali_weak_histogram, 50, "Weak misalignment")

        plt.tight_layout()
        plt.savefig(self.out_figure)

        self.calculateF1score(WM_weak_histogram, Ali_weak_histogram)

    @staticmethod
    def calculateF1score(v1, v2):
        """ Method to calculate odds ration between to vectors """
        v1 = sorted(v1)
        v2 = sorted(v2)

        lenV1 = len(v1)
        lenV2 = len(v2)

        step_size = 0.001
        values = np.arange(step_size, 1 - step_size, step_size)
        # values = [0.5]

        maxOddsRatio = 0
        maxI = 0

        for i in values:
            countV1 = 1
            countV2 = 1

            for element in v1:
                if element < i:
                    countV1 += 1
                else:
                    break

            for element in v2:
                if element < i:
                    countV2 += 1
                else:
                    break

            oddsRatio = (2 * (lenV2 - countV2)) / (2 * (lenV2 - countV2) + countV2 + (lenV1 - countV1))

            if i == 0.5:
                print("sensitivity at 0.5 %f" % ((lenV2 - countV2) / lenV2))
                print("specificity at 0.5 %f" % (countV1 / lenV1))
                print("F1 score at 0.5 %f" % (
                        (2 * (lenV2 - countV2)) / (2 * (lenV2 - countV2) + countV2 + (lenV1 - countV1))))

            if oddsRatio > maxOddsRatio:
                maxOddsRatio = oddsRatio
                maxI = i

        print("maximun odds ratio %f, for threshold %f" % (maxOddsRatio, maxI))

    @staticmethod
    def generate_histogram(ax, data, bins, label):
        """ Method to generate histogram from scores vector """
        ax.hist(data, bins=bins, color='#000000', alpha=0.7)
        ax.set_xlabel(label)


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Analyze SQLite files using a regex.")

    # Add argument for input SQLite regex
    parser.add_argument('--inputSqlite', dest='input_regex', required=True,
                        help='Input regex for the SQLite files to be analyzed.')

    # Add argument for output image path
    parser.add_argument('--outFigure', dest='out_figure', required=True,
                        help='Path to save generated image.')

    # Parse command line arguments
    args = parser.parse_args()

    # Create an instance of PlotSubtomoScores with the provided regex
    plotter = PlotSubtomoScores(args.input_regex,
                                args.out_figure)

    # Perform plotting or other operations based on the provided regex
    plotter.read_scores()


if __name__ == "__main__":
    main()
