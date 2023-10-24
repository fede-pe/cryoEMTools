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

        histograms = []

        for file_path in file_list:
            # Connect to SQLite database
            conn = sqlite3.connect(file_path)
            cursor = conn.cursor()

            # Execute a SELECT query to retrieve columns c33 and c34
            cursor.execute("SELECT c33, c34 FROM Objects")

            # Fetch all rows
            rows = cursor.fetchall()

            # Print columns c33 and c34
            print(f"File: {file_path}")

            weakMisaliScore = []
            strongMisaliScore = []

            for row in rows:
                weakMisaliScore.append(row[1])
                strongMisaliScore.append(row[0])

            histograms.append(strongMisaliScore)
            histograms.append(weakMisaliScore)

            # Close the connection
            conn.close()

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # 2 rows, 3 columns

        self.generate_histogram(axes[0, 0], histograms[0], 50)
        self.generate_histogram(axes[1, 0], histograms[1], 50)
        self.generate_histogram(axes[0, 1], histograms[2], 50)
        self.generate_histogram(axes[1, 1], histograms[3], 50)
        self.generate_histogram(axes[0, 2], histograms[4], 50)
        self.generate_histogram(axes[1, 2], histograms[5], 50)

        plt.tight_layout()
        plt.savefig(self.out_figure)

    @staticmethod
    def generate_histogram(ax, data, bins):
        """ Method to generate histogram from scores vector """
        ax.hist(data, bins=bins, color='#000000', alpha=0.7)


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
