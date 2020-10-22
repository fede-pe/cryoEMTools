if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python calculateBinFactor.py <beadSize> <tomoBoxSize>\n"
              "<beadSize>:  Size in Angstroms (A) of the gold beads to be picked in the tomogram. \n"
              "<pathAngleFile>: Path to file containing the tilt angles of the tilt-series. /n")
        exit()

    targetBeadPixSize = 10
    inputBeadRealSize = sys.argv[1]
    inputTomoBoxSize = sys.argv[2]

    # Calculate the size of the beads from the original tomogram in pixels
    inputBeadPixSize = inputBeadRealSize / inputTomoBoxSize



