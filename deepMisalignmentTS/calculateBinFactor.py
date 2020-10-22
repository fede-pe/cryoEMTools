import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python calculateBinFactor.py <beadSize> <tomoBoxSize>\n"
              "<beadSize>:  Size in Angstroms (A) of the gold beads to be picked in the tomogram. \n"
              "<tomoBoxSize>: Path to file containing the tilt angles of the tilt-series. /n")
        exit()

    targetBeadPixSize = 10
    inputBeadRealSize = float(sys.argv[1])
    inputTomoBoxSize = float(sys.argv[2])

    # Calculate the size of the beads from the original tomogram in pixels
    inputBeadPixSize = inputBeadRealSize / inputTomoBoxSize

    # Calculate the binning factor to match the pixel size of the beads in the tomogram with target
    binFactor = targetBeadPixSize / inputBeadPixSize

    print("\n"
          "\t Binning factor: %.4f \n" % binFactor,
          "\t Target bead pixel size: %d pixels\n" % targetBeadPixSize)



