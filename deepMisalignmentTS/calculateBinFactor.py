import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python calculateBinFactor.py <beadDiameter> <tomoVoxelSize>\n"
              "<beadDiameter>:  Diameter in nanometers (nm) of the gold beads to be picked in the tomogram. \n"
              "<tomoVoxelSize>: Voxel size of the tomogram containing the beads to be picked in angstroms/pixel"
              "(A/px). /n")
        exit()

    targetBeadPixRadius = 10
    inputBeadRealDiameter = float(sys.argv[1])
    inputTomoBoxSize = float(sys.argv[2])

    # Calculate the diameter of the beads from the original tomogram in pixels
    # * 10 to nanometers -> angstroms
    # / 2 to diameter -> radius
    inputBeadPixRadius = (inputBeadRealDiameter * 10) / (2 * inputTomoBoxSize)

    # Calculate the binning factor to match the pixel size of the beads in the tomogram with target
    binFactor = targetBeadPixRadius / inputBeadPixRadius

    print("\n"
          "\t Binning factor: %.4f \n" % binFactor,
          "\t Target bead pixel size: %d pixels\n" % targetBeadPixSize)



