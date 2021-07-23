import sys
import os

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python translateCoordFile.py <inputFile> <outputFile>\n"
              "<inputFile>:  Input xmipp metadata (xmd) file containing 3D coordinates. \n"
              "<outputFile>: Location of the output coordinate file in IMOD format. \n")
        exit()

    inputFile = sys.argv[1]
    outputFile = sys.argv[2]

    with open(os.path.abspath(inputFile)) as f:
        inputLines = f.readlines()

    inputLines = inputLines[7:]

    outputLines = []

    for i, line in enumerate(inputLines):
        outputLine = " %d \t %s \t 1 \t %d\n" % (i+1, line[:-1], i+1)
        outputLines.append(outputLine)

    with open(os.path.abspath(outputFile), 'w') as f:
        f.writelines(outputLines)

