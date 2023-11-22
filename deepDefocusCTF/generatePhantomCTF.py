import argparse
import os
import re
import glob
from random import choice, uniform


class GeneratorPhantomCTF:
    def __init__(self):
        self.input_files = []
        self.number_ctf = None
        self.min_defocus = 0
        self.max_defocus = 0
        self.astigmatism = 0
        self.output_location = ""

        # CTF parameters
        self.fields = ['_ctfSamplingRate', '_ctfVoltage', '_ctfDefocusU', '_ctfSphericalAberration',
                       '_ctfChromaticAberration', '_ctfEnergyLoss', '_ctfLensStability', '_ctfConvergenceCone',
                       '_ctfLongitudinalDisplacement', '_ctfTransversalDisplacement', '_ctfQ0', '_ctfK', '_ctfEnvR0',
                       '_ctfEnvR1', '_ctfEnvR2', '_ctfBgGaussianK', '_ctfBgGaussianSigmaU', '_ctfBgGaussianCU',
                       '_ctfBgSqrtK', '_ctfBgSqrtU', '_ctfBgBaseline', '_ctfBgGaussian2K', '_ctfBgGaussian2SigmaU',
                       '_ctfBgGaussian2CU', '_ctfBgR1', '_ctfBgR2', '_ctfBgR3', '_ctfDefocusV', '_ctfDefocusAngle',
                       '_ctfBgGaussianSigmaV', '_ctfBgGaussianCV', '_ctfBgGaussianAngle', '_ctfBgSqrtV',
                       '_ctfBgSqrtAngle', '_ctfBgGaussian2SigmaV', '_ctfBgGaussian2CV', '_ctfBgGaussian2Angle',
                       '_ctfX0', '_ctfXF', '_ctfY0', '_ctfYF', '_ctfCritFitting', '_ctfCritCorr13', '_ctfCritIceness',
                       '_CtfDownsampleFactor', '_ctfCritPsdStdQ', '_ctfCritPsdPCA1', '_ctfCritPsdPCARuns']

        self.paramsDict = {key: [] for key in self.fields}

        self.readInput()
        self.processParamsFiles()
        self.generatePhantomFiles()

    def readInput(self):
        parser = argparse.ArgumentParser(description="Generate new CTFs based on a characterized population.")

        # Add the inputFiles parameter with type=str
        parser.add_argument(
            '--inputFiles',
            required=True,
            help='Input CTF parameter files to characterize the population from which new CTFs will be generated. '
                 'Accepts a regex.'
        )

        # Add the outputLocation parameter
        parser.add_argument(
            '--outputLocation',
            required=True,
            help='Output location to save the generated files.'
        )

        # Add the numberCTF parameter
        parser.add_argument(
            '--numberCTF',
            type=int,
            required=True,
            help='Number of phantoms CTFs to be generated.'
        )

        # Add the minDefocus parameter
        parser.add_argument(
            '--minDefocus',
            type=int,
            required=True,
            help='Minimum defocus value.'
        )

        # Add the maxDefocus parameter
        parser.add_argument(
            '--maxDefocus',
            type=int,
            required=True,
            help='Maximum defocus value.'
        )

        # Add the astigmatism parameter
        parser.add_argument(
            '--astigmatism',
            type=float,
            required=True,
            help='Maximum astigmatism value (as a factor of the given defocus).'
        )

        # Parse the command line arguments
        args = parser.parse_args()

        # Access the values using args.inputFiles, args.numberCTF, args.minDefocus, args.maxDefocus, args.astigmatism
        self.input_files = glob.glob(args.inputFiles)
        self.number_ctf = args.numberCTF
        self.min_defocus = args.minDefocus
        self.max_defocus = args.maxDefocus
        self.astigmatism = args.astigmatism
        self.output_location = args.outputLocation

        # Use the re module to work with the regex
        try:
            compiled_regex = re.compile(args.inputFiles)
        except re.error:
            print("Invalid regular expression for inputFiles.")
            return False

        return True

    def processParamsFiles(self):
        for f in self.input_files:
            with open(f, 'r') as file:
                # Read each line from the file
                for line in file:
                    # Split the line into words
                    words = line.split()

                    # Check if the first word is in self.fields
                    if words and words[0] in self.fields:
                        self.paramsDict[words[0]].append(float(words[1]))

        print("%d ctfparam files read successfully!" % len(self.input_files))

    def generatePhantomFiles(self):
        for i in range(self.number_ctf):
            random_dict = {key: choice(values) for key, values in self.paramsDict.items()}

            file_path = os.path.join(self.output_location, "simulated_ctf_%d.ctfparam" % i)

            # Generate random defocus values
            defocusU = uniform(self.min_defocus, self.max_defocus)
            absoluteAstimatism = self.astigmatism * defocusU

            random_sign = choice([1, -1])
            defocusV = defocusU + (absoluteAstimatism * random_sign)

            defocusU, defocusV = (defocusU, defocusV) if defocusU >= defocusV else (defocusV, defocusU)

            # Generate random defocus angle
            defocusAngle = uniform(0, 180)

            # Update dict
            random_dict['_ctfDefocusU'] = defocusU
            random_dict['_ctfDefocusV'] = defocusV
            random_dict['_ctfDefocusAngle'] = defocusAngle

            with open(file_path, 'w') as file:
                # Write the header
                file.write("# XMIPP_STAR_1 *\n"
                           "#")

                for key, values in random_dict.items():
                    line = f"{key} {values}\n"
                    file.write(line)

        print("%d phantom ctfparam files generated successfully!" % self.number_ctf)


if __name__ == '__main__':
    generator = GeneratorPhantomCTF()
