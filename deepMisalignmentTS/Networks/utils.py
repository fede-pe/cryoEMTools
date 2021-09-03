
""" Module containing functions to parse and process datasets and results. """


def normalizeInputDataStream(inputSubtomoStream):
    """ Method to normalize the input subtomo data stream to """
    std = inputSubtomoStream.std()
    mean = inputSubtomoStream.mean()

    normalizedInputDataStream = (inputSubtomoStream - mean) / std

    return normalizedInputDataStream


def produceClassesDistributionInfo(misalignmentInfoVector):
    """ This method output information of the classes distributions from the dataset between aligned and misaligned
    subtomos. """

    totalSubtomos = len(misalignmentInfoVector)
    numberOfMisalignedSubtomos = 0
    numberOfAlignedSubtomos = 0

    for subtomo in misalignmentInfoVector:
        if subtomo == 0:
            numberOfMisalignedSubtomos += 1
        elif subtomo == 1:
            numberOfAlignedSubtomos += 1

    print("\nClasses distribution:\n"
          "Aligned: %d (%.3f%%)\n"
          "Misaligned: %d (%.3f%%)\n\n"
          % (numberOfAlignedSubtomos, (numberOfAlignedSubtomos / totalSubtomos) * 100,
             numberOfMisalignedSubtomos, (numberOfMisalignedSubtomos / totalSubtomos) * 100))
