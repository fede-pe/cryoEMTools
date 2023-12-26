/***************************************************************************
 *
 * Authors:     Federico P. de Isidro-Gómez
 *
 * This program reads a CTF estimation from a ctfparam file and simulates it
 * over pure-noise micrograph.
 * 
 * To compile this standalone version run:
 * 
 *      scipion3 run xmipp_compile radialAverage3D.cpp 
 * 
 ***************************************************************************/


#include <iostream>
#include <core/xmipp_image.h>


int main(int argc, char **argv)
{
    FileName fnTomo=argv[1];

	Image<double> tomoMap;
	tomoMap.read(fnTomo);
	auto &tom = tomoMap();

	double minRes = argv[2];

	size_t xSize;
	size_t ySize;
	size_t zSize;
	tiltSeriesImages.getDimensions(xSize, ySize, zSize);
	
	size_t xSize_half = xSize/2;
	size_t ySize_half = ySize/2;
	size_t zSize_half = zSize/2;

	size_t maxRadius = std::max(xSize, ySize, zSize);

	std::vector<double> radialResolution(0, maxRadius);
	std::vector<double> radialCounter(0, maxRadius);

	for (int i = -xSize_half; i < xSize_half; i++)
	{
		size_t i2 = i*i;

		for (int j = -ySize_half; j < ySize_half; j++)
		{
			size_t j2i2 = i2 + j*j;

			for (int k = -zSize_half; k < zSize_half; k++)
			{
				size_t r2 = j2i2 + k*k;
				
				auto value = DIRECT_A3D_ELEM(tom, zSize_half + k, ySize_half + i, xSize_half + j);

				if (value < minRes)
				{
					auto radius = int(sqrt(r2));
					radialResolution[radius] += value;
					radialCounter[radius] += 1;
				}
			}
		}
	}

	for (size_t i = 0; i < radialCounter.size(); i++)
	{
		if (radialCounter[i] > 0)
		{
			radialResolution[i] /= radialCounter[i];
		}
	}
	
    return 0;
}