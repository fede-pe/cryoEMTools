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

	double minRes = strtod(argv[2], nullptr);

	int xSize = XSIZE(tom);
	int ySize = YSIZE(tom);
	int zSize = ZSIZE(tom);

	#ifdef DEBUG
	std::cout << "Map dimensions: " << xSize << ", " << ySize << ", " << zSize << std::endl;
	#endif
	
	int xSize_half = xSize/2;
	int ySize_half = ySize/2;
	int zSize_half = zSize/2;

	size_t maxRadius = int(sqrt(xSize*xSize + ySize*ySize + zSize*zSize));

	#ifdef DEBUG
	std::cout << "Maximum radius: " << maxRadius << std::endl;
	#endif

	std::vector<double> radialResolution(maxRadius, 0);
	std::vector<double> radialCounter(maxRadius, 0);

	for (int i = -xSize_half; i < xSize_half; i++)
	{
		size_t i2 = i*i;

		for (int j = -ySize_half; j < ySize_half; j++)
		{
			size_t j2i2 = i2 + j*j;

			for (int k = -zSize_half; k < zSize_half; k++)
			{
				size_t r2 = j2i2 + k*k;
				
				double value = DIRECT_A3D_ELEM(tom, zSize_half + k, ySize_half + j, xSize_half + i);				

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
	
	for (size_t i = 0; i < radialResolution.size(); i++)
	{
		std::cout << radialResolution[i] << "\t" << i << std::endl;
	}

    return 0;
}