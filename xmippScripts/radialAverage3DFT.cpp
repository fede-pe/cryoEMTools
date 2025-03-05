/***************************************************************************
 *
 * Authors:     Federico P. de Isidro-Gómez
 *
 * This program calculate the radial 3D average of the Fourier transfrom of 
 * a volume.
 * 
 * To compile this standalone version run:
 * 
 *      scipion3 run xmipp_compile radialAverage3DFT.cpp 
 * 
 ***************************************************************************/


#include <iostream>
#include <core/xmipp_image.h>
#include <algorithm>


int main(int argc, char **argv)
{
    FileName fnVol=argv[1];

	Image<double> volMap;
	volMap.read(fnVol);

	FourierTransformer ft;

	MultidimArray< std::complex<double> > fftVol;
	ft.FourierTransform(volMap(), fftVol, false);


	int xSize = XSIZE(fftVol);
	int ySize = YSIZE(fftVol);
	int zSize = ZSIZE(fftVol);

	#ifdef DEBUG
	std::cout << "Map dimensions: " << xSize << ", " << ySize << ", " << zSize << std::endl;
	#endif
	
	int xSize_half = xSize/2;
	int ySize_half = ySize/2;
	int zSize_half = zSize/2;

	int maxRadius = std::max(xSize, std::max(ySize, zSize));

	std::cout << "xSize " << xSize << std::endl;
	std::cout << "ySize " << ySize << std::endl;
	std::cout << "zSize " << zSize << std::endl;
	std::cout << "maxRadius " << maxRadius << std::endl;

	#ifdef DEBUG
	std::cout << "Maximum radius: " << maxRadius << std::endl;
	#endif

	std::vector<double> radialAvg(maxRadius, 0);
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
				
				double value = DIRECT_A3D_ELEM(volMap(), zSize_half + k, ySize_half + j, xSize_half + i);				

				if (value < minRes)
				{
					auto radius = int(sqrt(r2));
					radialAvg[radius] += value;
					radialCounter[radius] += 1;
				}
			}
		}
	}

	for (size_t i = 0; i < radialCounter.size(); i++)
	{
		if (radialCounter[i] > 0)
		{
			radialAvg[i] /= radialCounter[i];
		}
	}
	
	for (size_t i = 0; i < radialAvg.size(); i++)
	{
		std::cout << radialAvg[i] << "\t" << i << std::endl;
	}

    return 0;
}