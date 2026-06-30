/***************************************************************************
 *
 * Authors:     Federico P. de Isidro-Gómez
 *
 * This program calculate the radial 2D average of a particle/micrograph.
 * 
 * To compile this standalone version run:
 * 
 *      scipion3 run xmipp_compile radialAverage3D.cpp 
 * 
 ***************************************************************************/


#include <iostream>
#include <core/xmipp_image.h>
#include "core/metadata_vec.h"


int main(int argc, char **argv)
{
    FileName fnImg=argv[1];

	Image<double> tomoMap;
	tomoMap.read(fnImg);
	auto &tom = tomoMap();


	int xSize = XSIZE(tom);
	int ySize = YSIZE(tom);

	#ifdef DEBUG
	std::cout << "Image dimensions: " << xSize << ", " << ySize << std::endl;
	#endif
	
	int xSize_half = xSize/2;
	int ySize_half = ySize/2;

	size_t maxRadius = int(std::max(xSize, ySize));

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
			size_t r2 = i2 + j*j;
			
			double value = DIRECT_A2D_ELEM(tom, ySize_half + j, xSize_half + i);				

			if (value < maxRadius)
			{
				auto radius = int(sqrt(r2));
				radialResolution[radius] += value;
				radialCounter[radius] += 1;
			}
		}
	}

	// Save output metadata
	MetaDataVec md;

	size_t lastindex = fnImg.find_last_of(".");
	std::string rawname = fnImg.substr(0, lastindex);
	
	size_t atIndex = rawname.find("@");
	if (atIndex != std::string::npos) {
		rawname = rawname.substr(atIndex + 1);
	}

	size_t id;

	for(size_t i = 0; i < radialCounter.size(); i++)
	{
		id = md.addObject();
		md.setValue(MDL_X, radialResolution[i] / radialCounter[i], id);
		md.setValue(MDL_Y, radialCounter[i],  id);
	}

	std::string outputMD = rawname + "_radial_profile.xmd";
	md.write(outputMD);

	std::cout << "Output metadata file generated at: " << outputMD << std::endl;

    return 0;
}