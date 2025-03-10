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
#include "core/xmipp_fftw.h"
#include "core/metadata_vec.h"


int main(int argc, char **argv)
{
	// Generate side info
    FileName fnVol=argv[1];

	Image<double> volMap;
	volMap.read(fnVol);

	// Calculate FT
	FourierTransformer ft;
	MultidimArray< std::complex<double> > fftVol;
	ft.FourierTransform(volMap(), fftVol, false);

	// FT dimensions
	int xSize = XSIZE(fftVol);
	int ySize = YSIZE(fftVol);
	int zSize = ZSIZE(fftVol);
	int xSize_half = xSize/2;
	int ySize_half = ySize/2;
	int zSize_half = zSize/2;
	int maxRadius = (3 * std::max(xSize_half, std::max(ySize_half, zSize_half)) / 2) + 1;

	std::cout << "Map dimensions: " << std::endl;  
	std::cout << "xSize " << xSize << std::endl;
	std::cout << "ySize " << ySize << std::endl;
	std::cout << "zSize " << zSize << std::endl;
	std::cout << "maxRadius " << maxRadius << std::endl;

	// Compute real and magnitude FT maps
	MultidimArray<double> fftVol_real;
	fftVol_real.initZeros(fftVol);
	MultidimArray<double> fftVol_mod;
	fftVol_mod.initZeros(fftVol);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(fftVol)
	{
		DIRECT_MULTIDIM_ELEM(fftVol_real,n) += DIRECT_MULTIDIM_ELEM(fftVol,n).real();
		DIRECT_MULTIDIM_ELEM(fftVol_mod,n) += (DIRECT_MULTIDIM_ELEM(fftVol,n) * std::conj(DIRECT_MULTIDIM_ELEM(fftVol,n))).real();
	}

	// Save FT maps
	size_t lastindex = fnVol.find_last_of(".");
	std::string rawname = fnVol.substr(0, lastindex);
	Image<double> saveImage;

	std::string debugFileFn = rawname + "_FT_real.mrc";
	saveImage() = fftVol_real;
	saveImage.write(debugFileFn);

	debugFileFn = rawname + "_FT_mod.mrc";
	saveImage() = fftVol_mod;
	saveImage.write(debugFileFn);

	// Calculate radial average
	std::vector<double> radialAvg_real(maxRadius, 0);
	std::vector<double> radialAvg_mod(maxRadius, 0);
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
				auto radius = int(sqrt(r2));
				
				double value_real = DIRECT_A3D_ELEM(fftVol_real, zSize_half + k, ySize_half + j, xSize_half + i);				
				double value_mod  = DIRECT_A3D_ELEM(fftVol_mod,  zSize_half + k, ySize_half + j, xSize_half + i);				

				radialAvg_real[radius] += value_real;
				radialAvg_mod[radius]  += value_mod;
				radialCounter[radius]  += 1;
			}
		}
	}

	for (size_t i = 0; i < radialCounter.size(); i++)
	{
		if (radialCounter[i] > 0)
		{
			radialAvg_real[i] /= radialCounter[i];
			radialAvg_mod[i] /= radialCounter[i];
		}
	}

	MetaDataVec md;
	size_t id;

	for(size_t i = 0; i < radialCounter.size(); i++)
	{
		id = md.addObject();
		md.setValue(MDL_X, radialAvg_real[i], id);
		md.setValue(MDL_Y, radialAvg_mod[i],  id);
		md.setValue(MDL_Z, radialCounter[i],  id);
	}

	std::string outputMD = rawname + "_FT_profile.xmd";
	md.write(outputMD);

	std::cout << "Output metadata file generated at: " << outputMD << std::endl;

    return 0;
}