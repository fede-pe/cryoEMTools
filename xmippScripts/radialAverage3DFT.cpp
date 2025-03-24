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
	int maxRadius = std::min(xSize, std::min(ySize, zSize));	// Restric analysis to Nyquist

	std::cout << "FFT map dimensions: " << std::endl;  
	std::cout << "xSize " << xSize << std::endl;
	std::cout << "ySize " << ySize << std::endl;
	std::cout << "zSize " << zSize << std::endl;
	std::cout << "maxRadius " << maxRadius << std::endl;

	// Construct frequency map and initialize the frequency vectors
	MultidimArray< double > freqMap;
	Matrix1D<double> freq_fourier_x;
	Matrix1D<double> freq_fourier_y;
	Matrix1D<double> freq_fourier_z;

	freq_fourier_x.initZeros(xSize);
	freq_fourier_y.initZeros(ySize);
	freq_fourier_z.initZeros(zSize);

	double u;	// u is the frequency

	// Defining frequency components. First element should be 0, it is set as the smallest number to avoid singularities
	VEC_ELEM(freq_fourier_z,0) = std::numeric_limits<double>::min();
	for(size_t k=1; k<zSize; ++k){
		FFT_IDX2DIGFREQ(k,ZSIZE(volMap()), u);
		VEC_ELEM(freq_fourier_z, k) = u;
	}

	VEC_ELEM(freq_fourier_y,0) = std::numeric_limits<double>::min();
	for(size_t k=1; k<ySize; ++k){
		FFT_IDX2DIGFREQ(k,YSIZE(volMap()), u);
		VEC_ELEM(freq_fourier_y, k) = u;
	}

	VEC_ELEM(freq_fourier_x,0) = std::numeric_limits<double>::min();
	for(size_t k=1; k<xSize; ++k){
		FFT_IDX2DIGFREQ(k,XSIZE(volMap()), u);
		VEC_ELEM(freq_fourier_x, k) = u;
	}

	//Initializing map with frequencies
	freqMap.resizeNoCopy(fftVol);

	size_t xvoldim = XSIZE(volMap());	// Assume volume is cubic!
	// size_t yvoldim = YSIZE(volMap());
	// size_t zvoldim = ZSIZE(volMap());

	// Directional frequencies along each direction
	double uz, uy, ux, uz2, uz2y2;
	long n=0;
	int idx = 0;

	for(size_t k=0; k<ZSIZE(fftVol); ++k)
	{
		uz = VEC_ELEM(freq_fourier_z, k);
		uz2 = uz*uz;
		
		for(size_t i=0; i<YSIZE(fftVol); ++i)
		{
			uy = VEC_ELEM(freq_fourier_y, i);
			uz2y2 = uz2 + uy*uy;

			for(size_t j=0; j<XSIZE(fftVol); ++j)
			{
				ux = VEC_ELEM(freq_fourier_x, j);
				ux = sqrt(uz2y2 + ux*ux);

				idx = (int) round(ux * xvoldim);
				DIRECT_MULTIDIM_ELEM(freqMap,n) = idx;

				++n;
			}
		}
	}

	// Compute real and magnitude FT maps, and calculate radial average
	MultidimArray<double> fftVol_real;
	fftVol_real.initZeros(zSize, ySize, xSize);
	MultidimArray<double> fftVol_mod;
	fftVol_mod.initZeros(zSize, ySize, xSize);

	std::vector<double> radialAvg_real(maxRadius, 0);
	std::vector<double> radialAvg_mod(maxRadius, 0);
	std::vector<double> radialCounter(maxRadius, 0);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(fftVol)
	{
		double value_real = DIRECT_MULTIDIM_ELEM(fftVol,n).real();
		double value_mod  = sqrt((DIRECT_MULTIDIM_ELEM(fftVol,n) * std::conj(DIRECT_MULTIDIM_ELEM(fftVol,n))).real());

		DIRECT_MULTIDIM_ELEM(fftVol_real,n) = value_real;
		DIRECT_MULTIDIM_ELEM(fftVol_mod,n)  = value_mod;
		
		if(DIRECT_MULTIDIM_ELEM(freqMap,n) < maxRadius)
		{
			radialAvg_real[(int)(DIRECT_MULTIDIM_ELEM(freqMap,n))] += value_real;
			radialAvg_mod[(int)(DIRECT_MULTIDIM_ELEM(freqMap,n))]  += value_mod;
			radialCounter[(int)(DIRECT_MULTIDIM_ELEM(freqMap,n))]  += 1;
		}
	}

	// Save FT maps
	size_t lastindex = fnVol.find_last_of(".");
	std::string rawname = fnVol.substr(0, lastindex);
	Image<double> saveImage;
	
	std::string debugFileFn = rawname + "_freqMap.mrc";
	saveImage() = freqMap;
	saveImage.write(debugFileFn);

	debugFileFn = rawname + "_FT_real.mrc";
	saveImage() = fftVol_real;
	saveImage.write(debugFileFn);

	debugFileFn = rawname + "_FT_mod.mrc";
	saveImage() = fftVol_mod;
	saveImage.write(debugFileFn);

	// Save output metadata
	MetaDataVec md;
	size_t id;

	for(size_t i = 0; i < radialCounter.size(); i++)
	{
		id = md.addObject();
		md.setValue(MDL_X, radialAvg_real[i] / radialCounter[i], id);
		md.setValue(MDL_Y, radialAvg_mod[i] / radialCounter[i],  id);
		md.setValue(MDL_Z, radialCounter[i],  id);
	}

	std::string outputMD = rawname + "_FT_profile.xmd";
	md.write(outputMD);

	std::cout << "Output metadata file generated at: " << outputMD << std::endl;

    return 0;
}