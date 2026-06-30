/***************************************************************************
 *
 * Authors:     Federico P. de Isidro-Gómez
 *
 * This program calculate the radial 2D average of the Fourier transfrom of 
 * a particle/micropgraph.
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
    FileName fnImg=argv[1];

	Image<double> img;
	img.read(fnImg);

	// Calculate FT
	FourierTransformer ft;
	MultidimArray< std::complex<double> > fftImg;
	ft.FourierTransform(img(), fftImg, false);

	// FT dimensions
	int xSize = XSIZE(fftImg);
	int ySize = YSIZE(fftImg);
	int nSize = NSIZE(fftImg);

	if (nSize == 1)
	{
		nSize = ZSIZE(fftImg);
	}

	int maxRadius = std::min(xSize, ySize);	// Restric analysis to Nyquist

	std::cout << "FFT map dimensions: " << std::endl;  
	std::cout << "xSize " << xSize << std::endl;
	std::cout << "ySize " << ySize << std::endl;
	std::cout << "nSize " << nSize << std::endl;
	std::cout << "Final size: (" << xSize << ", " << ySize << ", " << nSize << ")" << std::endl;
	std::cout << "maxRadius " << maxRadius << std::endl;

	// Construct frequency map and initialize the frequency vectors
	MultidimArray< double > freqMap;
	Matrix1D<double> freq_fourier_x;
	Matrix1D<double> freq_fourier_y;

	freq_fourier_x.initZeros(xSize);
	freq_fourier_y.initZeros(ySize);

	double u;	// u is the frequency

	// Defining frequency components. First element should be 0, it is set as the smallest number to avoid singularities
	VEC_ELEM(freq_fourier_y,0) = std::numeric_limits<double>::min();
	for(size_t k=1; k<ySize; ++k){
		FFT_IDX2DIGFREQ(k,YSIZE(img()), u);
		VEC_ELEM(freq_fourier_y, k) = u;
	}

	VEC_ELEM(freq_fourier_x,0) = std::numeric_limits<double>::min();
	for(size_t k=1; k<xSize; ++k){
		FFT_IDX2DIGFREQ(k,XSIZE(img()), u);
		VEC_ELEM(freq_fourier_x, k) = u;
	}

	//Initializing map with frequencies
	freqMap.resizeNoCopy(fftImg);

	// Directional frequencies along each direction
	double uy, ux, uy2;
	long n=0;
	int idx = 0;

	for(size_t i=0; i<YSIZE(fftImg); ++i)
	{
		uy = VEC_ELEM(freq_fourier_y, i);
		uy2 = uy*uy;

		for(size_t j=0; j<XSIZE(fftImg); ++j)
		{
			ux = VEC_ELEM(freq_fourier_x, j);
			ux = sqrt(uy2 + ux*ux);

			idx = (int) round(ux * XSIZE(img()));
			DIRECT_MULTIDIM_ELEM(freqMap,n) = idx;

			++n;
		}
	}
	

	// Compute real and magnitude FT maps, and calculate radial average
	MultidimArray<double> fftImg_real;
	fftImg_real.initZeros(ySize, xSize);
	MultidimArray<double> fftImg_mod;
	fftImg_mod.initZeros(ySize, xSize);

	std::vector<double> radialAvg_real(maxRadius, 0);
	std::vector<double> radialAvg_mod(maxRadius, 0);
	std::vector<double> radialCounter(maxRadius, 0);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(fftImg)
	{
		double value_real = DIRECT_MULTIDIM_ELEM(fftImg,n).real();
		double value_mod  = sqrt((DIRECT_MULTIDIM_ELEM(fftImg,n) * std::conj(DIRECT_MULTIDIM_ELEM(fftImg,n))).real());

		DIRECT_MULTIDIM_ELEM(fftImg_real,n) = value_real;
		DIRECT_MULTIDIM_ELEM(fftImg_mod,n)  = value_mod;
		
		if(DIRECT_MULTIDIM_ELEM(freqMap,n) < maxRadius)
		{
			radialAvg_real[(int)(DIRECT_MULTIDIM_ELEM(freqMap,n))] += value_real;
			radialAvg_mod[(int)(DIRECT_MULTIDIM_ELEM(freqMap,n))]  += value_mod;
			radialCounter[(int)(DIRECT_MULTIDIM_ELEM(freqMap,n))]  += 1;
		}
	}

	// Save FT maps
	size_t lastindex = fnImg.find_last_of(".");
	std::string rawname = fnImg.substr(0, lastindex);
	
	size_t atIndex = rawname.find("@");
	if (atIndex != std::string::npos) {
		rawname = rawname.substr(atIndex + 1);
	}

	Image<double> saveImage;
	
	std::string debugFileFn = rawname + "_freqMap.mrc";
	saveImage() = freqMap;
	saveImage.write(debugFileFn);

	debugFileFn = rawname + "_FT_real.mrc";
	saveImage() = fftImg_real;
	saveImage.write(debugFileFn);

	debugFileFn = rawname + "_FT_mod.mrc";
	saveImage() = fftImg_mod;
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