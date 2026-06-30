/***************************************************************************
 *
 * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
 *              Federico P. de Isidro-Gómez
 *
 * This program reads a CTF estimation from a ctfparam file and simulates it
 * over pure-noise micrograph.
 * 
 * To compile this standalone version run:
 * 
 *      scipion3 run xmipp_compile simulateMicrograh.cpp 
 * 
 ***************************************************************************/


#include<core/xmipp_image.h>
#include<libraries/data/ctf.h>
#include<core/xmipp_fftw.h>

int main(int argc, char **argv)
{
    FileName fnOut=argv[1];
    FileName fnCTF=argv[2];

    CTFDescription CTF;
    CTF.enable_CTF=true;
    CTF.enable_CTFnoise=true;
    CTF.read(fnCTF);
    CTF.produceSideInfo();

    const int Xdim=4096;
    const double Ts=1.0;
    Image<double> I(Xdim,Xdim);
    I().initRandom(0,1,RND_GAUSSIAN);

    FourierTransformer transformer;
    MultidimArray<std::complex<double> > FI;
    transformer.FourierTransform(I(), FI);

    Matrix1D<double> w(2);
    for (size_t i=0; i<YSIZE(FI); i++)
    {
        FFT_IDX2DIGFREQ(i,Xdim,YY(w));
        for (size_t j=0; j<XSIZE(FI); j++)
        {
            FFT_IDX2DIGFREQ(j,Xdim,XX(w));
            CTF.precomputeValues(XX(w)/Ts,YY(w)/Ts);
            //double sigmaN=CTF.getValueNoiseAt();
            double sigmaN=sqrt(std::max(0.0,CTF.getValueNoiseAt()));
            double ctf=CTF.getValuePureAt();
            DIRECT_A2D_ELEM(FI,i,j)*=ctf+sigmaN;
            if (i==0)
            {
                std::cout<<"j="<<j<<"ctf=" << ctf << " sigmaN=" << sigmaN << std::endl;
            }
        }
    }

    transformer.inverseFourierTransform(FI, I());
    I.write(fnOut);
    return 0;
}