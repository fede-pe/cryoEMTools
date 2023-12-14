/***************************************************************************
 *
 * Authors:     Federico P. de Isidro-Gómez
 *
 * This program reads a CTF estimation from a ctfparam file and simulates it
 * over pure-noise micrograph.
 * 
 * To compile this standalone version run:
 * 
 *      scipion3 run xmipp_compile compareDefocusDifference.cpp 
 * 
 ***************************************************************************/


#include<libraries/data/ctf.h>
#include <iostream>
#include <fstream>


int main(int argc, char **argv)
{
    FileName fnCTF=argv[1];

    double defocusDifference = 100;

    std::vector<double> ctfProfile(2048, 0);
    std::vector<double> ctfDefocalizedProfile(2048, 0);
    
    CTFDescription CTF;
    CTF.enable_CTF=true;
    CTF.enable_CTFnoise=false;
    CTF.DeltafU = 20000;
    CTF.DeltafV = 20000;
    CTF.read(fnCTF);
    CTF.produceSideInfo();

    CTFDescription CTFdef;
    CTFdef.enable_CTF=true;
    CTFdef.enable_CTFnoise=false;
    CTFdef.read(fnCTF);
    CTFdef.DeltafU += 200;
    CTFdef.DeltafV += 200;
    CTFdef.produceSideInfo();

    const int Xdim=4096;
    const double Ts=1.0;

    Matrix1D<double> w(2);
    for (size_t i=0; i<2048; i++)
    {
        FFT_IDX2DIGFREQ(i,Xdim,YY(w));
        FFT_IDX2DIGFREQ(0,Xdim,XX(w));
        
        CTF.precomputeValues(XX(w)/Ts,YY(w)/Ts);
        CTFdef.precomputeValues(XX(w)/Ts,YY(w)/Ts);

        ctfProfile[i] = CTF.getValuePureAt();
        ctfDefocalizedProfile[i] = CTFdef.getValuePureAt();
    }

    const char* filePathCtf = "ctfProfile.txt";
    const char* filePathCtfDefocalized = "ctfDefocalizedProfile.txt";

    std::ofstream outFileCtf(filePathCtf);
    std::ofstream outFileCtfDef(filePathCtfDefocalized);

    if (outFileCtf.is_open()) {
        for (double value : ctfProfile) {
            outFileCtf << value << std::endl;
        }
        outFileCtf.close();
    }

    if (outFileCtfDef.is_open()) {
        for (double value : ctfDefocalizedProfile) {
            outFileCtfDef << value << std::endl;
        }
        outFileCtfDef.close();
    }

    return 0;
}