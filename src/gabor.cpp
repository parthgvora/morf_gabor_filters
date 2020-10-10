#include <pybind11/pybind11.h>
#include <cmath>
#include <vector>

#define PI 3.141592654

std::vector<std::vector<double>> gabor(int kwidth, int kheight, double sigma, double theta, double lambd, double gamma, double psi) {
   

    // initialize parameter space
    double sigma_x = sigma;
    double sigma_y = sigma/gamma;
    int nstds = 3;
    int xmin, xmax, ymin, ymax;
    double c = cos(theta), s = sin(theta);

    if( kwidth > 0 )
        xmax = kwidth/2;
    else
        xmax = int(std::max(fabs(nstds*sigma_x*c), fabs(nstds*sigma_y*s)));

    if( kheight > 0 )
        ymax = kheight/2;
    else
        ymax = int(std::max(fabs(nstds*sigma_x*s), fabs(nstds*sigma_y*c)));

    xmin = -xmax;
    ymin = -ymax;

    double scale = 1;
    double ex = -0.5/(sigma_x*sigma_x);
    double ey = -0.5/(sigma_y*sigma_y);
    double cscale = PI*2/lambd;

    // initialize kernel
    std::vector<std::vector<double>> kernel(kheight, std::vector<double> (kwidth, 0));

    // calculate values for kernel
    for( int y = ymin; y <= ymax; y++ )
        for( int x = xmin; x <= xmax; x++ )
        {
            double xr = x*c + y*s;
            double yr = -x*s + y*c;

            double v = scale*std::exp(ex*xr*xr + ey*yr*yr)*cos(cscale*xr + psi);
            kernel[y][x] = v;
            
            /*
            if( ktype == CV_32F )
                kernel.at<float>(ymax - y, xmax - x) = (float)v;
            else
                kernel.at<double>(ymax - y, xmax - x) = v;
            */
        }


    return kernel;
}


namespace py = pybind11;

PYBIND11_MODULE(gabor, m) {

    m.doc() = "gabor";
    m.def("gabor", &gabor, "Function to create a gabor filter.");

}
