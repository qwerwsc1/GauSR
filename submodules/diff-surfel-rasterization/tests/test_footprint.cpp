#include "../cuda_rasterizer/footprint_activation.cuh"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>

using namespace footprint;
static int checks = 0;
static void check(float got, float ref, float atol = 3e-4f, float rtol = 0.015f) {
    ++checks;
    if (!std::isfinite(got) || std::fabs(got-ref) > atol + rtol * std::fabs(ref)) {
        std::cerr << "FAIL: " << got << " vs " << ref << '\n';
        std::exit(1);
    }
}
static float eval(float o, float G, float m, Config c) { return activation(o,G,m,c).alpha; }
int main() {
    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> u(0.0f,1.0f);
    const float h=0.001f;
    for (int mode=0; mode<6; ++mode) {
        for(int k=0;k<400;++k) {
            Config c={static_cast<Mode>(mode),1.3f,0.15f+0.7f*u(rng),
                      0.1f+0.6f*u(rng),0.05f,4.0f,0.99f};
            float o=.05f+.35f*u(rng), G=.05f+.5f*u(rng), m=.2f+.7f*u(rng);
            auto r=activation(o,G,m,c);
            check(r.d_opacity,(eval(o+h,G,m,c)-eval(o-h,G,m,c))/(2*h));
            check(r.d_G,(eval(o,G+h,m,c)-eval(o,G-h,m,c))/(2*h));
            check(r.d_mu,(eval(o,G,m+h,c)-eval(o,G,m-h,c))/(2*h));
            auto cp=c, cm=c; cp.slope_t+=h; cm.slope_t-=h;
            check(r.d_slope_t,(eval(o,G,m,cp)-eval(o,G,m,cm))/(2*h));
            cp=c;cm=c;cp.slope_n+=h;cm.slope_n-=h;
            check(r.d_slope_n,(eval(o,G,m,cp)-eval(o,G,m,cm))/(2*h));
        }
    }
    Config c={Generalized,1,0.35f,0.15f,0.05f,4,0.99f};
    // Hard alpha cap and grazing clamps have zero derivatives in their interiors.
    auto sat=activation(100,1,.6f,c);
    check(sat.alpha,.99f);check(sat.d_opacity,0);check(sat.d_mu,0);
    auto grazing=activation(.2f,.4f,.01f,c);
    check(grazing.d_mu,0);
    check(grazing.d_opacity,(eval(.2f+h,.4f,.01f,c)-eval(.2f-h,.4f,.01f,c))/(2*h));
    // Frontal normalization and deterministic planar limit.
    for(int k=0;k<100;++k) {
        float o=u(rng), G=u(rng), m=.06f+.94f*u(rng);
        check(eval(o,G,1,c),-std::expm1(-o*G));
        auto smooth=c;smooth.slope_t=smooth.slope_n=0;
        check(eval(o,G,m,smooth),-std::expm1(-o*G));
    }
    // Independent numerical quadrature of E[max(mu+s Z,0)].
    for(float m : {0.0f,.1f,.5f,1.0f}) for(float sd : {.1f,.35f,1.0f}) {
        double v=0, dz=16.0/100000;
        for(int k=0;k<100000;++k) {
            double z=-8+(k+.5)*dz;
            v+=std::max(0.0,double(m)+sd*z)*std::exp(-z*z/2)/std::sqrt(2*3.141592653589793)*dz;
        }
        check(positive_moment(m,sd).value,float(v),2e-6f,2e-5f);
    }
    // Independent double-precision CDF reference, including the finite-offset normalization.
    for(int k=0;k<200;++k) {
        double x=double(k)/30.0, cc=FOOTPRINT_CDF_OFFSET;
        double Q=std::erfc(-(cc-x)/std::sqrt(2.0))/std::erfc(-cc/std::sqrt(2.0));
        Config gc={GaussianCDF,1,0,0,.05f,4,.99f};
        check(eval(float(x),1,.5f,gc),float(std::min(.99,1-Q*Q)),2e-6f,1e-4f);
    }
    // CDF derivatives at larger field amplitudes, away from the clipping transition.
    for(float o : {.1f,.3f,.5f,.7f,.9f}) {
        Config gc={GaussianCDF,4.60517f,0,0,.05f,4,.99f};
        auto a=activation(o,.8f,.5f,gc);
        check(a.d_opacity,(eval(o+h,.8f,.5f,gc)-eval(o-h,.8f,.5f,gc))/(2*h));
    }
    // SGGX projection, reciprocity and SPD eigensystem for the implemented family.
    for(int k=0;k<100;++k) {
        float m=.1f+.8f*u(rng), r=.05f+.6f*u(rng);
        Config sg={SGGX,1,r,0,.05f,100,.99f};
        float projected=std::sqrt(m*m+r*r*(1-m*m));
        check(eval(.1f,.3f,m,sg),-std::expm1(-.03f*projected/m));
        check(eval(.1f,.3f,1,sg),-std::expm1(-.03f));
    }
    // Chain rule through non-unit camera-space normal and compositing/background.
    for (Mode direction_mode : {Generalized, SGGX}) {
    c.mode=direction_mode;
    Vec3 n={.45f,-.2f,.9f}, ray=camera_ray(80,90,640,480,500,510);
    auto objective=[&](Vec3 a,float o,float G){
        auto inc=incidence(a,ray);
        float alpha=eval(o,G,inc.mu,c);
        // Fixed foreground transmittance, current color and composite behind it.
        float pixel=.7f*(alpha*.8f+(1-alpha)*.2f)+.1f;
        return .5f*(pixel-.6f)*(pixel-.6f);
    };
    auto inc=incidence(n,ray);auto r=activation(.4f,.65f,inc.mu,c);
    float pixel=.7f*(r.alpha*.8f+(1-r.alpha)*.2f)+.1f;
    float da=(pixel-.6f)*.7f*(.8f-.2f);
    for(int k=0;k<3;++k){
        Vec3 a=n,b=n;
        if(k==0){a.x+=h;b.x-=h;} if(k==1){a.y+=h;b.y-=h;} if(k==2){a.z+=h;b.z-=h;}
        float dn=k==0?inc.d_normal.x:(k==1?inc.d_normal.y:inc.d_normal.z);
        check(da*r.d_mu*dn,(objective(a,.4f,.65f)-objective(b,.4f,.65f))/(2*h),1e-5f);
    }
    check(da*r.d_opacity,(objective(n,.4f+h,.65f)-objective(n,.4f-h,.65f))/(2*h),1e-5f);
    check(da*r.d_G,(objective(n,.4f,.65f+h)-objective(n,.4f,.65f-h))/(2*h),1e-5f);
    }
    // Positive, monotonic radial footprints and declared finite support bound.
    for(int mode=0;mode<6;++mode){
        c.mode=static_cast<Mode>(mode);
        for(float m : {0.f,.02f,.1f,.5f,1.f}){
            float last=1;
            for(int k=0;k<=200;++k){
                auto v=activation(4.0f,std::exp(-.5f*k*.15f),m,c).alpha;
                assert(v>=0 && v<=.99f && v<=last+1e-6f);last=v;
            }
        }
    }
    assert(4.60517f*4.f*4.f*std::exp(-.5f*FOOTPRINT_CUTOFF*FOOTPRINT_CUTOFF)<-std::log1p(-1.f/255.f));
    std::cout<<"PASS: "<<checks<<" numerical checks; plus range, monotonicity and support assertions.\n";
}
