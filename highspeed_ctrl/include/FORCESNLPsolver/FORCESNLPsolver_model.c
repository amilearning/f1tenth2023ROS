/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) FORCESNLPsolver_model_ ## ID
#endif

#include <math.h> 
#include "FORCESNLPsolver_model.h"

#ifndef casadi_real
#define casadi_real FORCESNLPsolver_float
#endif

#ifndef casadi_int
#define casadi_int solver_int32_default
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_f1 CASADI_PREFIX(f1)
#define casadi_f2 CASADI_PREFIX(f2)
#define casadi_f3 CASADI_PREFIX(f3)
#define casadi_f4 CASADI_PREFIX(f4)
#define casadi_f5 CASADI_PREFIX(f5)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_sq CASADI_PREFIX(sq)

/* Symbol visibility in DLLs */
#if 0
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

casadi_real casadi_sq(casadi_real x) { return x*x;}

static const casadi_int casadi_s0[11] = {7, 1, 0, 7, 0, 1, 2, 3, 4, 5, 6};
static const casadi_int casadi_s1[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s2[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s3[14] = {1, 7, 0, 1, 2, 3, 4, 4, 4, 4, 0, 0, 0, 0};
static const casadi_int casadi_s4[9] = {5, 1, 0, 5, 0, 1, 2, 3, 4};
static const casadi_int casadi_s5[31] = {5, 7, 0, 4, 8, 9, 10, 14, 17, 21, 0, 1, 2, 3, 0, 1, 3, 4, 0, 1, 0, 1, 2, 3, 0, 1, 3, 0, 1, 3, 4};

/* FORCESNLPsolver_objective_0:(i0[7],i1[2])->(o0) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a2, a3;
  a0=100.;
  a1=arg[0]? arg[0][2] : 0;
  a2=arg[1]? arg[1][0] : 0;
  a1=(a1-a2);
  a1=casadi_sq(a1);
  a1=(a0*a1);
  a2=arg[0]? arg[0][3] : 0;
  a3=arg[1]? arg[1][1] : 0;
  a2=(a2-a3);
  a2=casadi_sq(a2);
  a0=(a0*a2);
  a1=(a1+a0);
  a0=1.0000000000000001e-01;
  a2=arg[0]? arg[0][0] : 0;
  a2=casadi_sq(a2);
  a2=(a0*a2);
  a1=(a1+a2);
  a2=arg[0]? arg[0][1] : 0;
  a2=casadi_sq(a2);
  a0=(a0*a2);
  a1=(a1+a0);
  if (res[0]!=0) res[0][0]=a1;
  return 0;
}

int FORCESNLPsolver_objective_0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

int FORCESNLPsolver_objective_0_alloc_mem(void) {
  return 0;
}

int FORCESNLPsolver_objective_0_init_mem(int mem) {
  return 0;
}

void FORCESNLPsolver_objective_0_free_mem(int mem) {
}

int FORCESNLPsolver_objective_0_checkout(void) {
  return 0;
}

void FORCESNLPsolver_objective_0_release(int mem) {
}

void FORCESNLPsolver_objective_0_incref(void) {
}

void FORCESNLPsolver_objective_0_decref(void) {
}

casadi_int FORCESNLPsolver_objective_0_n_in(void) { return 2;}

casadi_int FORCESNLPsolver_objective_0_n_out(void) { return 1;}

casadi_real FORCESNLPsolver_objective_0_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

const char* FORCESNLPsolver_objective_0_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    default: return 0;
  }
}

const char* FORCESNLPsolver_objective_0_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

const casadi_int* FORCESNLPsolver_objective_0_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

const casadi_int* FORCESNLPsolver_objective_0_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    default: return 0;
  }
}

int FORCESNLPsolver_objective_0_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 2;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

/* FORCESNLPsolver_dobjective_0:(i0[7],i1[2])->(o0[1x7,4nz]) */
static int casadi_f1(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a2;
  a0=1.0000000000000001e-01;
  a1=arg[0]? arg[0][0] : 0;
  a1=(a1+a1);
  a1=(a0*a1);
  if (res[0]!=0) res[0][0]=a1;
  a1=arg[0]? arg[0][1] : 0;
  a1=(a1+a1);
  a0=(a0*a1);
  if (res[0]!=0) res[0][1]=a0;
  a0=100.;
  a1=arg[0]? arg[0][2] : 0;
  a2=arg[1]? arg[1][0] : 0;
  a1=(a1-a2);
  a1=(a1+a1);
  a1=(a0*a1);
  if (res[0]!=0) res[0][2]=a1;
  a1=arg[0]? arg[0][3] : 0;
  a2=arg[1]? arg[1][1] : 0;
  a1=(a1-a2);
  a1=(a1+a1);
  a0=(a0*a1);
  if (res[0]!=0) res[0][3]=a0;
  return 0;
}

int FORCESNLPsolver_dobjective_0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f1(arg, res, iw, w, mem);
}

int FORCESNLPsolver_dobjective_0_alloc_mem(void) {
  return 0;
}

int FORCESNLPsolver_dobjective_0_init_mem(int mem) {
  return 0;
}

void FORCESNLPsolver_dobjective_0_free_mem(int mem) {
}

int FORCESNLPsolver_dobjective_0_checkout(void) {
  return 0;
}

void FORCESNLPsolver_dobjective_0_release(int mem) {
}

void FORCESNLPsolver_dobjective_0_incref(void) {
}

void FORCESNLPsolver_dobjective_0_decref(void) {
}

casadi_int FORCESNLPsolver_dobjective_0_n_in(void) { return 2;}

casadi_int FORCESNLPsolver_dobjective_0_n_out(void) { return 1;}

casadi_real FORCESNLPsolver_dobjective_0_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

const char* FORCESNLPsolver_dobjective_0_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    default: return 0;
  }
}

const char* FORCESNLPsolver_dobjective_0_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

const casadi_int* FORCESNLPsolver_dobjective_0_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

const casadi_int* FORCESNLPsolver_dobjective_0_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    default: return 0;
  }
}

int FORCESNLPsolver_dobjective_0_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 2;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

/* FORCESNLPsolver_dynamics_0:(i0[7],i1[2])->(o0[5]) */
static int casadi_f2(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a3, a4, a5, a6, a7, a8, a9;
  a0=arg[0]? arg[0][2] : 0;
  a1=1.6666666666666666e-02;
  a2=arg[0]? arg[0][4] : 0;
  a3=arg[0]? arg[0][5] : 0;
  a4=5.0000000000000000e-01;
  a5=arg[0]? arg[0][6] : 0;
  a6=tan(a5);
  a6=(a4*a6);
  a6=atan(a6);
  a7=(a3+a6);
  a7=cos(a7);
  a7=(a2*a7);
  a8=2.;
  a9=5.0000000000000003e-02;
  a10=arg[0]? arg[0][0] : 0;
  a11=2.0000000000000001e-01;
  a12=(a10/a11);
  a13=(a9*a12);
  a13=(a2+a13);
  a14=1.4999999999999999e-01;
  a15=(a2/a14);
  a16=sin(a6);
  a15=(a15*a16);
  a16=(a9*a15);
  a16=(a3+a16);
  a17=arg[0]? arg[0][1] : 0;
  a18=(a9*a17);
  a18=(a5+a18);
  a18=tan(a18);
  a18=(a4*a18);
  a18=atan(a18);
  a19=(a16+a18);
  a19=cos(a19);
  a19=(a13*a19);
  a19=(a8*a19);
  a7=(a7+a19);
  a19=(a10/a11);
  a20=(a9*a19);
  a20=(a2+a20);
  a21=(a13/a14);
  a22=sin(a18);
  a21=(a21*a22);
  a22=(a9*a21);
  a22=(a3+a22);
  a9=(a9*a17);
  a9=(a5+a9);
  a9=tan(a9);
  a9=(a4*a9);
  a9=atan(a9);
  a23=(a22+a9);
  a23=cos(a23);
  a23=(a20*a23);
  a23=(a8*a23);
  a7=(a7+a23);
  a23=1.0000000000000001e-01;
  a24=(a10/a11);
  a25=(a23*a24);
  a25=(a2+a25);
  a26=(a20/a14);
  a27=sin(a9);
  a26=(a26*a27);
  a27=(a23*a26);
  a27=(a3+a27);
  a23=(a23*a17);
  a23=(a5+a23);
  a23=tan(a23);
  a4=(a4*a23);
  a4=atan(a4);
  a23=(a27+a4);
  a23=cos(a23);
  a23=(a25*a23);
  a7=(a7+a23);
  a7=(a1*a7);
  a0=(a0+a7);
  if (res[0]!=0) res[0][0]=a0;
  a0=arg[0]? arg[0][3] : 0;
  a6=(a3+a6);
  a6=sin(a6);
  a6=(a2*a6);
  a16=(a16+a18);
  a16=sin(a16);
  a13=(a13*a16);
  a13=(a8*a13);
  a6=(a6+a13);
  a22=(a22+a9);
  a22=sin(a22);
  a20=(a20*a22);
  a20=(a8*a20);
  a6=(a6+a20);
  a27=(a27+a4);
  a27=sin(a27);
  a27=(a25*a27);
  a6=(a6+a27);
  a6=(a1*a6);
  a0=(a0+a6);
  if (res[0]!=0) res[0][1]=a0;
  a19=(a8*a19);
  a12=(a12+a19);
  a24=(a8*a24);
  a12=(a12+a24);
  a10=(a10/a11);
  a12=(a12+a10);
  a12=(a1*a12);
  a2=(a2+a12);
  if (res[0]!=0) res[0][2]=a2;
  a21=(a8*a21);
  a15=(a15+a21);
  a26=(a8*a26);
  a15=(a15+a26);
  a25=(a25/a14);
  a4=sin(a4);
  a25=(a25*a4);
  a15=(a15+a25);
  a15=(a1*a15);
  a3=(a3+a15);
  if (res[0]!=0) res[0][3]=a3;
  a3=(a8*a17);
  a3=(a17+a3);
  a8=(a8*a17);
  a3=(a3+a8);
  a3=(a3+a17);
  a1=(a1*a3);
  a5=(a5+a1);
  if (res[0]!=0) res[0][4]=a5;
  return 0;
}

int FORCESNLPsolver_dynamics_0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f2(arg, res, iw, w, mem);
}

int FORCESNLPsolver_dynamics_0_alloc_mem(void) {
  return 0;
}

int FORCESNLPsolver_dynamics_0_init_mem(int mem) {
  return 0;
}

void FORCESNLPsolver_dynamics_0_free_mem(int mem) {
}

int FORCESNLPsolver_dynamics_0_checkout(void) {
  return 0;
}

void FORCESNLPsolver_dynamics_0_release(int mem) {
}

void FORCESNLPsolver_dynamics_0_incref(void) {
}

void FORCESNLPsolver_dynamics_0_decref(void) {
}

casadi_int FORCESNLPsolver_dynamics_0_n_in(void) { return 2;}

casadi_int FORCESNLPsolver_dynamics_0_n_out(void) { return 1;}

casadi_real FORCESNLPsolver_dynamics_0_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

const char* FORCESNLPsolver_dynamics_0_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    default: return 0;
  }
}

const char* FORCESNLPsolver_dynamics_0_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

const casadi_int* FORCESNLPsolver_dynamics_0_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

const casadi_int* FORCESNLPsolver_dynamics_0_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    default: return 0;
  }
}

int FORCESNLPsolver_dynamics_0_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 2;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

/* FORCESNLPsolver_ddynamics_0:(i0[7],i1[2])->(o0[5x7,21nz]) */
static int casadi_f3(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a5, a50, a51, a6, a7, a8, a9;
  a0=1.6666666666666666e-02;
  a1=2.;
  a2=2.5000000000000000e-01;
  a3=arg[0]? arg[0][5] : 0;
  a4=5.0000000000000003e-02;
  a5=arg[0]? arg[0][4] : 0;
  a6=1.4999999999999999e-01;
  a7=(a5/a6);
  a8=5.0000000000000000e-01;
  a9=arg[0]? arg[0][6] : 0;
  a10=tan(a9);
  a10=(a8*a10);
  a11=atan(a10);
  a12=sin(a11);
  a13=(a7*a12);
  a13=(a4*a13);
  a13=(a3+a13);
  a14=arg[0]? arg[0][1] : 0;
  a15=(a4*a14);
  a15=(a9+a15);
  a16=tan(a15);
  a16=(a8*a16);
  a17=atan(a16);
  a18=(a13+a17);
  a19=cos(a18);
  a20=(a2*a19);
  a20=(a1*a20);
  a21=arg[0]? arg[0][0] : 0;
  a22=2.0000000000000001e-01;
  a23=(a21/a22);
  a23=(a4*a23);
  a23=(a5+a23);
  a24=(a23/a6);
  a25=sin(a17);
  a26=(a24*a25);
  a26=(a4*a26);
  a26=(a3+a26);
  a27=(a4*a14);
  a27=(a9+a27);
  a28=tan(a27);
  a28=(a8*a28);
  a29=atan(a28);
  a30=(a26+a29);
  a31=cos(a30);
  a32=(a2*a31);
  a33=(a21/a22);
  a33=(a4*a33);
  a33=(a5+a33);
  a30=sin(a30);
  a34=1.6666666666666667e+00;
  a35=(a34*a25);
  a36=(a4*a35);
  a37=(a30*a36);
  a37=(a33*a37);
  a32=(a32-a37);
  a32=(a1*a32);
  a20=(a20+a32);
  a32=1.0000000000000001e-01;
  a37=(a33/a6);
  a38=sin(a29);
  a39=(a37*a38);
  a39=(a32*a39);
  a39=(a3+a39);
  a14=(a32*a14);
  a14=(a9+a14);
  a40=tan(a14);
  a40=(a8*a40);
  a41=atan(a40);
  a42=(a39+a41);
  a43=cos(a42);
  a44=(a8*a43);
  a21=(a21/a22);
  a21=(a32*a21);
  a21=(a5+a21);
  a42=sin(a42);
  a34=(a34*a38);
  a22=(a32*a34);
  a45=(a42*a22);
  a45=(a21*a45);
  a44=(a44-a45);
  a20=(a20+a44);
  a20=(a0*a20);
  if (res[0]!=0) res[0][0]=a20;
  a13=(a13+a17);
  a20=sin(a13);
  a44=(a2*a20);
  a44=(a1*a44);
  a26=(a26+a29);
  a45=sin(a26);
  a2=(a2*a45);
  a26=cos(a26);
  a36=(a26*a36);
  a36=(a33*a36);
  a2=(a2+a36);
  a2=(a1*a2);
  a44=(a44+a2);
  a39=(a39+a41);
  a2=sin(a39);
  a36=(a8*a2);
  a39=cos(a39);
  a22=(a39*a22);
  a22=(a21*a22);
  a36=(a36+a22);
  a44=(a44+a36);
  a44=(a0*a44);
  if (res[0]!=0) res[0][1]=a44;
  if (res[0]!=0) res[0][2]=a8;
  a35=(a1*a35);
  a34=(a1*a34);
  a35=(a35+a34);
  a34=3.3333333333333335e+00;
  a44=sin(a41);
  a34=(a34*a44);
  a35=(a35+a34);
  a35=(a0*a35);
  if (res[0]!=0) res[0][3]=a35;
  a18=sin(a18);
  a15=cos(a15);
  a15=casadi_sq(a15);
  a35=(a4/a15);
  a35=(a8*a35);
  a34=1.;
  a16=casadi_sq(a16);
  a16=(a34+a16);
  a35=(a35/a16);
  a36=(a18*a35);
  a36=(a23*a36);
  a36=(a1*a36);
  a17=cos(a17);
  a22=(a17*a35);
  a22=(a24*a22);
  a46=(a4*a22);
  a27=cos(a27);
  a27=casadi_sq(a27);
  a47=(a4/a27);
  a47=(a8*a47);
  a28=casadi_sq(a28);
  a28=(a34+a28);
  a47=(a47/a28);
  a48=(a46+a47);
  a48=(a30*a48);
  a48=(a33*a48);
  a48=(a1*a48);
  a36=(a36+a48);
  a29=cos(a29);
  a48=(a29*a47);
  a48=(a37*a48);
  a49=(a32*a48);
  a14=cos(a14);
  a14=casadi_sq(a14);
  a50=(a32/a14);
  a50=(a8*a50);
  a40=casadi_sq(a40);
  a40=(a34+a40);
  a50=(a50/a40);
  a51=(a49+a50);
  a51=(a42*a51);
  a51=(a21*a51);
  a36=(a36+a51);
  a36=(a0*a36);
  a36=(-a36);
  if (res[0]!=0) res[0][4]=a36;
  a13=cos(a13);
  a35=(a13*a35);
  a35=(a23*a35);
  a35=(a1*a35);
  a46=(a46+a47);
  a46=(a26*a46);
  a46=(a33*a46);
  a46=(a1*a46);
  a35=(a35+a46);
  a49=(a49+a50);
  a49=(a39*a49);
  a49=(a21*a49);
  a35=(a35+a49);
  a35=(a0*a35);
  if (res[0]!=0) res[0][5]=a35;
  a22=(a1*a22);
  a48=(a1*a48);
  a22=(a22+a48);
  a6=(a21/a6);
  a41=cos(a41);
  a50=(a41*a50);
  a50=(a6*a50);
  a22=(a22+a50);
  a22=(a0*a22);
  if (res[0]!=0) res[0][6]=a22;
  if (res[0]!=0) res[0][7]=a32;
  if (res[0]!=0) res[0][8]=a34;
  if (res[0]!=0) res[0][9]=a34;
  a22=(a3+a11);
  a50=cos(a22);
  a48=6.6666666666666670e+00;
  a12=(a48*a12);
  a35=(a4*a12);
  a49=(a18*a35);
  a49=(a23*a49);
  a19=(a19-a49);
  a19=(a1*a19);
  a50=(a50+a19);
  a25=(a48*a25);
  a19=(a4*a25);
  a49=(a30*a19);
  a49=(a33*a49);
  a31=(a31-a49);
  a31=(a1*a31);
  a50=(a50+a31);
  a38=(a48*a38);
  a31=(a32*a38);
  a49=(a42*a31);
  a49=(a21*a49);
  a43=(a43-a49);
  a50=(a50+a43);
  a50=(a0*a50);
  if (res[0]!=0) res[0][10]=a50;
  a3=(a3+a11);
  a50=sin(a3);
  a35=(a13*a35);
  a35=(a23*a35);
  a20=(a20+a35);
  a20=(a1*a20);
  a50=(a50+a20);
  a19=(a26*a19);
  a19=(a33*a19);
  a45=(a45+a19);
  a45=(a1*a45);
  a50=(a50+a45);
  a31=(a39*a31);
  a31=(a21*a31);
  a2=(a2+a31);
  a50=(a50+a2);
  a50=(a0*a50);
  if (res[0]!=0) res[0][11]=a50;
  if (res[0]!=0) res[0][12]=a34;
  a25=(a1*a25);
  a12=(a12+a25);
  a38=(a1*a38);
  a12=(a12+a38);
  a48=(a48*a44);
  a12=(a12+a48);
  a12=(a0*a12);
  if (res[0]!=0) res[0][13]=a12;
  a22=sin(a22);
  a12=(a5*a22);
  a48=(a23*a18);
  a48=(a1*a48);
  a12=(a12+a48);
  a48=(a33*a30);
  a48=(a1*a48);
  a12=(a12+a48);
  a48=(a21*a42);
  a12=(a12+a48);
  a12=(a0*a12);
  a12=(-a12);
  if (res[0]!=0) res[0][14]=a12;
  a3=cos(a3);
  a12=(a5*a3);
  a48=(a23*a13);
  a48=(a1*a48);
  a12=(a12+a48);
  a48=(a33*a26);
  a48=(a1*a48);
  a12=(a12+a48);
  a48=(a21*a39);
  a12=(a12+a48);
  a12=(a0*a12);
  if (res[0]!=0) res[0][15]=a12;
  if (res[0]!=0) res[0][16]=a34;
  a9=cos(a9);
  a9=casadi_sq(a9);
  a9=(a8/a9);
  a10=casadi_sq(a10);
  a10=(a34+a10);
  a9=(a9/a10);
  a22=(a22*a9);
  a22=(a5*a22);
  a11=cos(a11);
  a11=(a11*a9);
  a7=(a7*a11);
  a11=(a4*a7);
  a15=(a8/a15);
  a15=(a15/a16);
  a16=(a11+a15);
  a18=(a18*a16);
  a18=(a23*a18);
  a18=(a1*a18);
  a22=(a22+a18);
  a17=(a17*a15);
  a24=(a24*a17);
  a4=(a4*a24);
  a27=(a8/a27);
  a27=(a27/a28);
  a28=(a4+a27);
  a30=(a30*a28);
  a30=(a33*a30);
  a30=(a1*a30);
  a22=(a22+a30);
  a29=(a29*a27);
  a37=(a37*a29);
  a32=(a32*a37);
  a8=(a8/a14);
  a8=(a8/a40);
  a40=(a32+a8);
  a42=(a42*a40);
  a42=(a21*a42);
  a22=(a22+a42);
  a22=(a0*a22);
  a22=(-a22);
  if (res[0]!=0) res[0][17]=a22;
  a3=(a3*a9);
  a5=(a5*a3);
  a11=(a11+a15);
  a13=(a13*a11);
  a23=(a23*a13);
  a23=(a1*a23);
  a5=(a5+a23);
  a4=(a4+a27);
  a26=(a26*a4);
  a33=(a33*a26);
  a33=(a1*a33);
  a5=(a5+a33);
  a32=(a32+a8);
  a39=(a39*a32);
  a21=(a21*a39);
  a5=(a5+a21);
  a5=(a0*a5);
  if (res[0]!=0) res[0][18]=a5;
  a24=(a1*a24);
  a7=(a7+a24);
  a1=(a1*a37);
  a7=(a7+a1);
  a41=(a41*a8);
  a6=(a6*a41);
  a7=(a7+a6);
  a0=(a0*a7);
  if (res[0]!=0) res[0][19]=a0;
  if (res[0]!=0) res[0][20]=a34;
  return 0;
}

int FORCESNLPsolver_ddynamics_0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f3(arg, res, iw, w, mem);
}

int FORCESNLPsolver_ddynamics_0_alloc_mem(void) {
  return 0;
}

int FORCESNLPsolver_ddynamics_0_init_mem(int mem) {
  return 0;
}

void FORCESNLPsolver_ddynamics_0_free_mem(int mem) {
}

int FORCESNLPsolver_ddynamics_0_checkout(void) {
  return 0;
}

void FORCESNLPsolver_ddynamics_0_release(int mem) {
}

void FORCESNLPsolver_ddynamics_0_incref(void) {
}

void FORCESNLPsolver_ddynamics_0_decref(void) {
}

casadi_int FORCESNLPsolver_ddynamics_0_n_in(void) { return 2;}

casadi_int FORCESNLPsolver_ddynamics_0_n_out(void) { return 1;}

casadi_real FORCESNLPsolver_ddynamics_0_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

const char* FORCESNLPsolver_ddynamics_0_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    default: return 0;
  }
}

const char* FORCESNLPsolver_ddynamics_0_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

const casadi_int* FORCESNLPsolver_ddynamics_0_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

const casadi_int* FORCESNLPsolver_ddynamics_0_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s5;
    default: return 0;
  }
}

int FORCESNLPsolver_ddynamics_0_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 2;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

/* FORCESNLPsolver_objective_1:(i0[7],i1[2])->(o0) */
static int casadi_f4(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a2, a3;
  a0=200.;
  a1=arg[0]? arg[0][2] : 0;
  a2=arg[1]? arg[1][0] : 0;
  a1=(a1-a2);
  a1=casadi_sq(a1);
  a1=(a0*a1);
  a2=arg[0]? arg[0][3] : 0;
  a3=arg[1]? arg[1][1] : 0;
  a2=(a2-a3);
  a2=casadi_sq(a2);
  a0=(a0*a2);
  a1=(a1+a0);
  a0=2.0000000000000001e-01;
  a2=arg[0]? arg[0][0] : 0;
  a2=casadi_sq(a2);
  a2=(a0*a2);
  a1=(a1+a2);
  a2=arg[0]? arg[0][1] : 0;
  a2=casadi_sq(a2);
  a0=(a0*a2);
  a1=(a1+a0);
  if (res[0]!=0) res[0][0]=a1;
  return 0;
}

int FORCESNLPsolver_objective_1(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f4(arg, res, iw, w, mem);
}

int FORCESNLPsolver_objective_1_alloc_mem(void) {
  return 0;
}

int FORCESNLPsolver_objective_1_init_mem(int mem) {
  return 0;
}

void FORCESNLPsolver_objective_1_free_mem(int mem) {
}

int FORCESNLPsolver_objective_1_checkout(void) {
  return 0;
}

void FORCESNLPsolver_objective_1_release(int mem) {
}

void FORCESNLPsolver_objective_1_incref(void) {
}

void FORCESNLPsolver_objective_1_decref(void) {
}

casadi_int FORCESNLPsolver_objective_1_n_in(void) { return 2;}

casadi_int FORCESNLPsolver_objective_1_n_out(void) { return 1;}

casadi_real FORCESNLPsolver_objective_1_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

const char* FORCESNLPsolver_objective_1_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    default: return 0;
  }
}

const char* FORCESNLPsolver_objective_1_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

const casadi_int* FORCESNLPsolver_objective_1_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

const casadi_int* FORCESNLPsolver_objective_1_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    default: return 0;
  }
}

int FORCESNLPsolver_objective_1_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 2;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

/* FORCESNLPsolver_dobjective_1:(i0[7],i1[2])->(o0[1x7,4nz]) */
static int casadi_f5(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a2;
  a0=2.0000000000000001e-01;
  a1=arg[0]? arg[0][0] : 0;
  a1=(a1+a1);
  a1=(a0*a1);
  if (res[0]!=0) res[0][0]=a1;
  a1=arg[0]? arg[0][1] : 0;
  a1=(a1+a1);
  a0=(a0*a1);
  if (res[0]!=0) res[0][1]=a0;
  a0=200.;
  a1=arg[0]? arg[0][2] : 0;
  a2=arg[1]? arg[1][0] : 0;
  a1=(a1-a2);
  a1=(a1+a1);
  a1=(a0*a1);
  if (res[0]!=0) res[0][2]=a1;
  a1=arg[0]? arg[0][3] : 0;
  a2=arg[1]? arg[1][1] : 0;
  a1=(a1-a2);
  a1=(a1+a1);
  a0=(a0*a1);
  if (res[0]!=0) res[0][3]=a0;
  return 0;
}

int FORCESNLPsolver_dobjective_1(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f5(arg, res, iw, w, mem);
}

int FORCESNLPsolver_dobjective_1_alloc_mem(void) {
  return 0;
}

int FORCESNLPsolver_dobjective_1_init_mem(int mem) {
  return 0;
}

void FORCESNLPsolver_dobjective_1_free_mem(int mem) {
}

int FORCESNLPsolver_dobjective_1_checkout(void) {
  return 0;
}

void FORCESNLPsolver_dobjective_1_release(int mem) {
}

void FORCESNLPsolver_dobjective_1_incref(void) {
}

void FORCESNLPsolver_dobjective_1_decref(void) {
}

casadi_int FORCESNLPsolver_dobjective_1_n_in(void) { return 2;}

casadi_int FORCESNLPsolver_dobjective_1_n_out(void) { return 1;}

casadi_real FORCESNLPsolver_dobjective_1_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

const char* FORCESNLPsolver_dobjective_1_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    default: return 0;
  }
}

const char* FORCESNLPsolver_dobjective_1_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

const casadi_int* FORCESNLPsolver_dobjective_1_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

const casadi_int* FORCESNLPsolver_dobjective_1_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    default: return 0;
  }
}

int FORCESNLPsolver_dobjective_1_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 2;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
