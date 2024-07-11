
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro3d_blast.cpp
/// \brief Defines a test problem for a 3D explosion.
///

#include <limits>
#include <math.h>
#include <iostream>
#include <random>

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_Config.H"
#include "AMReX_FabArrayUtility.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_SPACE.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_ParallelContext.H"
#include "AMReX_TableData.H"
#include "AMReX_RandomEngine.H"
#include "AMReX_Random.H"

#include "ODEIntegrate.hpp"
#include "RadhydroSimulation.hpp"
#include "hydro_system.hpp"
#include "radiation_system.hpp"
#include "test_sne.hpp"
#include "quadrature.hpp"
#include "NSCBC_outflow.hpp"
#include "AMReX_TableData.H"
#include "FastMath.hpp"
#include "interpolate.hpp"

using amrex::Real;
using namespace amrex;
int arrshape = 4999;
std::string input_data_file; //="/g/data/jh2/av5889/quokka_myrepo/quokka/sims/GasGravity/PhiGas_R8.h5";
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, 4999> phi_data;
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, 4999> g_data;
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, 4999> z_data;
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, 1> z_star, Sigma_star, rho_dm, R0, ks_sigma_sfr, hscale;
double sigma1, sigma2, rho01, rho02;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto linearInterpolate(amrex::GpuArray<amrex::Real, 4999>& x, amrex::GpuArray<amrex::Real, 4999>& y, double x_interp) {
    // Find the two closest data points
    size_t i = 0;
    while (i < x.size() - 1 && x_interp > x[i + 1]) {
        i++;
    }

    // Perform linear interpolation
    double x1 = x[i];
    double x2 = x[i + 1];
    double y1 = y[i];
    double y2 = y[i + 1];

    return y1 + (y2 - y1) * (x_interp - x1) / (x2 - x1);
}


#define MAX 100

struct NewProblem {
  amrex::Real dummy;
};

template <> struct HydroSystem_Traits<NewProblem> {
  static constexpr double gamma = 5./3.;
  static constexpr bool reconstruct_eint = true; //Set to true - temperature
};

template <> struct quokka::EOS_Traits<NewProblem> {
	static constexpr double gamma = 5./3.;
	static constexpr double mean_molecular_weight = C::m_u;
	static constexpr double boltzmann_constant = C::k_B;
};

template <> struct Physics_Traits<NewProblem> {
  static constexpr bool is_hydro_enabled = true;
  static constexpr bool is_radiation_enabled = false;
  static constexpr bool is_chemistry_enabled = false;
  static constexpr bool is_mhd_enabled = false;
  static constexpr int numMassScalars = 0;		     // number of mass scalars
  static constexpr int numPassiveScalars = 1; // number of passive scalars
  static constexpr int nGroups = 1; // number of radiation groups
};

/************************************************************/

// template <>
void read_potential(amrex::GpuArray<amrex::Real, 4999> &z_data,
                    amrex::GpuArray<amrex::Real, 4999> &phi_data,
                    amrex::GpuArray<amrex::Real, 4999> &g_data)
{
  const double small_fastlog_value = FastMath::log10(1.0e-99);
  
	
 
	// Read cooling data from hdf5 file
	hid_t file_id = 0;
	hid_t dset_id = 0;
	hid_t attr_id = 0;
	herr_t status = 0;
	herr_t h5_error = -1;

	file_id = H5Fopen(input_data_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  
	// Open cooling dataset and get grid dimensions
  
	std::string parameter_name;
	parameter_name = "PhiGas" ;
	dset_id = H5Dopen2(file_id, parameter_name.c_str(),
			   H5P_DEFAULT); // new API in HDF5 1.8.0+  
  auto *phidata = new double[4999]; // NOLINT(cppcoreguidelines-owning-memory)
	{
		status = H5Dread(dset_id, HDF5_R8, H5S_ALL, H5S_ALL, H5P_DEFAULT, phidata);
   	for (int64_t q = 0; q < 4999; q++) {
			double value = phidata[q];
			phi_data[q] =  FastMath::log10(value) ;
		}
  }
	status = H5Dclose(dset_id);

  parameter_name = "ZVal" ;
  dset_id = H5Dopen2(file_id, parameter_name.c_str(),
			   H5P_DEFAULT); // new API in HDF5 1.8.0+  
  auto *zdata = new double[4999]; // NOLINT(cppcoreguidelines-owning-memory)
	{
		status = H5Dread(dset_id, HDF5_R8, H5S_ALL, H5S_ALL, H5P_DEFAULT, zdata);
    for (int64_t q = 0; q < 4999; q++) {
			double value = zdata[q];
			z_data[q] =  value;
		}
  }
		status = H5Dclose(dset_id);

 parameter_name = "gGas" ;
  dset_id = H5Dopen2(file_id, parameter_name.c_str(),
			   H5P_DEFAULT); // new API in HDF5 1.8.0+  
  auto *gdata = new double[4999]; // NOLINT(cppcoreguidelines-owning-memory)
	{
		status = H5Dread(dset_id, HDF5_R8, H5S_ALL, H5S_ALL, H5P_DEFAULT, gdata);
    for (int64_t q = 0; q < 4999; q++) {
			double value = gdata[q];
			g_data[q] =  FastMath::log10(value);
		}
  }
  status = H5Dclose(dset_id);   

  parameter_name = "zStar" ;
  dset_id = H5Dopen2(file_id, parameter_name.c_str(),
			   H5P_DEFAULT); // new API in HDF5 1.8.0+  
  auto *zstar = new double[1]; // NOLINT(cppcoreguidelines-owning-memory)
	{
		status = H5Dread(dset_id, HDF5_R8, H5S_ALL, H5S_ALL, H5P_DEFAULT, zstar);
    for (int64_t q = 0; q < 1; q++) {
			double value = zstar[q];
			z_star =  value;
		}
  } 
  status = H5Dclose(dset_id);   

  parameter_name = "Sigma_star" ;
  dset_id = H5Dopen2(file_id, parameter_name.c_str(),
			   H5P_DEFAULT); // new API in HDF5 1.8.0+  
  auto *sigstar = new double[1]; // NOLINT(cppcoreguidelines-owning-memory)
	{
		status = H5Dread(dset_id, HDF5_R8, H5S_ALL, H5S_ALL, H5P_DEFAULT, sigstar);
		Sigma_star =  sigstar[0];
		
  } 
		status = H5Dclose(dset_id); 

  parameter_name = "rho_dm" ;
  dset_id = H5Dopen2(file_id, parameter_name.c_str(),
			   H5P_DEFAULT); // new API in HDF5 1.8.0+  
  auto *rhodm = new double[1]; // NOLINT(cppcoreguidelines-owning-memory)
	{
		status = H5Dread(dset_id, HDF5_R8, H5S_ALL, H5S_ALL, H5P_DEFAULT, rhodm);
		rho_dm =  rhodm[0];
		
  } 
		status = H5Dclose(dset_id);   

  parameter_name = "R0" ;
  dset_id = H5Dopen2(file_id, parameter_name.c_str(),
			   H5P_DEFAULT); // new API in HDF5 1.8.0+  
  auto *rnought = new double[1]; // NOLINT(cppcoreguidelines-owning-memory)
	{
		status = H5Dread(dset_id, HDF5_R8, H5S_ALL, H5S_ALL, H5P_DEFAULT, rnought);
		R0 =  rnought[0];
		
  } 
		status = H5Dclose(dset_id);   

  parameter_name = "ks_sigma_sfr" ;
  dset_id = H5Dopen2(file_id, parameter_name.c_str(),
			   H5P_DEFAULT); // new API in HDF5 1.8.0+  
  auto *kssigma = new double[1]; // NOLINT(cppcoreguidelines-owning-memory)
	{
		status = H5Dread(dset_id, HDF5_R8, H5S_ALL, H5S_ALL, H5P_DEFAULT, kssigma);
		ks_sigma_sfr =  kssigma[0];
		
  } 
		status = H5Dclose(dset_id);  
  parameter_name = "hscale" ;
  dset_id = H5Dopen2(file_id, parameter_name.c_str(),
			   H5P_DEFAULT); // new API in HDF5 1.8.0+  
  auto *h_scale = new double[1]; // NOLINT(cppcoreguidelines-owning-memory)
	{
		status = H5Dread(dset_id, HDF5_R8, H5S_ALL, H5S_ALL, H5P_DEFAULT, h_scale);
		hscale =  h_scale[0];
		
  } 
		status = H5Dclose(dset_id);    

  parameter_name = "sigma1" ;
  dset_id = H5Dopen2(file_id, parameter_name.c_str(),
			   H5P_DEFAULT); // new API in HDF5 1.8.0+  
  auto *sigma_1 = new double[1]; // NOLINT(cppcoreguidelines-owning-memory)
	{
		status = H5Dread(dset_id, HDF5_R8, H5S_ALL, H5S_ALL, H5P_DEFAULT, sigma_1);
		sigma1 =  sigma_1[0];
		
  } 
		status = H5Dclose(dset_id);   


  parameter_name = "sigma2" ;
  dset_id = H5Dopen2(file_id, parameter_name.c_str(),
			   H5P_DEFAULT); // new API in HDF5 1.8.0+  
  auto *sigma_2 = new double[1]; // NOLINT(cppcoreguidelines-owning-memory)
	{
		status = H5Dread(dset_id, HDF5_R8, H5S_ALL, H5S_ALL, H5P_DEFAULT, sigma_2);
		sigma2 =  sigma_2[0];
		
  }
  status = H5Dclose(dset_id);   

  parameter_name = "rho1" ;
  dset_id = H5Dopen2(file_id, parameter_name.c_str(),
			   H5P_DEFAULT); // new API in HDF5 1.8.0+  
  auto *rho_1 = new double[1]; // NOLINT(cppcoreguidelines-owning-memory)
	{
		status = H5Dread(dset_id, HDF5_R8, H5S_ALL, H5S_ALL, H5P_DEFAULT, rho_1);
		rho01 =  rho_1[0];
		
  } 
		status = H5Dclose(dset_id); 

  parameter_name = "rho2" ;
  dset_id = H5Dopen2(file_id, parameter_name.c_str(),
			   H5P_DEFAULT); // new API in HDF5 1.8.0+  
  auto *rho_2 = new double[1]; // NOLINT(cppcoreguidelines-owning-memory)
	{
		status = H5Dread(dset_id, HDF5_R8, H5S_ALL, H5S_ALL, H5P_DEFAULT, rho_2);
		rho02 =  rho_2[0];
		
  } 
		status = H5Dclose(dset_id);   
	   
  printf("Gasgravity file read!\n");
  printf("R0, rho_dm=%.2e,%.2e\n", R0, rho_dm);
}

/************************************************************/

// global variables needed for Dirichlet boundary condition and initial conditions
#if 0 // workaround AMDGPU compiler bug
namespace
{
#endif
Real rho0 = NAN;                   // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED Real Tgas0 = NAN;              // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED Real P_outflow = NAN;              // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
#if 0                      // workaround AMDGPU compiler bug
};                       // namespace
#endif


template <> struct SimulationData<NewProblem> {

  // cloudy_tables cloudyTables;
  std::unique_ptr<amrex::TableData<Real, 3>> table_data;

	std::unique_ptr<amrex::TableData<Real, 1>> blast_x;
	std::unique_ptr<amrex::TableData<Real, 1>> blast_y;
	std::unique_ptr<amrex::TableData<Real, 1>> blast_z;

	int nblast = 0;
	int SN_counter_cumulative = 0;
	Real SN_rate_per_vol = NAN; // rate per unit time per unit volume
	Real E_blast = 1.0e51;	    // ergs
	Real M_ejecta = 5.0 * Msun ;	    // 5.0 * Msun; // g
	Real refine_threshold = 1.0; // gradient refinement threshold
};



template <>
void RadhydroSimulation<NewProblem>::setInitialConditionsOnGrid(quokka::grid grid_elem) {
  
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
    const amrex::Box &indexRange = grid_elem.indexRange_;
    const amrex::Array4<double>& state_cc = grid_elem.array_;
    
    double vol       =  AMREX_D_TERM(dx[0], *dx[1], *dx[2]);


  amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {

    
      amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
			amrex::Real const y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];
      amrex::Real const z = prob_lo[2] + (k + amrex::Real(0.5)) * dx[2];


      /*Calculate DM Potential*/
      double prefac;
      prefac = 2.* 3.1415 * Const_G * rho_dm * std::pow(R0,2);
      double Phidm =  (prefac * std::log(1. + std::pow(z/R0, 2)));

      /*Calculate Stellar Disk Potential*/
      double prefac2;
      prefac2 = 2.* 3.1415 * Const_G * Sigma_star * z_star ;
      double Phist =  prefac2 * (std::pow(1. + z*z/z_star/z_star,0.5) -1.);

      /*Calculate Gas Disk Potential*/
      
      double Phigas;
      Phigas =FastMath::pow10( linearInterpolate(z_data, phi_data, std::abs(z)));
    
      double Phitot = Phist + Phidm + Phigas; 

			double rho, rho_disk, rho_halo;
             rho_disk = rho01 * std::exp(-Phitot/std::pow(sigma1,2.0)) ;
             rho_halo = rho02 * std::exp(-Phitot/std::pow(sigma2,2.0));         //in g/cc
             rho = (rho_disk + rho_halo);

      double P = rho_disk * std::pow(sigma1, 2.0) + rho_halo * std::pow(sigma2, 2.0);

      AMREX_ASSERT(!std::isnan(rho));
      
			const auto gamma = HydroSystem<NewProblem>::gamma_;

      //For a uniform box
        // rho01  = 1.e-2 * Const_mH;
        // rho = rho01;
        // sigma1 = 37. * kmps;
        // P = rho01 * std::pow(sigma1, 2.0);
     

      state_cc(i, j, k, HydroSystem<NewProblem>::density_index)    = rho;
      state_cc(i, j, k, HydroSystem<NewProblem>::x1Momentum_index) = 0.0;
      state_cc(i, j, k, HydroSystem<NewProblem>::x2Momentum_index) = 0.0;
      state_cc(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) = 0.0;
      state_cc(i, j, k, HydroSystem<NewProblem>::internalEnergy_index) = P / (gamma - 1.);
      state_cc(i, j, k, HydroSystem<NewProblem>::energy_index)         = P / (gamma - 1.);
      state_cc(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex)    = 1.e-5/vol;  //Injected tracer

    });
  }

template <>
void RadhydroSimulation<NewProblem>::ErrorEst(int lev,
                                                amrex::TagBoxArray &tags,
                                                amrex::Real /*time*/ ,
                                                int /*ngrow*/) {
  // tag cells for refinement

  const amrex::Real eta_threshold = 4.0; // gradient refinement threshold
 
  for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
    const amrex::Box &box = mfi.validbox();
    const auto state = state_new_cc_[lev].const_array(mfi);
    const auto tag = tags.array(mfi);
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo   = geom[lev].ProbLoArray();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx = geom[lev].CellSizeArray();
   
    amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {

        amrex::Real  delMoxy = Msun;
        amrex::Real  Znorm = 1.e3;
        amrex::Real  ZOinit = 8.6e-3;
        amrex::Real rho_oxy_ = ZOinit *  state(i, j, k, HydroSystem<NewProblem>::density_index) ;
         
         amrex::Real scal_xyz   = ZOinit + ((delMoxy/Znorm) * state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex)/
                                                    state(i, j, k, HydroSystem<NewProblem>::density_index)) ;

        amrex::Real scal_xplus  = ZOinit + ((delMoxy/Znorm) * state(i+1, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex)/
                                                              state(i+1, j, k, HydroSystem<NewProblem>::density_index) ) ;

        amrex::Real scal_xminus = ZOinit + ((delMoxy/Znorm) * state(i-1, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex)/
                                                              state(i-1, j, k, HydroSystem<NewProblem>::density_index)) ;

        amrex::Real scal_yplus  = ZOinit + ((delMoxy/Znorm) *  state(i, j+1, k, Physics_Indices<NewProblem>::pscalarFirstIndex)/ 
                                                               state(i, j+1, k, HydroSystem<NewProblem>::density_index));

        amrex::Real scal_yminus = ZOinit + ((delMoxy/Znorm) *  state(i, j-1, k, Physics_Indices<NewProblem>::pscalarFirstIndex) / 
                                                               state(i, j-1, k, HydroSystem<NewProblem>::density_index));

        amrex::Real scal_zplus  = ZOinit + ((delMoxy/Znorm) *  state(i, j, k+1, Physics_Indices<NewProblem>::pscalarFirstIndex)/ 
                                                               state(i, j, k+1, HydroSystem<NewProblem>::density_index));

        amrex::Real scal_zminus = ZOinit + ((delMoxy/Znorm) *  state(i, j, k-1, Physics_Indices<NewProblem>::pscalarFirstIndex) / 
                                                               state(i, j, k-1, HydroSystem<NewProblem>::density_index));
        
        amrex::Real del_scalx   = std::abs(scal_xplus - scal_xminus)/2;
        amrex::Real del_scaly   = std::abs(scal_yplus - scal_zminus)/2.;
        amrex::Real del_scalz   = std::abs(scal_zplus - scal_zminus)/2.;
        // std::max(std::abs(scal_xplus - scal_xyz), std::abs(scal_xminus - scal_xyz));
        // amrex::Real del_scaly   = std::max(std::abs(scal_yplus - scal_xyz), std::abs(scal_yminus - scal_xyz));
        // amrex::Real del_scalz   = std::max(std::abs(scal_zplus - scal_xyz), std::abs(scal_zminus - scal_xyz));
        
        amrex::Real const grad_scal = (del_scalx  +  del_scaly  + del_scalz )/scal_xyz;          
        
        if ((grad_scal > eta_threshold)) {
        tag(i, j, k) = amrex::TagBox::SET;
        // printf("Reached here=%d, %d, %d, %.2e\n", i, j, k, grad_scal);
      }

     
    });
  }
}


void AddSupernova(amrex::MultiFab &mf, amrex::GpuArray<Real, AMREX_SPACEDIM> prob_lo, amrex::GpuArray<Real, AMREX_SPACEDIM> prob_hi,
		  amrex::GpuArray<Real, AMREX_SPACEDIM> dx, SimulationData<NewProblem> const &userData, int level)
{
	// inject energy into cells with stochastic sampling
	BL_PROFILE("RadhydroSimulation::Addsupernova()")

	const Real cell_vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]); // cm^3
	const Real rho_eint_blast = userData.E_blast / cell_vol;   // ergs cm^-3
	const Real rho_blast = userData.M_ejecta / cell_vol  ;   // g cm^-3
	const int cum_sn = userData.SN_counter_cumulative;

	const Real Lx = prob_hi[0] - prob_lo[0];
	const Real Ly = prob_hi[1] - prob_lo[1];
	const Real Lz = prob_hi[2] - prob_lo[2];

	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &box = iter.validbox();
		auto const &state = mf.array(iter);
		auto const &px = userData.blast_x->table();
		auto const &py = userData.blast_y->table();
		auto const &pz = userData.blast_z->table();
		const int np = userData.nblast;
		
   
		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const Real xc = prob_lo[0] + static_cast<Real>(i) * dx[0] + 0.5 * dx[0];
			const Real yc = prob_lo[1] + static_cast<Real>(j) * dx[1] + 0.5 * dx[1];
			const Real zc = prob_lo[2] + static_cast<Real>(k) * dx[2] + 0.5 * dx[2];

			for (int n = 0; n < 1; ++n) {
				Real x0 = NAN;
				Real y0 = NAN;
				Real z0 = NAN;
        Real Rpds = 0.0;
        
        x0 = std::abs(xc -px(n));
        y0 = std::abs(yc -py(n));
        z0 = std::abs(zc -pz(n));

        if(x0<0.5 *dx[0] && y0<0.5 *dx[1] && z0< 0.5 *dx[2] ) {
        // if(i==32 & j==32 & k==32){
        state(i, j, k, HydroSystem<NewProblem>::density_index)        +=   rho_blast; 
        state(i, j, k, HydroSystem<NewProblem>::energy_index)         +=   rho_eint_blast; 
        state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index) +=    rho_eint_blast; 
        state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+1)+=  1.e3/cell_vol;
        // printf("The location of SN=%d,%d,%d\n",i, j, k);
        // printf("SN added at level=%d\n", level);
        printf("The total number of SN gone off=%d\n", cum_sn);
        Rpds = 14. * std::pow(state(i, j, k, HydroSystem<NewProblem>::density_index)/Const_mH, -3./7.);
        // printf("Rpds = %.2e pc\n", Rpds);
        }
			}
		});
	}
}


template <> void RadhydroSimulation<NewProblem>::computeBeforeTimestep()
{
	// compute how many (and where) SNe will go off on the this coarse timestep
	// sample from Poisson distribution
  
	const Real dt_coarse = dt_[0];
	const Real domain_vol = geom[0].ProbSize();
  const Real domain_area = geom[0].ProbLength(0) * geom[0].ProbLength(1); 
  const Real mean = 0.0;
  const Real stddev = hscale/geom[0].ProbLength(2)/2.;
  
	const Real expectation_value = ks_sigma_sfr * domain_area * dt_coarse;
  
	const int count = static_cast<int>(amrex::RandomPoisson(expectation_value));
  
	if (count > 0) {
		// amrex::Print() << "\t" << count << " SNe to be exploded.\n";
    // amrex::Print() << "\t" << ks_sigma_sfr << " Expectation value.\n";
  }
	// resize particle arrays
	amrex::Array<int, 1> const lo{0};
	amrex::Array<int, 1> const hi{count};
	userData_.blast_x = std::make_unique<amrex::TableData<Real, 1>>(lo, hi, amrex::The_Pinned_Arena());
	userData_.blast_y = std::make_unique<amrex::TableData<Real, 1>>(lo, hi, amrex::The_Pinned_Arena());
	userData_.blast_z = std::make_unique<amrex::TableData<Real, 1>>(lo, hi, amrex::The_Pinned_Arena());
	userData_.nblast = count;
	userData_.SN_counter_cumulative += count;

	// for each, sample location at random
	auto const &px = userData_.blast_x->table();
	auto const &py = userData_.blast_y->table();
	auto const &pz = userData_.blast_z->table();
	for (int i = 0; i < count; ++i) {
		px(i) = geom[0].ProbLength(0) * amrex::Random();
		py(i) = geom[0].ProbLength(1) * amrex::Random();
		pz(i) = geom[0].ProbLength(2) * amrex::RandomNormal(mean, stddev);
	}
} 

/*******************************************************************/



template <>
void RadhydroSimulation<NewProblem>::computeAfterLevelAdvance(int lev, amrex::Real time,
								 amrex::Real dt_lev, int ncycle)
{
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo   = geom[lev].ProbLoArray();
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi   = geom[lev].ProbHiArray();
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx = geom[lev].CellSizeArray();
  
  AddSupernova(state_new_cc_[lev], prob_lo, prob_hi, dx, userData_, lev);
  
}

template <> AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto
HydroSystem<NewProblem>::GetGradFixedPotential(amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> posvec)
                                  -> amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> {
 
     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> grad_potential;
// auto const &dummy = userData_.blast_x;
    
      double x = posvec[0];
     
     grad_potential[0] =  0.0;

    #if (AMREX_SPACEDIM >= 2)
       double y = posvec[1];
       grad_potential[1] = 0.0;
    #endif
    #if (AMREX_SPACEDIM >= 3)
       double z      = posvec[2];
       grad_potential[2]  = 2.* 3.1415 * Const_G * rho_dm * std::pow(R0,2) * (2.* z/std::pow(R0,2))/(1. + std::pow(z,2)/std::pow(R0,2));
       grad_potential[2] += 2.* 3.1415 * Const_G * Sigma_star * (z/z_star) * (std::pow(1. + z*z/(z_star*z_star), -0.5));
       grad_potential[2] += (z/std::abs(z))*FastMath::pow10( linearInterpolate(z_data, g_data, std::abs(z)));;
    #endif

return grad_potential;
}

/* Add Strang Split Source Term for External Fixed Potential Here */
template <>
void RadhydroSimulation<NewProblem>::addStrangSplitSources(amrex::MultiFab &mf, int lev, amrex::Real time,
				 amrex::Real dt_lev)
{
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo   = geom[lev].ProbLoArray();
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx = geom[lev].CellSizeArray();
  const Real dt = dt_lev;

  for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
    const amrex::Box &indexRange = iter.validbox();
    auto const &state = mf.array(iter);

    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j,
                                                        int k) noexcept {

      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> posvec, GradPhi;
      double x1mom_new, x2mom_new, x3mom_new;

      const Real rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
      const Real x1mom =
          state(i, j, k, HydroSystem<NewProblem>::x1Momentum_index);
      const Real x2mom =
          state(i, j, k, HydroSystem<NewProblem>::x2Momentum_index);
      const Real x3mom =
          state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index);
      const Real Egas = state(i, j, k, HydroSystem<NewProblem>::energy_index);

					const auto vx = x1mom / rho;
					const auto vy = x2mom / rho;
					const auto vz = x3mom / rho;
					const double vel_mag = std::sqrt(vx * vx + vy * vy + vz * vz);
      
      Real Eint = RadSystem<NewProblem>::ComputeEintFromEgas(rho, x1mom, x2mom,
                                                              x3mom, Egas);
      

      posvec[0] = prob_lo[0] + (i+0.5)*dx[0];

        #if (AMREX_SPACEDIM >= 2)
          posvec[1] = prob_lo[1] + (j+0.5)*dx[1]; 
        #endif

        #if (AMREX_SPACEDIM >= 3)
          posvec[2] = prob_lo[2] + (k+0.5)*dx[2]; 
        #endif

      GradPhi = HydroSystem<NewProblem>::GetGradFixedPotential(posvec);   
      // GradPhi[1] = 0.0;
      // GradPhi[2] = 0.0;

      x1mom_new = state(i, j, k, HydroSystem<NewProblem>::x1Momentum_index) + dt * (-rho * GradPhi[0]);
      x2mom_new = state(i, j, k, HydroSystem<NewProblem>::x2Momentum_index) + dt * (-rho * GradPhi[1]);
      x3mom_new = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) + dt * (-rho * GradPhi[2]);

      state(i, j, k, HydroSystem<NewProblem>::x1Momentum_index) = x1mom_new;
      state(i, j, k, HydroSystem<NewProblem>::x2Momentum_index) = x2mom_new;
      state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) = x3mom_new;
      
      state(i, j, k, HydroSystem<NewProblem>::energy_index) = RadSystem<NewProblem>::ComputeEgasFromEint(
          rho, x1mom_new, x2mom_new, x3mom_new, Eint);
     });  
   }
}

/**************************End Adding Strang Split Source Term *****************/


/*********---Projection----******/
template <> auto RadhydroSimulation<NewProblem>::ComputeProjections(const int dir) const -> std::unordered_map<std::string, amrex::BaseFab<amrex::Real>>
{
  // compute density projection
  std::unordered_map<std::string, amrex::BaseFab<amrex::Real>> proj;

  proj["mass_outflow"] = computePlaneProjection<amrex::ReduceOpSum>(
      [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
        // int nmscalars = Physics_Traits<NewProblem>::numMassScalars;
        Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
        Real const vz = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index)/rho;
        
        amrex::Real const vx3 = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) / rho;
        amrex::Real Eint = state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index);
  
        amrex::GpuArray<Real, 0> massScalars = RadSystem<NewProblem>::ComputeMassScalars(state, i, j, k);
        Real const primTemp = quokka::EOS<NewProblem>::ComputeTgasFromEint(rho, Eint, massScalars);
        return (rho * vz) ;
      },
      dir);

   proj["hot_mass_outflow"] = computePlaneProjection<amrex::ReduceOpSum>(
      [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
        
        double flux;
        Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
        Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+1);
        Real const vx3 = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) / rho;
        Real const Eint = state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index);
        amrex::GpuArray<Real, 0> massScalars = RadSystem<NewProblem>::ComputeMassScalars(state, i, j, k);
		    Real const primTemp = quokka::EOS<NewProblem>::ComputeTgasFromEint(rho, Eint, massScalars);
        if(primTemp>1.e6) {
          flux = rho * vx3;
        } else {
          flux = 0.0;
        }
        return flux ;
      },
      dir);

  proj["warm_mass_outflow"] = computePlaneProjection<amrex::ReduceOpSum>(
      [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
        
        double flux;
        Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
        Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+1);
        Real const vx3 = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) / rho;
        Real const Eint = state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index);
        amrex::GpuArray<Real, 0> massScalars = RadSystem<NewProblem>::ComputeMassScalars(state, i, j, k);
		    Real const primTemp = quokka::EOS<NewProblem>::ComputeTgasFromEint(rho, Eint, massScalars);
        if(primTemp<2.e4) {
          flux = rho * vx3;
        } else {
          flux = 0.0;
        }
        return flux ;
      },
      dir);   

  proj["scalar_outflow"] = computePlaneProjection<amrex::ReduceOpSum>(
      [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
        Real const rho  = state(i, j, k, HydroSystem<NewProblem>::density_index);
        Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+1);
        Real const vz   = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index)/rho;
        return (rhoZ * vz) ;
      },
      dir);

  proj["warm_scalar_outflow"] = computePlaneProjection<amrex::ReduceOpSum>(
      [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
        
        double flux;
        Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
        Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+1);
        Real const vx3 = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) / rho;
        Real const Eint = state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index);
        amrex::GpuArray<Real, 0> massScalars = RadSystem<NewProblem>::ComputeMassScalars(state, i, j, k);
		    Real const primTemp = quokka::EOS<NewProblem>::ComputeTgasFromEint(rho, Eint, massScalars);
        if(primTemp<2.e4) {
          flux = rhoZ * vx3;
        } else {
          flux = 0.0;
        }
        return flux ;
      },
      dir); 

  proj["hot_scalar_outflow"] = computePlaneProjection<amrex::ReduceOpSum>(
      [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
        
        double flux;
        Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
        Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+1);
        Real const vx3 = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) / rho;
        Real const Eint = state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index);
        amrex::GpuArray<Real, 0> massScalars = RadSystem<NewProblem>::ComputeMassScalars(state, i, j, k);
		    Real const primTemp = quokka::EOS<NewProblem>::ComputeTgasFromEint(rho, Eint, massScalars);
        if(primTemp>1.e6) {
          flux = rhoZ * vx3;
        } else {
          flux = 0.0;
        }
        return flux ;
      },
      dir);      

  proj["rho"] = computePlaneProjection<amrex::ReduceOpSum>(
      [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
        Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
        return (rho) ;
      },
      dir);

  proj["scalar"] = computePlaneProjection<amrex::ReduceOpSum>(
      [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
        Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+1);
        return (rhoZ) ;
      },
      dir);    
  return proj;
}

/**************************Begin Diode BC *****************/

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void AMRSimulation<NewProblem>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<Real> const &consVar,
                          int /*dcomp*/ , int /*numcomp*/, amrex::GeometryData const &geom,
                           const Real /*time*/, const amrex::BCRec * /*bcr*/, int /*bcomp*/,
                           int /*orig_comp*/ )
{
  auto [i, j, k] = iv.dim3();
  amrex::Box const &box = geom.Domain();
  const auto &domain_lo = box.loVect3d();
  const auto &domain_hi = box.hiVect3d();
  const int klo = domain_lo[2];
  const int khi = domain_hi[2];
  int kedge, normal;


   if (k < klo) {
      kedge = klo;
      normal = -1;
   }
   else if (k > khi) {
      kedge = khi;
      normal = 1.0;
   }

    const double rho_edge   = consVar(i, j, kedge, HydroSystem<NewProblem>::density_index);
    const double x1Mom_edge = consVar(i, j, kedge, HydroSystem<NewProblem>::x1Momentum_index);
    const double x2Mom_edge = consVar(i, j, kedge, HydroSystem<NewProblem>::x2Momentum_index);
          double x3Mom_edge = consVar(i, j, kedge, HydroSystem<NewProblem>::x3Momentum_index);
    const double etot_edge  = consVar(i, j, kedge, HydroSystem<NewProblem>::energy_index);
    const double eint_edge  = consVar(i, j, kedge, HydroSystem<NewProblem>::internalEnergy_index);


    if((x3Mom_edge*normal)<0){//gas is inflowing
      x3Mom_edge = -1. *consVar(i, j, kedge, HydroSystem<NewProblem>::x3Momentum_index);
    }

        consVar(i, j, k, HydroSystem<NewProblem>::density_index)    = rho_edge ;
        consVar(i, j, k, HydroSystem<NewProblem>::x1Momentum_index) =  x1Mom_edge;
        consVar(i, j, k, HydroSystem<NewProblem>::x2Momentum_index) =  x2Mom_edge;
        consVar(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) =  x3Mom_edge;
        consVar(i, j, k, HydroSystem<NewProblem>::energy_index)     = etot_edge;
        consVar(i, j, k, HydroSystem<NewProblem>::internalEnergy_index) = eint_edge;

}

/**************************End NSCBC *****************/

auto problem_main() -> int {

  const int ncomp_cc = Physics_Indices<NewProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);

  /*Implementing Outflowing Boundary Conditions in the Z-direction*/

	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
				// outflowing boundary conditions
        if(i==2){
				 BCs_cc[n].setLo(i, amrex::BCType::foextrap);
				 BCs_cc[n].setHi(i, amrex::BCType::foextrap);
        }
        else{
           BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
           BCs_cc[n].setHi(i, amrex::BCType::int_dir); // periodic
        }
        }}
   
  // Problem initialization
  RadhydroSimulation<NewProblem> sim(BCs_cc);
  
  amrex::ParmParse const pp("phi_file");
	pp.query("name", input_data_file); 
  
  sim.reconstructionOrder_ = 3; // 2=PLM, 3=PPM
  sim.cflNumber_ = 0.3;         // *must* be less than 1/3 in 3D!
  
  read_potential(z_data, phi_data, g_data);
  
  // readCloudyData(sim.userData_.cloudyTables);
  // initialize
  sim.setInitialConditions();

  // evolve
  sim.evolve();

  // Cleanup and exit
  amrex::Print() << "Finished." << std::endl;
  return 0;
}

