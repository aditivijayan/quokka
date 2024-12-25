
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_sne.cpp
/// \brief Defines a problem for disk galaxy ISM.
///

#include <cmath>
#include <iostream>

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_Random.H"
#include "AMReX_SPACE.H"
#include "AMReX_TableData.H"

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "radiation/radiation_system.hpp"
#include "test_sne.hpp"

// global variables needed for Dirichlet boundary condition and initial conditions


//---########----Values for R8 Model-------################---//

AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, 64> logphi_data{3.24975864, 6.27954616, 6.8675804 , 7.2142687 , 7.46059677,
       7.65149496, 7.80718072, 7.93849124, 8.05191629, 8.15165531,
       8.24058664, 8.32076511, 8.3937092 , 8.46057409, 8.52225965,
       8.57947934, 8.63280826, 8.682718  , 8.72959937, 8.77377954,
       8.81553494, 8.85510245, 8.89268623, 8.92846337, 8.96258787,
       8.99519552, 9.02640573, 9.05632431, 9.0850453 , 9.1126533 ,
       9.13922402, 9.16482567, 9.18951991, 9.21336297, 9.23640588,
       9.25869529, 9.28027478, 9.30118469, 9.32146184, 9.34113998,
       9.36025027, 9.37882149, 9.39688021, 9.41445096, 9.43155666,
       9.44821863, 9.46445669, 9.48028925, 9.49573374, 9.51080637,
       9.52552235, 9.53989592, 9.55394059, 9.56766904, 9.58109321,
       9.59422433, 9.60707313, 9.61964974, 9.63196405, 9.64402569,
       9.6558438 , 9.66742698, 9.67878337, 9.6899207};
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, 64> logg_data{-8.84486253, -8.53977369, -8.41945731, -8.36855149, -8.34812648,
       -8.34077673, -8.33835551, -8.33764733, -8.33744008, -8.33738188,
       -8.3373655 , -8.33736028, -8.33735794, -8.33735633, -8.33735492,
       -8.33735358, -8.33735228, -8.33735101, -8.33734977, -8.33734856,
       -8.33734737, -8.33734622, -8.33734509, -8.33734399, -8.33734291,
       -8.33734186, -8.33734084, -8.33733984, -8.33733887, -8.33733792,
       -8.337337  , -8.33733609, -8.33733522, -8.33733436, -8.33733353,
       -8.33733272, -8.33733193, -8.33733115, -8.3373304 , -8.33732966,
       -8.33732894, -8.33732824, -8.33732756, -8.33732689, -8.33732625,
       -8.33732561, -8.337325  , -8.3373244 , -8.33732382, -8.33732326,
       -8.33732271, -8.33732217, -8.33732165, -8.33732115, -8.33732066,
       -8.33732019, -8.33731973, -8.33731928, -8.33731885, -8.33731843,
       -8.33731801, -8.33731761, -8.33731722, -8.33731684
        };
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, 64> z_data{1.50900000e+20, 3.40123810e+20, 5.29347619e+20, 7.18571429e+20,
       9.07795238e+20, 1.09701905e+21, 1.28624286e+21, 1.47546667e+21,
       1.66469048e+21, 1.85391429e+21, 2.04313810e+21, 2.23236190e+21,
       2.42158571e+21, 2.61080952e+21, 2.80003333e+21, 2.98925714e+21,
       3.17848095e+21, 3.36770476e+21, 3.55692857e+21, 3.74615238e+21,
       3.93537619e+21, 4.12460000e+21, 4.31382381e+21, 4.50304762e+21,
       4.69227143e+21, 4.88149524e+21, 5.07071905e+21, 5.25994286e+21,
       5.44916667e+21, 5.63839048e+21, 5.82761429e+21, 6.01683810e+21,
       6.20606190e+21, 6.39528571e+21, 6.58450952e+21, 6.77373333e+21,
       6.96295714e+21, 7.15218095e+21, 7.34140476e+21, 7.53062857e+21,
       7.71985238e+21, 7.90907619e+21, 8.09830000e+21, 8.28752381e+21,
       8.47674762e+21, 8.66597143e+21, 8.85519524e+21, 9.04441905e+21,
       9.23364286e+21, 9.42286667e+21, 9.61209048e+21, 9.80131429e+21,
       9.99053810e+21, 1.01797619e+22, 1.03689857e+22, 1.05582095e+22,
       1.07474333e+22, 1.09366571e+22, 1.11258810e+22, 1.13151048e+22,
       1.15043286e+22, 1.16935524e+22, 1.18827762e+22, 1.20720000e+22
      };


AMREX_GPU_MANAGED Real z_star = 245.0 * pc;
AMREX_GPU_MANAGED Real Sigma_star = 208 * Msun/pc/pc;
AMREX_GPU_MANAGED Real rho_dm = 2.4e-2 * Msun/pc/pc/pc;
AMREX_GPU_MANAGED Real R0 = 4.e3 * pc; 
AMREX_GPU_MANAGED Real ks_sigma_sfr = 1.3857971188361884e-54; 
AMREX_GPU_MANAGED Real hscale= 30. * pc;
AMREX_GPU_MANAGED Real sigma1 = 7.* kmps;
AMREX_GPU_MANAGED Real sigma2 = 10. * 7.* kmps;
AMREX_GPU_MANAGED Real rho01 =1.668*Const_mH;
AMREX_GPU_MANAGED Real rho02 = 1.e-5 * 1.668*Const_mH;;



AMREX_GPU_MANAGED Real hscaleIa= 2. * 30. * pc;
AMREX_GPU_MANAGED Real hscaleAGB= 300. * pc;

AMREX_GPU_MANAGED Real Tgas0 = 1.e4 ; //Temperature of gas ejected by AGB

struct NewProblem {
};

template <> struct HydroSystem_Traits<NewProblem> {
	static constexpr double gamma = 5. / 3.;
	static constexpr bool reconstruct_eint = true; // Set to true - temperature
};

template <> struct quokka::EOS_Traits<NewProblem> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
	static constexpr double boltzmann_constant = C::k_B;
};

template <> struct Physics_Traits<NewProblem> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_chemistry_enabled = false;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int numMassScalars = 0;    // number of mass scalars
	static constexpr int numPassiveScalars = 3; // number of passive scalars
	static constexpr int nGroups = 1;	    // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> struct SimulationData<NewProblem> {

	std::unique_ptr<amrex::TableData<Real, 1>> blast_x;
	std::unique_ptr<amrex::TableData<Real, 1>> blast_y;
	std::unique_ptr<amrex::TableData<Real, 1>> blast_z;

	std::unique_ptr<amrex::TableData<Real, 1>> blast_x1a;
	std::unique_ptr<amrex::TableData<Real, 1>> blast_y1a;
	std::unique_ptr<amrex::TableData<Real, 1>> blast_z1a;

	std::unique_ptr<amrex::TableData<Real, 1>> blast_xAGB;
	std::unique_ptr<amrex::TableData<Real, 1>> blast_yAGB;
	std::unique_ptr<amrex::TableData<Real, 1>> blast_zAGB;

	int nblast = 0;
	int nblast1a = 0;
	int nblastAGB = 0;
	int SN_counter_cumulative = 0;
	Real SN_rate_per_vol = NAN;  // rate per unit time per unit volume
	Real E_blast = 1.0e51;	     // ergs
	Real M_ejecta = 5.0 * Msun;  // 5.0 * Msun; // g
	Real refine_threshold = 1.0; // gradient refinement threshold
	Real M_ejecta_AGB = 1.3 * Msun;  // 5.0 * Msun; // g
};

template <> void QuokkaSimulation<NewProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{

	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	double vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const z = prob_lo[2] + (k + amrex::Real(0.5)) * dx[2];

		// Calculate DM Potential
		double prefac;
		prefac = 2. * M_PI * Const_G * rho_dm * std::pow(R0, 2);
		double Phidm = (prefac * std::log(1. + std::pow(z / R0, 2)));

		// Calculate Stellar Disk Potential
		double prefac2;
		prefac2 = 2. * M_PI * Const_G * Sigma_star * z_star;
		double Phist = prefac2 * (std::pow(1. + z * z / z_star / z_star, 0.5) - 1.);

		// Calculate Gas Disk Potential

		double Phigas;
		// Interpolate to find the accurate g-value from array-- because linterp doesn't work on Setonix
		// TODO - AV to find out why linterp doesn't work
		size_t ii = 0;
		double x_interp = std::abs(z);
		while (ii < z_data.size() - 1 && x_interp > z_data[ii + 1]) {
			ii++;
		}

		// Perform linear interpolation
		const Real x1 = z_data[ii];
		const Real x2 = z_data[ii + 1];
		const Real y1 = logphi_data[ii];
		const Real y2 = logphi_data[ii + 1];
		amrex::Real phi_interp = (y1 + (y2 - y1) * (x_interp - x1) / (x2 - x1));
		Phigas = std::pow(10., phi_interp);

		double Phitot = Phist + Phidm + Phigas;

		double rho, rho_disk, rho_halo;
		rho_disk = rho01 * std::exp(-Phitot / std::pow(sigma1, 2.0));
		rho_halo = rho02 * std::exp(-Phitot / std::pow(sigma2, 2.0)); // in g/cc
		rho = (rho_disk + rho_halo);

		double P = rho_disk * std::pow(sigma1, 2.0) + rho_halo * std::pow(sigma2, 2.0);

		AMREX_ASSERT(!std::isnan(rho));

		const auto gamma = HydroSystem<NewProblem>::gamma_;

		state_cc(i, j, k, HydroSystem<NewProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<NewProblem>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<NewProblem>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<NewProblem>::internalEnergy_index) = P / (gamma - 1.);
		state_cc(i, j, k, HydroSystem<NewProblem>::energy_index) = P / (gamma - 1.);
		state_cc(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex) = 1.e-5 / vol; // Injected tracer
		state_cc(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+1) = 1.e-5 / vol; // Injected tracer 2
		state_cc(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+2) = 1.e-5 / vol; // Injected tracer 3
	});
}

void AddSupernova(amrex::MultiFab &mf, amrex::GpuArray<Real, AMREX_SPACEDIM> prob_lo, amrex::GpuArray<Real, AMREX_SPACEDIM> prob_hi,
		  amrex::GpuArray<Real, AMREX_SPACEDIM> dx, SimulationData<NewProblem> const &userData, int level)
{
	// TODO for AV - ave (and restore) the RNG state in the metadata.yaml file
	//  inject energy into cells with stochastic sampling
	BL_PROFILE("QuokkaSimulation::Addsupernova()")

	const Real cell_vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]); // cm^3
	const Real rho_eint_blast = userData.E_blast / cell_vol;   // ergs cm^-3
	const Real rho_blast = userData.M_ejecta / cell_vol;	   // g cm^-3
	const Real rho_blast_AGB = userData.M_ejecta_AGB / cell_vol;	   // g cm^-3
	const Real scalar_blast = 1.e3 / cell_vol;		   // g cm^-3
	const int cum_sn = userData.SN_counter_cumulative;

	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &box = iter.validbox();
		auto const &state = mf.array(iter);
		auto const &px = userData.blast_x->table();
		auto const &py = userData.blast_y->table();
		auto const &pz = userData.blast_z->table();
		const int np = userData.nblast;

		auto const &px1a = userData.blast_x1a->table();
		auto const &py1a = userData.blast_y1a->table();
		auto const &pz1a = userData.blast_z1a->table();
		const int np1a = userData.nblast1a;
		
		auto const &pxAGB = userData.blast_xAGB->table();
		auto const &pyAGB = userData.blast_yAGB->table();
		auto const &pzAGB = userData.blast_zAGB->table();
		const int npAGB = userData.nblastAGB;

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const Real xc = prob_lo[0] + static_cast<Real>(i) * dx[0] + 0.5 * dx[0];
			const Real yc = prob_lo[1] + static_cast<Real>(j) * dx[1] + 0.5 * dx[1];
			const Real zc = prob_lo[2] + static_cast<Real>(k) * dx[2] + 0.5 * dx[2];

			for (int n = 0; n < np; ++n) {
				Real x0 = std::abs(xc - px(n));
				Real y0 = std::abs(yc - py(n));
				Real z0 = std::abs(zc - pz(n));

				if (x0 < 0.5 * dx[0] && y0 < 0.5 * dx[1] && z0 < 0.5 * dx[2]) {
					state(i, j, k, HydroSystem<NewProblem>::density_index) += rho_blast;
					state(i, j, k, HydroSystem<NewProblem>::energy_index) += rho_eint_blast;
					state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index) += rho_eint_blast;
					state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex) += scalar_blast;

					printf("The total number of SN gone off=%d\n", cum_sn);
					Real Rpds = 14. * std::pow(state(i, j, k, HydroSystem<NewProblem>::density_index) / Const_mH, -3. / 7.);
					printf("Rpds = %.2e pc\n", Rpds);
				}
			}
				//Add SN1a
				for (int n = 0; n < np1a; ++n) {
				Real x0 = std::abs(xc - px1a(n));
				Real y0 = std::abs(yc - py1a(n));
				Real z0 = std::abs(zc - pz1a(n));

				if (x0 < 0.5 * dx[0] && y0 < 0.5 * dx[1] && z0 < 0.5 * dx[2]) {

					state(i, j, k, HydroSystem<NewProblem>::density_index) += rho_blast;
					state(i, j, k, HydroSystem<NewProblem>::energy_index) += rho_eint_blast;
					state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index) += rho_eint_blast;
					state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+1) += scalar_blast;

					printf("The total number of SN gone off=%d\n", cum_sn);
					Real Rpds = 14. * std::pow(state(i, j, k, HydroSystem<NewProblem>::density_index) / Const_mH, -3. / 7.);
					printf("Rpds (SN1a) = %.2e pc\n", Rpds);
				}
			}

			//Add AGB
				for (int n = 0; n < npAGB; ++n) {
				Real x0 = std::abs(xc - pxAGB(n));
				Real y0 = std::abs(yc - pyAGB(n));
				Real z0 = std::abs(zc - pzAGB(n));

				if (x0 < 0.5 * dx[0] && y0 < 0.5 * dx[1] && z0 < 0.5 * dx[2]) {
					double rho0 = state(i, j, k, HydroSystem<NewProblem>::density_index) ;

					Real const Eint = quokka::EOS<NewProblem>::ComputeEintFromTgas(rho0, Tgas0);

					state(i, j, k, HydroSystem<NewProblem>::density_index) += rho_blast_AGB;
					state(i, j, k, HydroSystem<NewProblem>::energy_index) += Eint;
					state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index) += Eint;
					state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+2) += scalar_blast;

					printf("The total number of SN gone off=%d\n", cum_sn);
					Real Rpds = 14. * std::pow(state(i, j, k, HydroSystem<NewProblem>::density_index) / Const_mH, -3. / 7.);
					printf("Rpds (AGB) = %.2e pc\n", Rpds);
				}
			}


		});
	}
}

template <> void QuokkaSimulation<NewProblem>::computeBeforeTimestep()
{
	// compute how many (and where) SNe will go off on the this coarse timestep
	// sample from Poisson distribution

	const Real dt_coarse = dt_[0];
	const Real domain_area = geom[0].ProbLength(0) * geom[0].ProbLength(1);
	const Real mean = 0.0;
	const Real stddev = hscale / geom[0].ProbLength(2) / 2.;
	const Real stddev1a = hscaleIa / geom[0].ProbLength(2) / 2.;
	const Real stddevAGB = hscaleAGB / geom[0].ProbLength(2) / 2.;

	const Real expectation_value = ks_sigma_sfr*0.6 * domain_area * dt_coarse;

	const Real expectation_value1a = (ks_sigma_sfr*0.4) * domain_area * dt_coarse; 

	const Real expectation_valueAGB = (ks_sigma_sfr*16.0) * domain_area * dt_coarse;

	const int count = static_cast<int>(amrex::RandomPoisson(expectation_value));
	const int count1a = static_cast<int>(amrex::RandomPoisson(expectation_value1a));
	const int countAGB = static_cast<int>(amrex::RandomPoisson(expectation_valueAGB));

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
		pz(i) = 2.*kpc;;
		while(1.*kpc < pz(i)){
			pz(i) = geom[0].ProbLength(2) * amrex::RandomNormal(mean, stddev);
		}
	}
	//Get probablities for Type Ias
	amrex::Array<int, 1> const hi1a{count1a};
	userData_.blast_x1a = std::make_unique<amrex::TableData<Real, 1>>(lo, hi1a, amrex::The_Pinned_Arena());
	userData_.blast_y1a = std::make_unique<amrex::TableData<Real, 1>>(lo, hi1a, amrex::The_Pinned_Arena());
	userData_.blast_z1a = std::make_unique<amrex::TableData<Real, 1>>(lo, hi1a, amrex::The_Pinned_Arena());
	userData_.nblast1a = count1a;
	userData_.SN_counter_cumulative +=  count1a;

	auto const &px1a = userData_.blast_x1a->table();
	auto const &py1a = userData_.blast_y1a->table();
	auto const &pz1a = userData_.blast_z1a->table();
	for (int i = 0; i < count1a; ++i) {
		px1a(i) = geom[0].ProbLength(0) * amrex::Random();
		py1a(i) = geom[0].ProbLength(1) * amrex::Random();
		pz1a(i) = 2.*kpc;
		while(1.*kpc < pz1a(i)){
			pz1a(i) = geom[0].ProbLength(2) * amrex::RandomNormal(mean, stddev1a);
		}

	}

	//Get probablities for AGBs
	amrex::Array<int, 1> const hiAGB{countAGB};
	userData_.blast_xAGB = std::make_unique<amrex::TableData<Real, 1>>(lo, hiAGB, amrex::The_Pinned_Arena());
	userData_.blast_yAGB = std::make_unique<amrex::TableData<Real, 1>>(lo, hiAGB, amrex::The_Pinned_Arena());
	userData_.blast_zAGB = std::make_unique<amrex::TableData<Real, 1>>(lo, hiAGB, amrex::The_Pinned_Arena());
	userData_.nblastAGB = countAGB;
	userData_.SN_counter_cumulative +=  countAGB ;

	auto const &pxAGB = userData_.blast_xAGB->table();
	auto const &pyAGB = userData_.blast_yAGB->table();
	auto const &pzAGB = userData_.blast_zAGB->table();
	for (int i = 0; i < countAGB; ++i) {
		pxAGB(i) = geom[0].ProbLength(0) * amrex::Random();
		pyAGB(i) = geom[0].ProbLength(1) * amrex::Random();
		pzAGB(i) = 2.*kpc;
		while(1.*kpc < pzAGB(i)){
			pzAGB(i) = geom[0].ProbLength(2) * amrex::RandomNormal(mean, stddevAGB);
		}

	}

}

template <> void QuokkaSimulation<NewProblem>::computeAfterLevelAdvance(int lev, amrex::Real time, amrex::Real dt_lev, int ncycle)
{
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom[lev].ProbLoArray();
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = geom[lev].ProbHiArray();
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx = geom[lev].CellSizeArray();

	AddSupernova(state_new_cc_[lev], prob_lo, prob_hi, dx, userData_, lev);
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto
HydroSystem<NewProblem>::GetGradFixedPotential(amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> posvec) -> amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>
{

	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> grad_potential;
	grad_potential[0] = 0.0;
	grad_potential[1] = 0.0;

	double z = posvec[2];

	// Interpolate to find the accurate g-value from array-- because linterp doesn't work on Setonix
	size_t i = 0;
	double x_interp = std::abs(z);
	while (i < z_data.size() - 1 && x_interp > z_data[i + 1]) {
		i++;
	}

	// Perform linear interpolation
	const Real x1 = z_data[i];
	const Real x2 = z_data[i + 1];
	const Real y1 = logg_data[i];
	const Real y2 = logg_data[i + 1];

	amrex::Real ginterp = (y1 + (y2 - y1) * (x_interp - x1) / (x2 - x1));

	grad_potential[2] = 2. * M_PI * Const_G * rho_dm * std::pow(R0, 2) * (2. * z / std::pow(R0, 2)) / (1. + std::pow(z, 2) / std::pow(R0, 2));
	grad_potential[2] += 2. * M_PI * Const_G * Sigma_star * (z / z_star) * (std::pow(1. + z * z / (z_star * z_star), -0.5));
	grad_potential[2] += (z / std::abs(z)) * std::pow(10., ginterp);

	return grad_potential;
}

// Add Strang Split Source Term for External Fixed Potential Here
template <> void QuokkaSimulation<NewProblem>::addStrangSplitSources(amrex::MultiFab &mf, int lev, amrex::Real time, amrex::Real dt_lev)
{
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom[lev].ProbLoArray();
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx = geom[lev].CellSizeArray();
	const Real dt = dt_lev;

	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = mf.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> posvec, GradPhi;
			double x1mom_new, x2mom_new, x3mom_new;

			const Real rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
			const Real x1mom = state(i, j, k, HydroSystem<NewProblem>::x1Momentum_index);
			const Real x2mom = state(i, j, k, HydroSystem<NewProblem>::x2Momentum_index);
			const Real x3mom = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index);
			const Real Egas = state(i, j, k, HydroSystem<NewProblem>::energy_index);

			Real Eint = RadSystem<NewProblem>::ComputeEintFromEgas(rho, x1mom, x2mom, x3mom, Egas);

			posvec[0] = prob_lo[0] + (i + 0.5) * dx[0];

#if (AMREX_SPACEDIM >= 2)
			posvec[1] = prob_lo[1] + (j + 0.5) * dx[1];
#endif

#if (AMREX_SPACEDIM >= 3)
			posvec[2] = prob_lo[2] + (k + 0.5) * dx[2];
#endif

			GradPhi = HydroSystem<NewProblem>::GetGradFixedPotential(posvec);

			x1mom_new = x1mom + dt * (-rho * GradPhi[0]);
			x2mom_new = x2mom + dt * (-rho * GradPhi[1]);
			x3mom_new = x3mom + dt * (-rho * GradPhi[2]);

			// State momentum values need to be updated this way.
			state(i, j, k, HydroSystem<NewProblem>::x1Momentum_index) = x1mom_new;
			state(i, j, k, HydroSystem<NewProblem>::x2Momentum_index) = x2mom_new;
			state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) = x3mom_new;

			state(i, j, k, HydroSystem<NewProblem>::energy_index) =
			    RadSystem<NewProblem>::ComputeEgasFromEint(rho, x1mom_new, x2mom_new, x3mom_new, Eint);
		});
	}
}

// Code for producing in-situ Projection plots
template <> auto QuokkaSimulation<NewProblem>::ComputeProjections(const int dir) const -> std::unordered_map<std::string, amrex::BaseFab<amrex::Real>>
{
	// compute density projection
	std::unordered_map<std::string, amrex::BaseFab<amrex::Real>> proj;

	proj["mass_outflow"] = computePlaneProjection<amrex::ReduceOpSum>(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    // int nmscalars = Physics_Traits<NewProblem>::numMassScalars;
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const vx3 = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) / rho;
		    return (rho * vx3);
	    },
	    dir);

	proj["hot_mass_outflow"] = computePlaneProjection<amrex::ReduceOpSum>(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    double flux;
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const vx3 = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) / rho;
		    Real const Eint = state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index);
		    amrex::GpuArray<Real, 0> massScalars = RadSystem<NewProblem>::ComputeMassScalars(state, i, j, k);
		    Real const primTemp = quokka::EOS<NewProblem>::ComputeTgasFromEint(rho, Eint, massScalars);
		    if (primTemp > 5.e5) {
			    flux = rho * vx3;
		    } else {
			    flux = 0.0;
		    }
		    return flux;
	    },
	    dir);

	proj["warm_mass_outflow"] = computePlaneProjection<amrex::ReduceOpSum>(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    double flux;
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const vx3 = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) / rho;
		    Real const Eint = state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index);
		    amrex::GpuArray<Real, 0> massScalars = RadSystem<NewProblem>::ComputeMassScalars(state, i, j, k);
		    Real const primTemp = quokka::EOS<NewProblem>::ComputeTgasFromEint(rho, Eint, massScalars);
		    if (primTemp < 2.e4) {
			    flux = rho * vx3;
		    } else {
			    flux = 0.0;
		    }
		    return flux;
	    },
	    dir);

	proj["scalar0_outflow"] = computePlaneProjection<amrex::ReduceOpSum>(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex);
		    Real const vz = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) / rho;
		    return (rhoZ * vz);
	    },
	    dir);

	proj["warm_scalar0_outflow"] = computePlaneProjection<amrex::ReduceOpSum>(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    double flux;
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex);
		    Real const vx3 = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) / rho;
		    Real const Eint = state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index);
		    amrex::GpuArray<Real, 0> massScalars = RadSystem<NewProblem>::ComputeMassScalars(state, i, j, k);
		    Real const primTemp = quokka::EOS<NewProblem>::ComputeTgasFromEint(rho, Eint, massScalars);
		    if (primTemp < 2.e4) {
			    flux = rhoZ * vx3;
		    } else {
			    flux = 0.0;
		    }
		    return flux;
	    },
	    dir);

	proj["hot_scalar0_outflow"] = computePlaneProjection<amrex::ReduceOpSum>(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    double flux;
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex);
		    Real const vx3 = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) / rho;
		    Real const Eint = state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index);
		    amrex::GpuArray<Real, 0> massScalars = RadSystem<NewProblem>::ComputeMassScalars(state, i, j, k);
		    Real const primTemp = quokka::EOS<NewProblem>::ComputeTgasFromEint(rho, Eint, massScalars);
		    if (primTemp > 1.e6) {
			    flux = rhoZ * vx3;
		    } else {
			    flux = 0.0;
		    }
		    return flux;
	    },
	    dir);

	proj["scalar1_outflow"] = computePlaneProjection<amrex::ReduceOpSum>(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+1);
		    Real const vz = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) / rho;
		    return (rhoZ * vz);
	    },
	    dir);

	proj["warm_scalar1_outflow"] = computePlaneProjection<amrex::ReduceOpSum>(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    double flux;
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+1);
		    Real const vx3 = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) / rho;
		    Real const Eint = state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index);
		    amrex::GpuArray<Real, 0> massScalars = RadSystem<NewProblem>::ComputeMassScalars(state, i, j, k);
		    Real const primTemp = quokka::EOS<NewProblem>::ComputeTgasFromEint(rho, Eint, massScalars);
		    if (primTemp < 2.e4) {
			    flux = rhoZ * vx3;
		    } else {
			    flux = 0.0;
		    }
		    return flux;
	    },
	    dir);

	proj["hot_scalar1_outflow"] = computePlaneProjection<amrex::ReduceOpSum>(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    double flux;
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+1);
		    Real const vx3 = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) / rho;
		    Real const Eint = state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index);
		    amrex::GpuArray<Real, 0> massScalars = RadSystem<NewProblem>::ComputeMassScalars(state, i, j, k);
		    Real const primTemp = quokka::EOS<NewProblem>::ComputeTgasFromEint(rho, Eint, massScalars);
		    if (primTemp > 1.e6) {
			    flux = rhoZ * vx3;
		    } else {
			    flux = 0.0;
		    }
		    return flux;
	    },
	    dir);


	proj["scalar2_outflow"] = computePlaneProjection<amrex::ReduceOpSum>(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+2);
		    Real const vz = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) / rho;
		    return (rhoZ * vz);
	    },
	    dir);

	proj["warm_scalar2_outflow"] = computePlaneProjection<amrex::ReduceOpSum>(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    double flux;
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+2);
		    Real const vx3 = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) / rho;
		    Real const Eint = state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index);
		    amrex::GpuArray<Real, 0> massScalars = RadSystem<NewProblem>::ComputeMassScalars(state, i, j, k);
		    Real const primTemp = quokka::EOS<NewProblem>::ComputeTgasFromEint(rho, Eint, massScalars);
		    if (primTemp < 2.e4) {
			    flux = rhoZ * vx3;
		    } else {
			    flux = 0.0;
		    }
		    return flux;
	    },
	    dir);

	proj["hot_scalar2_outflow"] = computePlaneProjection<amrex::ReduceOpSum>(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    double flux;
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+2);
		    Real const vx3 = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) / rho;
		    Real const Eint = state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index);
		    amrex::GpuArray<Real, 0> massScalars = RadSystem<NewProblem>::ComputeMassScalars(state, i, j, k);
		    Real const primTemp = quokka::EOS<NewProblem>::ComputeTgasFromEint(rho, Eint, massScalars);
		    if (primTemp > 1.e6) {
			    flux = rhoZ * vx3;
		    } else {
			    flux = 0.0;
		    }
		    return flux;
	    },
	    dir);


	proj["rho"] = computePlaneProjection<amrex::ReduceOpSum>(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    return (rho);
	    },
	    dir);

	proj["scalar0"] = computePlaneProjection<amrex::ReduceOpSum>(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex);
		    return (rhoZ);
	    },
	    dir);
	proj["scalar1"] = computePlaneProjection<amrex::ReduceOpSum>(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+1);
		    return (rhoZ);
	    },
	    dir);

	proj["scalar2"] = computePlaneProjection<amrex::ReduceOpSum>(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex+2);
		    return (rhoZ);
	    },
	    dir);	
	return proj;
}

// Implement User-defined diode BC
template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void AMRSimulation<NewProblem>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<Real> const &consVar,
												int /*dcomp*/, int /*numcomp*/, amrex::GeometryData const &geom,
												const Real /*time*/, const amrex::BCRec * /*bcr*/,
												int /*bcomp*/, int /*orig_comp*/)
{
	auto [i, j, k] = iv.dim3();
	amrex::Box const &box = geom.Domain();
	const auto &domain_lo = box.loVect3d();
	const auto &domain_hi = box.hiVect3d();
	const int klo = domain_lo[2];
	const int khi = domain_hi[2];
	int kedge = 0;
	int normal = 0;

	if (k < klo) {
		kedge = klo;
		normal = -1;
	} else if (k > khi) {
		kedge = khi;
		normal = 1.0;
	}

	const double rho_edge = consVar(i, j, kedge, HydroSystem<NewProblem>::density_index);
	const double x1Mom_edge = consVar(i, j, kedge, HydroSystem<NewProblem>::x1Momentum_index);
	const double x2Mom_edge = consVar(i, j, kedge, HydroSystem<NewProblem>::x2Momentum_index);
	double x3Mom_edge = consVar(i, j, kedge, HydroSystem<NewProblem>::x3Momentum_index);
	const double etot_edge = consVar(i, j, kedge, HydroSystem<NewProblem>::energy_index);
	const double eint_edge = consVar(i, j, kedge, HydroSystem<NewProblem>::internalEnergy_index);
	const double pscalar0_edge = consVar(i, j, kedge, HydroSystem<NewProblem>::scalar0_index);
	const double pscalar1_edge = consVar(i, j, kedge, HydroSystem<NewProblem>::scalar0_index+1);
	const double pscalar2_edge = consVar(i, j, kedge, HydroSystem<NewProblem>::scalar0_index+2);

	if ((x3Mom_edge * normal) < 0) { // gas is inflowing
		x3Mom_edge = -1. * consVar(i, j, kedge, HydroSystem<NewProblem>::x3Momentum_index);
	}

	consVar(i, j, k, HydroSystem<NewProblem>::density_index) = rho_edge;
	consVar(i, j, k, HydroSystem<NewProblem>::x1Momentum_index) = x1Mom_edge;
	consVar(i, j, k, HydroSystem<NewProblem>::x2Momentum_index) = x2Mom_edge;
	consVar(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) = x3Mom_edge;
	consVar(i, j, k, HydroSystem<NewProblem>::energy_index) = etot_edge;
	consVar(i, j, k, HydroSystem<NewProblem>::internalEnergy_index) = eint_edge;
	consVar(i, j, k, HydroSystem<NewProblem>::scalar0_index) = pscalar0_edge;
	consVar(i, j, k, HydroSystem<NewProblem>::scalar0_index+1) = pscalar1_edge;
	consVar(i, j, k, HydroSystem<NewProblem>::scalar0_index+2) = pscalar2_edge;
}


auto problem_main() -> int
{

	const int ncomp_cc = Physics_Indices<NewProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);

	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			// diode boundary conditions
			if (i == 2) {
				BCs_cc[n].setLo(i, amrex::BCType::ext_dir);
				BCs_cc[n].setHi(i, amrex::BCType::ext_dir);
			} else {
				BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
				BCs_cc[n].setHi(i, amrex::BCType::int_dir); // periodic
			}
		}
	}

	// set random state
	const int seed = 42;
	amrex::InitRandom(seed, 1); // all ranks should produce the same values

	// Problem initialization
	QuokkaSimulation<NewProblem> sim(BCs_cc);

	sim.reconstructionOrder_ = 3; // 2=PLM, 3=PPM
	sim.cflNumber_ = 0.3;	      // *must* be less than 1/3 in 3D!

	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// Cleanup and exit
	amrex::Print() << "Finished." << std::endl;
	return 0;
}
