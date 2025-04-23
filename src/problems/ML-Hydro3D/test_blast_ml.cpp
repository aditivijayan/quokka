//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro3d_blast.cpp
/// \brief Defines a test problem for a 3D explosion.
///

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_SPACE.H"

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "radiation/radiation_system.hpp"
#include "test_blast_ml.hpp"
#include "math/quadrature.hpp"

struct SedovProblem {
};

// if false, use octant symmetry instead
constexpr bool simulate_full_box = false;

bool test_passes = false; // if one of the energy checks fails, set to false

template <> struct quokka::EOS_Traits<SedovProblem> {
	static constexpr double gamma = 1.4;
	static constexpr double mean_molecular_weight = C::m_u;
	static constexpr double boltzmann_constant = C::k_B;
};

template <> struct HydroSystem_Traits<SedovProblem> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<SedovProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 2; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1; // number of radiation groups
};

#if 0 // workaround AMDGPU compiler bug
namespace
{
#endif
Real rho0 = NAN;										// NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
int nsmooth  = 0;										// NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
#if 0												// workaround AMDGPU compiler bug
};											 // namespace
#endif
double E_blast = 1.e51; // ergs

template <> void QuokkaSimulation<SedovProblem>::preCalculateInitialConditions()
{
	if constexpr (!simulate_full_box) {
		E_blast /= 8.0; // only one octant, so 1/8 of the total energy
	}
}

template <> void QuokkaSimulation<SedovProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// initialize a Sedov test problem using parameters from
	// Richard Klein and J. Bolstad
	// [Reference: J.R. Kamm and F.X. Timmes, On Efficient Generation of
	//   Numerically Robust Sedov Solutions, LA-UR-07-2849.]

	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const Real cell_vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
	double rho_copy = rho0 * Const_mH;
	double E_blast_copy = E_blast;
	double rho_S0 = 1.e3;
	amrex::Real x0 = NAN;
	amrex::Real y0 = NAN;
	amrex::Real z0 = NAN;
	if constexpr (simulate_full_box) {
		x0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);
		y0 = prob_lo[1] + 0.5 * (prob_hi[1] - prob_lo[1]);
		z0 = prob_lo[2] + 0.5 * (prob_hi[2] - prob_lo[2]);
	} else {
		x0 = 0.;
		y0 = 0.;
		z0 = 0.;
	}

	const Real r_scale = nsmooth * dx[0]; // TODO(ben): cannot be based on local dx when using AMR!
	const Real normfac = 1.0 / std::pow(r_scale, 3);

	auto kern = [=] AMREX_GPU_DEVICE(const Real x, const Real y, const Real z) {
			const Real r = std::sqrt(x * x + y * y + z * z);
			return kernel_wendland_c2(r / r_scale);
		};


	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		static_assert(!simulate_full_box, "single-cell initialization is only "
						  "implemented for octant symmetry!");
		double rho_e = NAN;
		double rho_Sinj = NAN;
		double rho_Samb = NAN;

		// if ((i == 0) && (j == 0) && (k == 0)) {
		// 	rho_e = E_blast_copy / cell_vol;
		// 	rho_Sinj = rho_S0 / cell_vol;
		// 	rho_Samb = 1.0e-10 * (rho_S0 / cell_vol);
		// } else {
		// 	rho_e    = 1.0e-10 * (E_blast_copy / cell_vol);
		// 	rho_Sinj = 1.0e-10 * (rho_S0 / cell_vol);
		// 	rho_Samb = (rho_S0 / cell_vol);
		// }

		rho_e = E_blast_copy / cell_vol;
		rho_Samb = (rho_S0 / cell_vol);
		rho_Sinj = rho_S0 / cell_vol;

		const Real xc = prob_lo[0] + static_cast<Real>(i) * dx[0];
		const Real yc = prob_lo[1] + static_cast<Real>(j) * dx[1];
		const Real zc = prob_lo[2] + static_cast<Real>(k) * dx[2];

		const Real weight = normfac * quad_3d(kern, xc, xc + dx[0], yc, yc + dx[1], zc, zc + dx[2]);
		const auto gamma = HydroSystem<SedovProblem>::gamma_;
		double P = rho_copy * std::pow(sigma, 2.0);

        // const Real weight = 1.0 ;
		if ((i == 0) && (j == 0) && (k == 0)) { 
				printf("weight, rhoe=%.3e, %.3e\n",weight, rho_e);
		}

		AMREX_ASSERT(!std::isnan(rho_copy));
		AMREX_ASSERT(!std::isnan(rho_e));

		for (int n = 0; n < state_cc.nComp(); ++n) {
			state_cc(i, j, k, n) = 0.; // zero fill all components
		}

	
		state_cc(i, j, k, HydroSystem<SedovProblem>::density_index) =  rho_copy;
		state_cc(i, j, k, HydroSystem<SedovProblem>::x1Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<SedovProblem>::x2Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<SedovProblem>::x3Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<SedovProblem>::energy_index) = weight * rho_e + P/(gamma - 1.);
		state_cc(i, j, k, HydroSystem<SedovProblem>::internalEnergy_index) = weight * rho_e  + P/(gamma - 1.);
		state_cc(i, j, k, Physics_Indices<SedovProblem>::pscalarFirstIndex) = weight * rho_Sinj;
		state_cc(i, j, k, Physics_Indices<SedovProblem>::pscalarFirstIndex+1) = (1.-weight) * rho_Samb;
	});
}

auto problem_main() -> int
{
	auto isNormalComp = [=](int n, int dim) {
		if ((n == HydroSystem<SedovProblem>::x1Momentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == HydroSystem<SedovProblem>::x2Momentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == HydroSystem<SedovProblem>::x3Momentum_index) && (dim == 2)) {
			return true;
		}
		return false;
	};

	const int ncomp_cc = Physics_Indices<SedovProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			if constexpr (simulate_full_box) { // periodic boundaries
				BCs_cc[n].setLo(i, amrex::BCType::int_dir);
				BCs_cc[n].setHi(i, amrex::BCType::int_dir);
			} else { // octant symmetry
				if (isNormalComp(n, i)) {
					BCs_cc[n].setLo(i, amrex::BCType::reflect_odd);
					BCs_cc[n].setHi(i, amrex::BCType::reflect_odd);
				} else {
					BCs_cc[n].setLo(i, amrex::BCType::reflect_even);
					BCs_cc[n].setHi(i, amrex::BCType::reflect_even);
				}
			}
		}
	}

	

	// Problem initialization
	QuokkaSimulation<SedovProblem> sim(BCs_cc);

	amrex::ParmParse const pp("blast");
	// initial condition parameters
	pp.query("rho0", ::rho0);   // initial density [g/cc]
	pp.query("nsmooth", nsmooth);   // initial density [g/cc]

	sim.reconstructionOrder_ = 3; // 2=PLM, 3=PPM
	sim.cflNumber_ = 0.3;	      // *must* be less than 1/3 in 3D!

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	return 0;
}
