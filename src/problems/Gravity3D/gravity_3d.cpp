/// \file gravity_3d.cpp
/// \brief Defines a test problem for self-gravity in 3D.
///

#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_Config.H"
#include "AMReX_DistributionMapping.H"
#include "AMReX_FabArrayUtility.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

#include "AMReX_REAL.H"
#include "QuokkaSimulation.hpp"
#include "gravity_3d.hpp"
#include "hydro/hydro_system.hpp"
#include <algorithm>

struct BinaryOrbit {
};

template <> struct quokka::EOS_Traits<BinaryOrbit> {
	static constexpr double gamma = 1.0;	     // isothermal
	static constexpr double cs_isothermal = 3.0; //
	static constexpr double mean_molecular_weight = 1.0;
};

// Test enum to demonstrate type checking of particle_switch
enum class TestEnum : unsigned int {
	MISTAKE = 0b00000100U,
};

template <> struct Particle_Traits<BinaryOrbit> {
	// The following will cause a compile error
	// static constexpr int particle_switch = 1;
	// static constexpr TestEnum particle_switch = TestEnum::MISTAKE;
	// static constexpr ParticleSwitch particle_switch = ParticleSwitch::CIC | TestEnum::MISTAKE;
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::StellarPop;
};

template <> struct HydroSystem_Traits<BinaryOrbit> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<BinaryOrbit> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr int nGroups = 1;			     // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = 1.0;
	static constexpr double radiation_constant = 1.0;
};

template <> void QuokkaSimulation<BinaryOrbit>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		double const rho = 1.0e-5; // g cm^{-3}
		state_cc(i, j, k, HydroSystem<BinaryOrbit>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<BinaryOrbit>::x1Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<BinaryOrbit>::x2Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<BinaryOrbit>::x3Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<BinaryOrbit>::energy_index) = 0;
		state_cc(i, j, k, HydroSystem<BinaryOrbit>::internalEnergy_index) = 0;
	});
}

template <> void QuokkaSimulation<BinaryOrbit>::computeAfterEvolve(amrex::Vector<amrex::Real> &initSumCons) {}

// Integer particle data is not supported by InitialFromAsciiFile yet
// template <> void QuokkaSimulation<BinaryOrbit>::createInitialStellarPopParticles()
// {
// 	// read particles from ASCII file
// 	const int nreal_extra = 4; // mass vx vy vz
// 	const int nint_extra = 1; // fate
// 	StellarPopParticles->SetVerbose(1);
// 	StellarPopParticles->InitFromAsciiFile("Gravity3D.txt", nreal_extra, nint_extra);
// }

auto problem_main() -> int
{
	auto isNormalComp = [=](int n, int dim) {
		if ((n == HydroSystem<BinaryOrbit>::x1Momentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == HydroSystem<BinaryOrbit>::x2Momentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == HydroSystem<BinaryOrbit>::x3Momentum_index) && (dim == 2)) {
			return true;
		}
		return false;
	};

	const int ncomp_cc = Physics_Indices<BinaryOrbit>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			if (isNormalComp(n, i)) {
				BCs_cc[n].setLo(i, amrex::BCType::reflect_odd);
				BCs_cc[n].setHi(i, amrex::BCType::reflect_odd);
			} else {
				BCs_cc[n].setLo(i, amrex::BCType::reflect_even);
				BCs_cc[n].setHi(i, amrex::BCType::reflect_even);
			}
		}
	}

	// Problem initialization
	QuokkaSimulation<BinaryOrbit> sim(BCs_cc);
	sim.doPoissonSolve_ = 1; // enable self-gravity
	// sim.initDt_ = 1.0e-2;	 // s

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// exact solution
	const double theta = 0.5 * sim.tNew_[0];
	const double exact_x = 1.0 * std::cos(theta);
	const double exact_y = 1.0 * std::sin(theta);
	const double exact_z = 0.0;

	double position_error = 0.0;
	double position_norm = 0.0;

	const int n_particle_expect = (3 * 3 * 3) * 1; // 2 particles from the initial condition, (3*3*3) particles from the creator

	int status = 0; // Initialize to success

	auto particle_data = sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::StellarPop)->getParticleData(0);

	if (amrex::ParallelDescriptor::IOProcessor()) {

		const auto n_particle_actual = particle_data.size();

		amrex::Print() << "Expected number of particles: " << n_particle_expect << "\n";
		amrex::Print() << "Actual number of particles: " << n_particle_actual << "\n";

		status = 1;
		if (n_particle_actual == n_particle_expect) {
			status = 0;
			amrex::Print() << "Number of particles matches expected.\n";
		}
	}

	return status;
}
