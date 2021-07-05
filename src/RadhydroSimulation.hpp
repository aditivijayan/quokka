#ifndef RADIATION_SIMULATION_HPP_ // NOLINT
#define RADIATION_SIMULATION_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file RadhydroSimulation.hpp
/// \brief Implements classes and functions to organise the overall setup,
/// timestepping, solving, and I/O of a simulation for radiation moments.

#include <array>
#include <climits>
#include <limits>
#include <string>
#include <tuple>
#include <utility>

#include "AMReX_Algorithm.H"
#include "AMReX_Arena.H"
#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_BCRec.H"
#include "AMReX_BLassert.H"
#include "AMReX_Box.H"
#include "AMReX_Config.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_GpuControl.H"
#include "AMReX_IntVect.H"
#include "AMReX_MultiFab.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_PhysBCFunct.H"
#include "AMReX_REAL.H"
#include "AMReX_Utility.H"

#include "hydro_system.hpp"
#include "radiation_system.hpp"
#include "simulation.hpp"

// Simulation class should be initialized only once per program (i.e., is a singleton)
template <typename problem_t> class RadhydroSimulation : public AMRSimulation<problem_t>
{
      public:
	using AMRSimulation<problem_t>::state_old_;
	using AMRSimulation<problem_t>::state_new_;
	using AMRSimulation<problem_t>::max_signal_speed_;

	using AMRSimulation<problem_t>::ncomp_;
	using AMRSimulation<problem_t>::nghost_;
	using AMRSimulation<problem_t>::areInitialConditionsDefined_;
	using AMRSimulation<problem_t>::boundaryConditions_;
	using AMRSimulation<problem_t>::componentNames_;

	using AMRSimulation<problem_t>::fillBoundaryConditions;
	using AMRSimulation<problem_t>::geom;
	using AMRSimulation<problem_t>::flux_reg_;
	using AMRSimulation<problem_t>::incrementFluxRegisters;
	using AMRSimulation<problem_t>::finest_level;

	std::vector<double> t_vec_;
	std::vector<double> Trad_vec_;
	std::vector<double> Tgas_vec_;

	static constexpr int nvarTotal_ = RadSystem<problem_t>::nvar_;
	static constexpr int ncompHydro_ = HydroSystem<problem_t>::nvar_; // hydro
	static constexpr int ncompHyperbolic_ = RadSystem<problem_t>::nvarHyperbolic_;
	static constexpr int nstartHyperbolic_ = RadSystem<problem_t>::nstartHyperbolic_;

	// amrex::Real dtRadiation_ = NAN; // this is radiation subcycle timestep
	amrex::Real radiationCflNumber_ = 0.3;
	bool is_hydro_enabled_ = false;
	bool is_radiation_enabled_ = true;

	// member functions

	RadhydroSimulation(amrex::IntVect &gridDims, amrex::RealBox &boxSize,
			   amrex::Vector<amrex::BCRec> &boundaryConditions)
	    : AMRSimulation<problem_t>(gridDims, boxSize, boundaryConditions,
				       RadSystem<problem_t>::nvar_, ncompHyperbolic_)
	{
		componentNames_ = {"gasDensity",    "x-GasMomentum", "y-GasMomentum",
				   "z-GasMomentum", "gasEnergy",     "radEnergy",
				   "x-RadFlux",	    "y-RadFlux",     "z-RadFlux"};
	}

	void computeMaxSignalLocal(int level) override;
	void setInitialConditionsAtLevel(int level) override;
	void advanceSingleTimestepAtLevel(int lev, amrex::Real time, amrex::Real dt_lev,
					  int iteration, int ncycle) override;
	void computeAfterTimestep() override;
	// tag cells for refinement
	void ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real time, int ngrow) override;

	// radiation subcycle
	void advanceSingleTimestepAtLevelRadiation(int lev, amrex::Real time,
						   amrex::Real dt_radiation);
	void subcycleRadiationAtLevel(int lev, amrex::Real time, amrex::Real dt_lev_hydro);

	void operatorSplitSourceTerms(amrex::Array4<amrex::Real> const &stateNew,
				      const amrex::Box &indexRange, int /*nvars*/, amrex::Real time,
				      double dt, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx);

	auto computeRadiationFluxes(amrex::Array4<const amrex::Real> const &consVar,
				    const amrex::Box &indexRange, int nvars,
				    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx)
	    -> std::tuple<std::array<amrex::FArrayBox, AMREX_SPACEDIM>,
			  std::array<amrex::FArrayBox, AMREX_SPACEDIM>>;

	auto computeHydroFluxes(amrex::Array4<const amrex::Real> const &consVar,
				const amrex::Box &indexRange, int nvars)
	    -> std::array<amrex::FArrayBox, AMREX_SPACEDIM>;

	template <FluxDir DIR>
	void fluxFunction(amrex::Array4<const amrex::Real> const &consState,
			  amrex::FArrayBox &x1Flux, amrex::FArrayBox &x1FluxDiffusive,
			  const amrex::Box &indexRange, int nvars,
			  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx);

	template <FluxDir DIR>
	void hydroFluxFunction(amrex::Array4<const amrex::Real> const &consState,
			       amrex::FArrayBox &x1Flux, const amrex::Box &indexRange, int nvars);
};

template <typename problem_t>
void RadhydroSimulation<problem_t>::computeMaxSignalLocal(int const level)
{
	// hydro: loop over local grids, compute hydro CFL timestep
	for (amrex::MFIter iter(state_new_[level]); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateNew = state_new_[level].const_array(iter);
		auto const &maxSignal = max_signal_speed_[level].array(iter);
		HydroSystem<problem_t>::ComputeMaxSignalSpeed(stateNew, maxSignal, indexRange);
	}
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::setInitialConditionsAtLevel(int level)
{
	// do nothing -- user should implement using problem-specific template specialization
}

template <typename problem_t> void RadhydroSimulation<problem_t>::computeAfterTimestep()
{
	// do nothing -- user should implement if desired
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::ErrorEst(int lev, amrex::TagBoxArray &tags,
					     amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement

	const amrex::Real epsilon_threshold = 0.01; // curvature refinement threshold

	for (amrex::MFIter mfi(state_new_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.tilebox();
		const auto state = state_new_[lev].const_array(mfi);
		const auto tag = tags.array(mfi);

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			int const n = HydroSystem<problem_t>::density_index;
			amrex::Real const delsq_x =
			    (state(i + 1, j, k, n) - 2.0 * state(i, j, k, n) +
			     state(i - 1, j, k, n));
			amrex::Real const delsq_y =
			    (state(i, j + 1, k, n) - 2.0 * state(i, j, k, n) +
			     state(i, j - 1, k, n));

			// estimate curvature from finite difference stencil
			// [see Athena++ paper]
			amrex::Real const epsilon = std::abs(delsq_x + delsq_y) / state(i, j, k, n);
			if (epsilon > epsilon_threshold) {
				tag(i, j, k) = amrex::TagBox::SET;
			}
		});
	}
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::advanceSingleTimestepAtLevel(int lev, amrex::Real time,
								 amrex::Real dt_lev,
								 int /*iteration*/, int /*ncycle*/)
{
	// based on amrex/Tests/EB/CNS/Source/CNS_advance.cpp

	// since we are starting a new timestep, need to swap old and new states on this level
	std::swap(state_old_[lev], state_new_[lev]);

	// get geometry (used only for cell sizes)
	auto const &geomLevel = geom[lev];

	// get flux registers
	amrex::YAFluxRegister &fr_as_crse = *flux_reg_[lev + 1];
	amrex::YAFluxRegister &fr_as_fine = *flux_reg_[lev];
	if (lev < finest_level) {
		fr_as_crse.reset(); // set flux register to zero
	}

	/// advance hydro
	if (is_hydro_enabled_) {
		// check state validity
		AMREX_ASSERT(!state_old_[lev].contains_nan(0, ncompHydro_));

		// update ghost zones [old timestep]
		fillBoundaryConditions(state_old_[lev], state_old_[lev], lev, time);

		// advance all grids on local processor (Stage 1 of integrator)
		for (amrex::MFIter iter(state_new_[lev]); iter.isValid(); ++iter) {
			const amrex::Box &indexRange =
			    iter.validbox(); // 'validbox' == exclude ghost zones
			auto const &stateOld = state_old_[lev].const_array(iter);
			auto const &stateNew = state_new_[lev].array(iter);
			auto fluxArrays = computeHydroFluxes(stateOld, indexRange, ncompHydro_);

			// Stage 1 of RK2-SSP
			HydroSystem<problem_t>::PredictStep(
			    stateOld, stateNew,
			    {AMREX_D_DECL(fluxArrays[0].const_array(), fluxArrays[1].const_array(),
					  fluxArrays[2].const_array())},
			    dt_lev, geomLevel.CellSizeArray(), indexRange, ncompHydro_);

			// increment flux registers
			// (N.B. flux arrays must have the same number of components as
			// state_new_[lev]!)
			incrementFluxRegisters(iter, fr_as_crse, fr_as_fine, fluxArrays, 0.5, lev,
					       dt_lev);
		}

		// update ghost zones [intermediate stage stored in state_new_]
		fillBoundaryConditions(state_new_[lev], state_new_[lev], lev, time + dt_lev);

		// advance all grids on local processor (Stage 2 of integrator)
		for (amrex::MFIter iter(state_new_[lev]); iter.isValid(); ++iter) {
			const amrex::Box &indexRange =
			    iter.validbox(); // 'validbox' == exclude ghost zones
			auto const &stateOld = state_old_[lev].const_array(iter);
			auto const &stateInter = state_new_[lev].const_array(iter);
			auto const &stateNew = state_new_[lev].array(iter);
			auto fluxArrays = computeHydroFluxes(stateInter, indexRange, ncompHydro_);

			// Stage 2 of RK2-SSP
			HydroSystem<problem_t>::AddFluxesRK2(
			    stateNew, stateOld, stateInter,
			    {AMREX_D_DECL(fluxArrays[0].const_array(), fluxArrays[1].const_array(),
					  fluxArrays[2].const_array())},
			    dt_lev, geomLevel.CellSizeArray(), indexRange, ncompHydro_);

			// increment flux registers
			// (N.B. flux arrays must have the same number of components as
			// state_new_[lev]!)
			incrementFluxRegisters(iter, fr_as_crse, fr_as_fine, fluxArrays, 0.5, lev,
					       dt_lev);
		}
	}

	// subcycle radiation
	if (is_radiation_enabled_) {
		subcycleRadiationAtLevel(lev, time, dt_lev);
	}
}

template <typename problem_t>
auto RadhydroSimulation<problem_t>::computeHydroFluxes(
    amrex::Array4<const amrex::Real> const &consVar, const amrex::Box &indexRange, const int nvars)
    -> std::array<amrex::FArrayBox, AMREX_SPACEDIM>
{
	amrex::Box const &x1FluxRange = amrex::surroundingNodes(indexRange, 0);
	amrex::FArrayBox x1Flux(x1FluxRange, nvars, amrex::The_Async_Arena()); // node-centered in x
#if (AMREX_SPACEDIM >= 2)
	amrex::Box const &x2FluxRange = amrex::surroundingNodes(indexRange, 1);
	amrex::FArrayBox x2Flux(x2FluxRange, nvars, amrex::The_Async_Arena()); // node-centered in y
#endif
#if (AMREX_SPACEDIM == 3)
	amrex::Box const &x3FluxRange = amrex::surroundingNodes(indexRange, 2);
	amrex::FArrayBox x3Flux(x3FluxRange, nvars, amrex::The_Async_Arena()); // node-centered in z
#endif

	AMREX_D_TERM(hydroFluxFunction<FluxDir::X1>(consVar, x1Flux, indexRange, nvars);
		     , hydroFluxFunction<FluxDir::X2>(consVar, x2Flux, indexRange, nvars);
		     , hydroFluxFunction<FluxDir::X3>(consVar, x3Flux, indexRange, nvars);)

	return {AMREX_D_DECL(std::move(x1Flux), std::move(x2Flux), std::move(x3Flux))};
}

template <typename problem_t>
template <FluxDir DIR>
void RadhydroSimulation<problem_t>::hydroFluxFunction(
    amrex::Array4<const amrex::Real> const &consState, amrex::FArrayBox &x1Flux,
    const amrex::Box &indexRange, const int nvars)
{
	int dir = 0;
	if constexpr (DIR == FluxDir::X1) {
		dir = 0;
	} else if constexpr (DIR == FluxDir::X2) {
		dir = 1;
	} else if constexpr (DIR == FluxDir::X3) {
		dir = 2;
	}

	// extend box to include ghost zones
	amrex::Box const &ghostRange = amrex::grow(indexRange, nghost_);
	// N.B.: A one-zone layer around the cells must be fully reconstructed in order for PPM to
	// work.
	amrex::Box const &reconstructRange = amrex::grow(indexRange, 1);
	amrex::Box const &flatteningRange = amrex::grow(indexRange, 2); // +1 greater than ppmRange
	amrex::Box const &x1ReconstructRange = amrex::surroundingNodes(reconstructRange, dir);

	amrex::FArrayBox primVar(ghostRange, nvars, amrex::The_Async_Arena()); // cell-centered
	amrex::FArrayBox x1Flat(ghostRange, nvars, amrex::The_Async_Arena());
	amrex::FArrayBox x1LeftState(x1ReconstructRange, nvars, amrex::The_Async_Arena());
	amrex::FArrayBox x1RightState(x1ReconstructRange, nvars, amrex::The_Async_Arena());

	// cell-centered kernel
	HydroSystem<problem_t>::ConservedToPrimitive(consState, primVar.array(), ghostRange);

	// mixed interface/cell-centered kernel
	HydroSystem<problem_t>::template ReconstructStatesPPM<DIR>(
	    primVar.array(), x1LeftState.array(), x1RightState.array(), reconstructRange,
	    x1ReconstructRange, nvars);

	// cell-centered kernel
	HydroSystem<problem_t>::template ComputeFlatteningCoefficients<DIR>(
	    primVar.array(), x1Flat.array(), flatteningRange);

	// cell-centered kernel
	HydroSystem<problem_t>::template FlattenShocks<DIR>(
	    primVar.array(), x1Flat.array(), x1LeftState.array(), x1RightState.array(),
	    reconstructRange, nvars);

	// interface-centered kernel
	amrex::Box const &x1FluxRange = amrex::surroundingNodes(indexRange, dir);
	HydroSystem<problem_t>::template ComputeFluxes<DIR>(
	    x1Flux.array(), x1LeftState.array(), x1RightState.array(),
	    x1FluxRange); // watch out for argument order!!
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::subcycleRadiationAtLevel(int lev, amrex::Real time,
							     amrex::Real dt_lev_hydro)
{
	auto const &dx = geom[lev].CellSizeArray();

	// compute radiation timestep 'dtrad_tmp'
	amrex::Real domain_signal_max = RadSystem<problem_t>::c_hat_;
	amrex::Real dx_min = std::min({AMREX_D_DECL(dx[0], dx[1], dx[2])});
	amrex::Real dtrad_tmp = radiationCflNumber_ * (dx_min / domain_signal_max);

	amrex::Long nsubSteps = 0;
	amrex::Real dt_radiation = NAN;
	if (is_hydro_enabled_) {
		// adjust to get integer number of substeps
		amrex::Real dt_hydro = dt_lev_hydro;
		nsubSteps = std::ceil(dt_hydro / dtrad_tmp);
		dt_radiation = dt_hydro / static_cast<double>(nsubSteps);
	}
	// else { // no hydro (this is necessary for radiation test problems)
	//	nsubSteps = 1;
	//	dt_radiation = dtrad_tmp;
	//	dt_ = dt_radiation; // adjust global timestep (ok because no hydro was computed)
	//}
	AMREX_ALWAYS_ASSERT(nsubSteps >= 1);
	AMREX_ALWAYS_ASSERT(nsubSteps < 1e4);
	AMREX_ALWAYS_ASSERT(dt_radiation > 0.0);

	amrex::Print() << "\nRadiation substeps: " << nsubSteps << "\tdt: " << dt_radiation << "\n";

	// subcycle
	for (int i = 0; i < nsubSteps; ++i) {
		advanceSingleTimestepAtLevelRadiation(lev, time,
						      dt_radiation); // using dt_radiation
	}
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::advanceSingleTimestepAtLevelRadiation(int lev, amrex::Real time,
									  amrex::Real dt_radiation)
{
	// get geometry (used only for cell sizes)
	auto const &geomLevel = geom[lev];
	auto const &dx = geomLevel.CellSizeArray();

	// get flux registers
	amrex::YAFluxRegister &fr_as_crse = *flux_reg_[lev + 1];
	amrex::YAFluxRegister &fr_as_fine = *flux_reg_[lev];

	// We use the RK2-SSP method here. It needs two registers: one to store the old timestep,
	// and another to store the intermediate stage (which is reused for the final stage).

	// update ghost zones [old timestep]
	fillBoundaryConditions(state_old_[lev], state_old_[lev], lev, time);

	// advance all grids on local processor (Stage 1 of integrator)
	for (amrex::MFIter iter(state_new_[lev]); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox(); // 'validbox' == exclude ghost zones
		auto const &stateOld = state_old_[lev].const_array(iter);
		auto const &stateNew = state_new_[lev].array(iter);
		auto [fluxArrays, fluxDiffusiveArrays] =
		    computeRadiationFluxes(stateOld, indexRange, ncompHyperbolic_, dx);

		// Stage 1 of RK2-SSP
		RadSystem<problem_t>::PredictStep(
		    stateOld, stateNew,
		    {AMREX_D_DECL(fluxArrays[0].const_array(), fluxArrays[1].const_array(),
				  fluxArrays[2].const_array())},
		    {AMREX_D_DECL(fluxDiffusiveArrays[0].const_array(),
				  fluxDiffusiveArrays[1].const_array(),
				  fluxDiffusiveArrays[2].const_array())},
		    dt_radiation, dx, indexRange, ncompHyperbolic_);

		// increment flux registers
		incrementFluxRegisters(iter, fr_as_crse, fr_as_fine, fluxArrays, 0.5, lev,
				       dt_radiation);
	}

	// update ghost zones [intermediate stage stored in state_new_]
	fillBoundaryConditions(state_new_[lev], state_new_[lev], lev, time);

	// advance all grids on local processor (Stage 2 of integrator)
	for (amrex::MFIter iter(state_new_[lev]); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox(); // 'validbox' == exclude ghost zones
		auto const &stateOld = state_old_[lev].const_array(iter);
		auto const &stateInter = state_new_[lev].const_array(iter);
		auto const &stateNew = state_new_[lev].array(iter);
		auto [fluxArrays, fluxDiffusiveArrays] =
		    computeRadiationFluxes(stateInter, indexRange, ncompHyperbolic_, dx);

		// Stage 2 of RK2-SSP
		RadSystem<problem_t>::AddFluxesRK2(
		    stateNew, stateOld, stateInter,
		    {AMREX_D_DECL(fluxArrays[0].const_array(), fluxArrays[1].const_array(),
				  fluxArrays[2].const_array())},
		    {AMREX_D_DECL(fluxDiffusiveArrays[0].const_array(),
				  fluxDiffusiveArrays[1].const_array(),
				  fluxDiffusiveArrays[2].const_array())},
		    dt_radiation, dx, indexRange, ncompHyperbolic_);

		// increment flux registers
		incrementFluxRegisters(iter, fr_as_crse, fr_as_fine, fluxArrays, 0.5, lev,
				       dt_radiation);
	}

	// matter-radiation exchange source terms
	for (amrex::MFIter iter(state_new_[lev]); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox(); // 'validbox' == exclude ghost zones
		auto const &stateNew = state_new_[lev].array(iter);
		operatorSplitSourceTerms(stateNew, indexRange, ncomp_, time, dt_radiation, dx);
	}
}

template <typename problem_t>
void RadhydroSimulation<problem_t>::operatorSplitSourceTerms(
    amrex::Array4<amrex::Real> const &stateNew, const amrex::Box &indexRange, const int /*nvars*/,
    const amrex::Real time, const double dt, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx)
{
	amrex::FArrayBox radEnergySource(indexRange, 1,
					 amrex::The_Async_Arena()); // cell-centered scalar
	amrex::FArrayBox advectionFluxes(indexRange, 3,
					 amrex::The_Async_Arena()); // cell-centered vector

	radEnergySource.template setVal<amrex::RunOn::Device>(0.);
	advectionFluxes.template setVal<amrex::RunOn::Device>(0.);

	// cell-centered radiation energy source (used only in test problems)
	RadSystem<problem_t>::SetRadEnergySource(radEnergySource.array(), indexRange, dx,
						 time + dt);

	// cell-centered source terms
	RadSystem<problem_t>::AddSourceTerms(stateNew, radEnergySource.const_array(),
					     advectionFluxes.const_array(), indexRange, dt);
}

template <typename problem_t>
auto RadhydroSimulation<problem_t>::computeRadiationFluxes(
    amrex::Array4<const amrex::Real> const &consVar, const amrex::Box &indexRange, const int nvars,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx)
    -> std::tuple<std::array<amrex::FArrayBox, AMREX_SPACEDIM>,
		  std::array<amrex::FArrayBox, AMREX_SPACEDIM>>
{
	amrex::Box const &x1FluxRange = amrex::surroundingNodes(indexRange, 0);
	amrex::FArrayBox x1Flux(x1FluxRange, nvars, amrex::The_Async_Arena()); // node-centered in x
	amrex::FArrayBox x1FluxDiffusive(x1FluxRange, nvars, amrex::The_Async_Arena());
#if (AMREX_SPACEDIM >= 2)
	amrex::Box const &x2FluxRange = amrex::surroundingNodes(indexRange, 1);
	amrex::FArrayBox x2Flux(x2FluxRange, nvars, amrex::The_Async_Arena()); // node-centered in y
	amrex::FArrayBox x2FluxDiffusive(x2FluxRange, nvars, amrex::The_Async_Arena());
#endif
#if (AMREX_SPACEDIM == 3)
	amrex::Box const &x3FluxRange = amrex::surroundingNodes(indexRange, 2);
	amrex::FArrayBox x3Flux(x3FluxRange, nvars, amrex::The_Async_Arena()); // node-centered in z
	amrex::FArrayBox x3FluxDiffusive(x3FluxRange, nvars, amrex::The_Async_Arena());
#endif

	AMREX_D_TERM(
	    fluxFunction<FluxDir::X1>(consVar, x1Flux, x1FluxDiffusive, indexRange, nvars, dx);
	    , fluxFunction<FluxDir::X2>(consVar, x2Flux, x2FluxDiffusive, indexRange, nvars, dx);
	    , fluxFunction<FluxDir::X3>(consVar, x3Flux, x3FluxDiffusive, indexRange, nvars, dx);)

	std::array<amrex::FArrayBox, AMREX_SPACEDIM> fluxArrays = {
	    AMREX_D_DECL(std::move(x1Flux), std::move(x2Flux), std::move(x3Flux))};
	std::array<amrex::FArrayBox, AMREX_SPACEDIM> fluxDiffusiveArrays{AMREX_D_DECL(
	    std::move(x1FluxDiffusive), std::move(x2FluxDiffusive), std::move(x3FluxDiffusive))};

	return std::make_tuple(std::move(fluxArrays), std::move(fluxDiffusiveArrays));
}

template <typename problem_t>
template <FluxDir DIR>
void RadhydroSimulation<problem_t>::fluxFunction(amrex::Array4<const amrex::Real> const &consState,
						 amrex::FArrayBox &x1Flux,
						 amrex::FArrayBox &x1FluxDiffusive,
						 const amrex::Box &indexRange, const int nvars,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx)
{
	int dir = 0;
	if constexpr (DIR == FluxDir::X1) {
		dir = 0;
	} else if constexpr (DIR == FluxDir::X2) {
		dir = 1;
	} else if constexpr (DIR == FluxDir::X3) {
		dir = 2;
	}

	// extend box to include ghost zones
	amrex::Box const &ghostRange = amrex::grow(indexRange, nghost_);
	// N.B.: A one-zone layer around the cells must be fully reconstructed in order for PPM to
	// work.
	amrex::Box const &reconstructRange = amrex::grow(indexRange, 1);
	amrex::Box const &x1ReconstructRange = amrex::surroundingNodes(reconstructRange, dir);

	amrex::FArrayBox primVar(ghostRange, nvars,
				 amrex::The_Async_Arena()); // cell-centered
	amrex::FArrayBox x1LeftState(x1ReconstructRange, nvars, amrex::The_Async_Arena());
	amrex::FArrayBox x1RightState(x1ReconstructRange, nvars, amrex::The_Async_Arena());

	// cell-centered kernel
	RadSystem<problem_t>::ConservedToPrimitive(consState, primVar.array(), ghostRange);

	// mixed interface/cell-centered kernel
	RadSystem<problem_t>::template ReconstructStatesPPM<DIR>(
	    primVar.array(), x1LeftState.array(), x1RightState.array(), reconstructRange,
	    x1ReconstructRange, nvars);
	// PLM and donor cell are interface-centered kernels
	// RadSystem<problem_t>::template ReconstructStatesConstant<DIR>(
	//     primVar.array(), x1LeftState.array(), x1RightState.array(), x1ReconstructRange,
	//     nvars);
	// RadSystem<problem_t>::template ReconstructStatesPLM<DIR>(
	//     primVar.array(), x1LeftState.array(), x1RightState.array(), x1ReconstructRange,
	//     nvars);

	// interface-centered kernel
	amrex::Box const &x1FluxRange = amrex::surroundingNodes(indexRange, dir);
	RadSystem<problem_t>::template ComputeFluxes<DIR>(x1Flux.array(), x1FluxDiffusive.array(),
							  x1LeftState.array(), x1RightState.array(),
							  x1FluxRange, consState,
							  dx); // watch out for argument order!!
}

#endif // RADIATION_SIMULATION_HPP_