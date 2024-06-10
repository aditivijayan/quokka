#ifndef RADIATION_SYSTEM_HPP_ // NOLINT
#define RADIATION_SYSTEM_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file radiation_system.hpp
/// \brief Defines a class for solving the (1d) radiation moment equations.
///

// c++ headers

#include <array>
#include <cmath>

// library headers
#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_BLassert.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_IParser_Y.H"
#include "AMReX_REAL.H"

// internal headers
#include "EOS.hpp"
#include "hyperbolic_system.hpp"
#include "math_impl.hpp"
#include "physics_info.hpp"
#include "planck_integral.hpp"
#include "valarray.hpp"

// Hyper parameters of the radiation solver

static constexpr bool include_work_term_in_source = true;
static constexpr bool use_D_as_base = true;

// Time integration scheme
// IMEX PD-ARS
static constexpr double IMEX_a22 = 1.0;
static constexpr double IMEX_a32 = 0.5; // 0 < IMEX_a32 <= 0.5
// SSP-RK2 + implicit radiation-matter exchange
// static constexpr double IMEX_a22 = 0.0;
// static constexpr double IMEX_a32 = 0.0;

// physical constants in CGS units
static constexpr double c_light_cgs_ = C::c_light;	    // cgs
static constexpr double radiation_constant_cgs_ = C::a_rad; // cgs
static constexpr double inf = std::numeric_limits<double>::max();

// Optional: include a wavespeed correction term in the radiation flux to suppress instability
static const bool use_wavespeed_correction = false;

// enum for opacity_model
enum class OpacityModel {
	user = 0,	  // user-defined opacity for each group, given as a function of density and temperature.
	piecewisePowerLaw // piecewise power-law opacity model with piecewise power-law fitting to a user-defined opacity function and on-the-fly piecewise
			  // power-law fitting to radiation energy density and flux.
};

// this struct is specialized by the user application code
//
template <typename problem_t> struct RadSystem_Traits {
	static constexpr double c_light = c_light_cgs_;
	static constexpr double c_hat = c_light_cgs_;
	static constexpr double radiation_constant = radiation_constant_cgs_;
	static constexpr double Erad_floor = 0.;
	static constexpr double energy_unit = C::ev2erg;
	static constexpr amrex::GpuArray<double, Physics_Traits<problem_t>::nGroups + 1> radBoundaries = {0., inf};
	static constexpr double beta_order = 1;
	static constexpr OpacityModel opacity_model = OpacityModel::user;
};

// A struct to hold the results of the ComputeRadPressure function.
struct RadPressureResult {
	quokka::valarray<double, 4> F; // components of radiation pressure tensor
	double S;		       // maximum wavespeed for the radiation system
};

[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto minmod_func(double a, double b) -> double
{
	return 0.5 * (sgn(a) + sgn(b)) * std::min(std::abs(a), std::abs(b));
}

// Use SFINAE (Substitution Failure Is Not An Error) to check if opacity_model is defined in RadSystem_Traits<problem_t>
template <typename problem_t, typename = void> struct RadSystem_Has_Opacity_Model : std::false_type {
};

template <typename problem_t>
struct RadSystem_Has_Opacity_Model<problem_t, std::void_t<decltype(RadSystem_Traits<problem_t>::opacity_model)>> : std::true_type {
};

/// Class for the radiation moment equations
///
template <typename problem_t> class RadSystem : public HyperbolicSystem<problem_t>
{
      public:
	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto MC(double a, double b) -> double
	{
		return 0.5 * (sgn(a) + sgn(b)) * std::min(0.5 * std::abs(a + b), std::min(2.0 * std::abs(a), 2.0 * std::abs(b)));
	}

	static constexpr int nmscalars_ = Physics_Traits<problem_t>::numMassScalars;
	static constexpr int numRadVars_ = Physics_NumVars::numRadVars;				 // number of radiation variables for each photon group
	static constexpr int nvarHyperbolic_ = numRadVars_ * Physics_Traits<problem_t>::nGroups; // total number of radiation variables
	static constexpr int nstartHyperbolic_ = Physics_Indices<problem_t>::radFirstIndex;
	static constexpr int nvar_ = nstartHyperbolic_ + nvarHyperbolic_;

	enum gasVarIndex {
		gasDensity_index = Physics_Indices<problem_t>::hydroFirstIndex,
		x1GasMomentum_index,
		x2GasMomentum_index,
		x3GasMomentum_index,
		gasEnergy_index,
		gasInternalEnergy_index,
		scalar0_index
	};

	enum radVarIndex { radEnergy_index = nstartHyperbolic_, x1RadFlux_index, x2RadFlux_index, x3RadFlux_index };

	enum primVarIndex {
		primRadEnergy_index = 0,
		x1ReducedFlux_index,
		x2ReducedFlux_index,
		x3ReducedFlux_index,
	};

	// C++ standard does not allow constexpr to be uninitialized, even in a
	// templated class!
	static constexpr double c_light_ = RadSystem_Traits<problem_t>::c_light;
	static constexpr double c_hat_ = RadSystem_Traits<problem_t>::c_hat;
	static constexpr double radiation_constant_ = RadSystem_Traits<problem_t>::radiation_constant;

	static constexpr int beta_order_ = RadSystem_Traits<problem_t>::beta_order;

	static constexpr int nGroups_ = Physics_Traits<problem_t>::nGroups;
	static constexpr amrex::GpuArray<double, nGroups_ + 1> radBoundaries_ = []() constexpr {
		if constexpr (nGroups_ > 1) {
			return RadSystem_Traits<problem_t>::radBoundaries;
		} else {
			amrex::GpuArray<double, 2> boundaries{0., inf};
			return boundaries;
		}
	}();
	static constexpr double Erad_floor_ = RadSystem_Traits<problem_t>::Erad_floor / nGroups_;

	static constexpr OpacityModel opacity_model_ = []() constexpr {
		if constexpr (RadSystem_Has_Opacity_Model<problem_t>::value) {
			return RadSystem_Traits<problem_t>::opacity_model;
		} else {
			return OpacityModel::user;
		}
	}();

	static constexpr double mean_molecular_mass_ = quokka::EOS_Traits<problem_t>::mean_molecular_mass;
	static constexpr double boltzmann_constant_ = quokka::EOS_Traits<problem_t>::boltzmann_constant;
	static constexpr double gamma_ = quokka::EOS_Traits<problem_t>::gamma;

	// static functions

	static void ComputeMaxSignalSpeed(amrex::Array4<const amrex::Real> const &cons, array_t &maxSignal, amrex::Box const &indexRange);
	static void ConservedToPrimitive(amrex::Array4<const amrex::Real> const &cons, array_t &primVar, amrex::Box const &indexRange);

	static void PredictStep(arrayconst_t &consVarOld, array_t &consVarNew, amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArray,
				amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxDiffusiveArray, double dt_in,
				amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange, int nvars);

	static void AddFluxesRK2(array_t &U_new, arrayconst_t &U0, arrayconst_t &U1, amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArrayOld,
				 amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArray, amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxDiffusiveArrayOld,
				 amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxDiffusiveArray, double dt_in,
				 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange, int nvars);

	template <FluxDir DIR>
	static void ComputeFluxes(array_t &x1Flux_in, array_t &x1FluxDiffusive_in, amrex::Array4<const amrex::Real> const &x1LeftState_in,
				  amrex::Array4<const amrex::Real> const &x1RightState_in, amrex::Box const &indexRange, arrayconst_t &consVar_in,
				  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx);

	static void SetRadEnergySource(array_t &radEnergySource, amrex::Box const &indexRange, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
				       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi,
				       amrex::Real time);

	static void AddSourceTerms(array_t &consVar, arrayconst_t &radEnergySource, amrex::Box const &indexRange, amrex::Real dt, int stage);

	static void balanceMatterRadiation(arrayconst_t &consPrev, array_t &consNew, amrex::Box const &indexRange);

	static void ComputeSourceTermsExplicit(arrayconst_t &consPrev, arrayconst_t &radEnergySource, array_t &src, amrex::Box const &indexRange,
					       amrex::Real dt);

	// Use an additionalr template for ComputeMassScalars as the Array type is not always the same
	template <typename ArrayType>
	AMREX_GPU_DEVICE static auto ComputeMassScalars(ArrayType const &arr, int i, int j, int k) -> amrex::GpuArray<Real, nmscalars_>;

	AMREX_GPU_HOST_DEVICE static auto ComputeEddingtonFactor(double f) -> double;
	AMREX_GPU_HOST_DEVICE static auto ComputePlanckOpacity(double rho, double Tgas) -> quokka::valarray<double, nGroups_>;
	AMREX_GPU_HOST_DEVICE static auto ComputeFluxMeanOpacity(double rho, double Tgas) -> quokka::valarray<double, nGroups_>;
	AMREX_GPU_HOST_DEVICE static auto ComputeEnergyMeanOpacity(double rho, double Tgas) -> quokka::valarray<double, nGroups_>;
	AMREX_GPU_HOST_DEVICE static auto DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> rad_boundaries, double rho,
									       double Tgas) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_>, 2>;
	AMREX_GPU_HOST_DEVICE static auto ComputeGroupMeanOpacity(amrex::GpuArray<amrex::GpuArray<double, nGroups_>, 2> kappa_expo_and_lower_value,
								  amrex::GpuArray<double, nGroups_> radBoundaryRatios,
								  amrex::GpuArray<double, nGroups_> alpha_quant) -> quokka::valarray<double, nGroups_>;
	AMREX_GPU_HOST_DEVICE static auto ComputePlanckOpacityTempDerivative(double rho, double Tgas) -> quokka::valarray<double, nGroups_>;
	AMREX_GPU_HOST_DEVICE static auto ComputeEintFromEgas(double density, double X1GasMom, double X2GasMom, double X3GasMom, double Etot) -> double;
	AMREX_GPU_HOST_DEVICE static auto ComputeEgasFromEint(double density, double X1GasMom, double X2GasMom, double X3GasMom, double Eint) -> double;

	template <typename ArrayType>
	AMREX_GPU_HOST_DEVICE static auto
	ComputeRadQuantityExponents(ArrayType const &quant, amrex::GpuArray<double, nGroups_ + 1> const &boundaries) -> amrex::GpuArray<double, nGroups_>;

	AMREX_GPU_HOST_DEVICE static void SolveLinearEqs(double a00, const quokka::valarray<double, nGroups_> &a0i,
							 const quokka::valarray<double, nGroups_> &ai0, const quokka::valarray<double, nGroups_> &aii,
							 const double &y0, const quokka::valarray<double, nGroups_> &yi, double &x0,
							 quokka::valarray<double, nGroups_> &xi);

	AMREX_GPU_HOST_DEVICE static auto Solve3x3matrix(double C00, double C01, double C02, double C10, double C11, double C12, double C20, double C21,
							 double C22, double Y0, double Y1, double Y2) -> std::tuple<amrex::Real, amrex::Real, amrex::Real>;

	AMREX_GPU_HOST_DEVICE static auto ComputePlanckEnergyFractions(amrex::GpuArray<double, nGroups_ + 1> const &boundaries,
								       amrex::Real temperature) -> quokka::valarray<amrex::Real, nGroups_>;

	AMREX_GPU_HOST_DEVICE static auto
	ComputeThermalRadiation(amrex::Real temperature, amrex::GpuArray<double, nGroups_ + 1> const &boundaries) -> quokka::valarray<amrex::Real, nGroups_>;

	AMREX_GPU_HOST_DEVICE static auto
	ComputeThermalRadiationTempDerivative(amrex::Real temperature,
					      amrex::GpuArray<double, nGroups_ + 1> const &boundaries) -> quokka::valarray<amrex::Real, nGroups_>;

	template <FluxDir DIR>
	AMREX_GPU_DEVICE static auto ComputeCellOpticalDepth(const quokka::Array4View<const amrex::Real, DIR> &consVar,
							     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, int i, int j,
							     int k) -> quokka::valarray<double, nGroups_>;

	AMREX_GPU_DEVICE static auto isStateValid(std::array<amrex::Real, nvarHyperbolic_> &cons) -> bool;

	AMREX_GPU_DEVICE static void amendRadState(std::array<amrex::Real, nvarHyperbolic_> &cons);

	template <FluxDir DIR>
	AMREX_GPU_DEVICE static auto ComputeRadPressure(double erad_L, double Fx_L, double Fy_L, double Fz_L, double fx_L, double fy_L,
							double fz_L) -> RadPressureResult;

	AMREX_GPU_DEVICE static auto ComputeEddingtonTensor(double fx_L, double fy_L, double fz_L) -> std::array<std::array<double, 3>, 3>;
};

// Compute radiation energy fractions for each photon group from a Planck function, given nGroups, radBoundaries, and temperature
template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputePlanckEnergyFractions(amrex::GpuArray<double, nGroups_ + 1> const &boundaries,
									      amrex::Real temperature) -> quokka::valarray<amrex::Real, nGroups_>
{
	quokka::valarray<amrex::Real, nGroups_> radEnergyFractions{};
	if constexpr (nGroups_ == 1) {
		// TODO(CCH): allow the total radEnergyFraction to be smaller than 1. One usage case is to allow, say, a single group from 13.6 eV to 24.6 eV.
		radEnergyFractions[0] = 1.0;
		return radEnergyFractions;
	} else {
		amrex::Real const energy_unit_over_kT = RadSystem_Traits<problem_t>::energy_unit / (boltzmann_constant_ * temperature);
		amrex::Real previous = integrate_planck_from_0_to_x(boundaries[0] * energy_unit_over_kT);
		for (int g = 0; g < nGroups_; ++g) {
			amrex::Real y = integrate_planck_from_0_to_x(boundaries[g + 1] * energy_unit_over_kT);
			radEnergyFractions[g] = y - previous;
			previous = y;
		}
		auto tote = sum(radEnergyFractions);
		// AMREX_ALWAYS_ASSERT(tote <= 1.0);
		// AMREX_ALWAYS_ASSERT(tote > 0.9999);
		radEnergyFractions /= tote;
		return radEnergyFractions;
	}
}

// define ComputeThermalRadiation, returns the thermal radiation power for each photon group. = a_r * T^4 * radEnergyFractions
template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeThermalRadiation(amrex::Real temperature, amrex::GpuArray<double, nGroups_ + 1> const &boundaries)
    -> quokka::valarray<amrex::Real, nGroups_>
{
	auto radEnergyFractions = ComputePlanckEnergyFractions(boundaries, temperature);
	double power = radiation_constant_ * std::pow(temperature, 4);
	auto Erad_g = power * radEnergyFractions;
	// set floor
	for (int g = 0; g < nGroups_; ++g) {
		if (Erad_g[g] < Erad_floor_) {
			Erad_g[g] = Erad_floor_;
		}
	}
	return Erad_g;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto
RadSystem<problem_t>::ComputeThermalRadiationTempDerivative(amrex::Real temperature,
							    amrex::GpuArray<double, nGroups_ + 1> const &boundaries) -> quokka::valarray<amrex::Real, nGroups_>
{
	// by default, d emission/dT = 4 emission / T
	auto erad = ComputeThermalRadiation(temperature, boundaries);
	return 4. * erad / temperature;
}

// Linear equation solver for matrix with non-zeros at the first row, first column, and diagonal only.
// solve the linear system
//   [a00 a0i] [x0] = [y0]
//   [ai0 aii] [xi]   [yi]
// for x0 and xi, where a0i = (a01, a02, a03, ...); ai0 = (a10, a20, a30, ...); aii = (a11, a22, a33, ...), xi = (x1, x2, x3, ...), yi = (y1, y2, y3, ...)
template <typename problem_t>
AMREX_GPU_HOST_DEVICE void RadSystem<problem_t>::SolveLinearEqs(const double a00, const quokka::valarray<double, nGroups_> &a0i,
								const quokka::valarray<double, nGroups_> &ai0, const quokka::valarray<double, nGroups_> &aii,
								const double &y0, const quokka::valarray<double, nGroups_> &yi, double &x0,
								quokka::valarray<double, nGroups_> &xi)
{
	auto ratios = a0i / aii;
	x0 = (-sum(ratios * yi) + y0) / (-sum(ratios * ai0) + a00);
	xi = (yi - ai0 * x0) / aii;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::Solve3x3matrix(const double C00, const double C01, const double C02, const double C10, const double C11,
								const double C12, const double C20, const double C21, const double C22, const double Y0,
								const double Y1, const double Y2) -> std::tuple<amrex::Real, amrex::Real, amrex::Real>
{
	// Solve the 3x3 matrix equation: C * X = Y under the assumption that only the diagonal terms
	// are guaranteed to be non-zero and are thus allowed to be divided by.

	auto E11 = C11 - C01 * C10 / C00;
	auto E12 = C12 - C02 * C10 / C00;
	auto E21 = C21 - C01 * C20 / C00;
	auto E22 = C22 - C02 * C20 / C00;
	auto Z1 = Y1 - Y0 * C10 / C00;
	auto Z2 = Y2 - Y0 * C20 / C00;
	auto X2 = (Z2 - Z1 * E21 / E11) / (E22 - E12 * E21 / E11);
	auto X1 = (Z1 - E12 * X2) / E11;
	auto X0 = (Y0 - C01 * X1 - C02 * X2) / C00;

	return std::make_tuple(X0, X1, X2);
}

template <typename problem_t>
void RadSystem<problem_t>::SetRadEnergySource(array_t &radEnergySource, amrex::Box const &indexRange, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
					      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
					      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi, amrex::Real time)
{
	// do nothing -- user implemented
}

template <typename problem_t>
void RadSystem<problem_t>::ConservedToPrimitive(amrex::Array4<const amrex::Real> const &cons, array_t &primVar, amrex::Box const &indexRange)
{
	// keep radiation energy density as-is
	// convert (Fx,Fy,Fz) into reduced flux components (fx,fy,fx):
	//   F_x -> F_x / (c*E_r)

	// cell-centered kernel
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// add reduced fluxes for each radiation group
		for (int g = 0; g < nGroups_; ++g) {
			const auto E_r = cons(i, j, k, radEnergy_index + numRadVars_ * g);
			const auto Fx = cons(i, j, k, x1RadFlux_index + numRadVars_ * g);
			const auto Fy = cons(i, j, k, x2RadFlux_index + numRadVars_ * g);
			const auto Fz = cons(i, j, k, x3RadFlux_index + numRadVars_ * g);

			// check admissibility of states
			AMREX_ASSERT(E_r > 0.0); // NOLINT

			primVar(i, j, k, primRadEnergy_index + numRadVars_ * g) = E_r;
			primVar(i, j, k, x1ReducedFlux_index + numRadVars_ * g) = Fx / (c_light_ * E_r);
			primVar(i, j, k, x2ReducedFlux_index + numRadVars_ * g) = Fy / (c_light_ * E_r);
			primVar(i, j, k, x3ReducedFlux_index + numRadVars_ * g) = Fz / (c_light_ * E_r);
		}
	});
}

template <typename problem_t>
void RadSystem<problem_t>::ComputeMaxSignalSpeed(amrex::Array4<const amrex::Real> const & /*cons*/, array_t &maxSignal, amrex::Box const &indexRange)
{
	// cell-centered kernel
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double signal_max = c_hat_;
		maxSignal(i, j, k) = signal_max;
	});
}

template <typename problem_t> AMREX_GPU_DEVICE auto RadSystem<problem_t>::isStateValid(std::array<amrex::Real, nvarHyperbolic_> &cons) -> bool
{
	// check if the state variable 'cons' is a valid state
	bool isValid = true;
	for (int g = 0; g < nGroups_; ++g) {
		const auto E_r = cons[radEnergy_index + numRadVars_ * g - nstartHyperbolic_];
		const auto Fx = cons[x1RadFlux_index + numRadVars_ * g - nstartHyperbolic_];
		const auto Fy = cons[x2RadFlux_index + numRadVars_ * g - nstartHyperbolic_];
		const auto Fz = cons[x3RadFlux_index + numRadVars_ * g - nstartHyperbolic_];

		const auto Fnorm = std::sqrt(Fx * Fx + Fy * Fy + Fz * Fz);
		const auto f = Fnorm / (c_light_ * E_r);

		bool isNonNegative = (E_r > 0.);
		bool isFluxCausal = (f <= 1.);
		isValid = (isValid && isNonNegative && isFluxCausal);
	}
	return isValid;
}

template <typename problem_t> AMREX_GPU_DEVICE void RadSystem<problem_t>::amendRadState(std::array<amrex::Real, nvarHyperbolic_> &cons)
{
	// amend the state variable 'cons' to be a valid state
	for (int g = 0; g < nGroups_; ++g) {
		auto E_r = cons[radEnergy_index + numRadVars_ * g - nstartHyperbolic_];
		if (E_r < Erad_floor_) {
			E_r = Erad_floor_;
			cons[radEnergy_index + numRadVars_ * g - nstartHyperbolic_] = Erad_floor_;
		}
		const auto Fx = cons[x1RadFlux_index + numRadVars_ * g - nstartHyperbolic_];
		const auto Fy = cons[x2RadFlux_index + numRadVars_ * g - nstartHyperbolic_];
		const auto Fz = cons[x3RadFlux_index + numRadVars_ * g - nstartHyperbolic_];
		if (Fx * Fx + Fy * Fy + Fz * Fz > c_light_ * c_light_ * E_r * E_r) {
			const auto Fnorm = std::sqrt(Fx * Fx + Fy * Fy + Fz * Fz);
			cons[x1RadFlux_index + numRadVars_ * g - nstartHyperbolic_] = Fx / Fnorm * c_light_ * E_r;
			cons[x2RadFlux_index + numRadVars_ * g - nstartHyperbolic_] = Fy / Fnorm * c_light_ * E_r;
			cons[x3RadFlux_index + numRadVars_ * g - nstartHyperbolic_] = Fz / Fnorm * c_light_ * E_r;
		}
	}
}

template <typename problem_t>
void RadSystem<problem_t>::PredictStep(arrayconst_t &consVarOld, array_t &consVarNew, amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArray,
				       amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> /*fluxDiffusiveArray*/, const double dt_in,
				       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange, const int /*nvars*/)
{
	// By convention, the fluxes are defined on the left edge of each zone,
	// i.e. flux_(i) is the flux *into* zone i through the interface on the
	// left of zone i, and -1.0*flux(i+1) is the flux *into* zone i through
	// the interface on the right of zone i.

	auto const dt = dt_in;
	const auto dx = dx_in[0];
	const auto x1Flux = fluxArray[0];
	// const auto x1FluxDiffusive = fluxDiffusiveArray[0];
#if (AMREX_SPACEDIM >= 2)
	const auto dy = dx_in[1];
	const auto x2Flux = fluxArray[1];
	// const auto x2FluxDiffusive = fluxDiffusiveArray[1];
#endif
#if (AMREX_SPACEDIM == 3)
	const auto dz = dx_in[2];
	const auto x3Flux = fluxArray[2];
	// const auto x3FluxDiffusive = fluxDiffusiveArray[2];
#endif

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		std::array<amrex::Real, nvarHyperbolic_> cons{};

		for (int n = 0; n < nvarHyperbolic_; ++n) {
			cons[n] = consVarOld(i, j, k, nstartHyperbolic_ + n) + (AMREX_D_TERM((dt / dx) * (x1Flux(i, j, k, n) - x1Flux(i + 1, j, k, n)),
											     +(dt / dy) * (x2Flux(i, j, k, n) - x2Flux(i, j + 1, k, n)),
											     +(dt / dz) * (x3Flux(i, j, k, n) - x3Flux(i, j, k + 1, n))));
		}

		if (!isStateValid(cons)) {
			amendRadState(cons);
		}
		AMREX_ASSERT(isStateValid(cons));

		for (int n = 0; n < nvarHyperbolic_; ++n) {
			consVarNew(i, j, k, nstartHyperbolic_ + n) = cons[n];
		}
	});
}

template <typename problem_t>
void RadSystem<problem_t>::AddFluxesRK2(array_t &U_new, arrayconst_t &U0, arrayconst_t &U1, amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArrayOld,
					amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArray,
					amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> /*fluxDiffusiveArrayOld*/,
					amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> /*fluxDiffusiveArray*/, const double dt_in,
					amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange, const int /*nvars*/)
{
	// By convention, the fluxes are defined on the left edge of each zone,
	// i.e. flux_(i) is the flux *into* zone i through the interface on the
	// left of zone i, and -1.0*flux(i+1) is the flux *into* zone i through
	// the interface on the right of zone i.

	auto const dt = dt_in;
	const auto dx = dx_in[0];
	const auto x1FluxOld = fluxArrayOld[0];
	const auto x1Flux = fluxArray[0];
#if (AMREX_SPACEDIM >= 2)
	const auto dy = dx_in[1];
	const auto x2FluxOld = fluxArrayOld[1];
	const auto x2Flux = fluxArray[1];
#endif
#if (AMREX_SPACEDIM == 3)
	const auto dz = dx_in[2];
	const auto x3FluxOld = fluxArrayOld[2];
	const auto x3Flux = fluxArray[2];
#endif

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		std::array<amrex::Real, nvarHyperbolic_> cons_new{};

		// y^n+1 = (1 - a32) y^n + a32 y^(2) + dt * (0.5 - a32) * s(y^n) + dt * 0.5 * s(y^(2)) + dt * (1 - a32) * f(y^n+1)          // the last term is
		// implicit and not used here
		for (int n = 0; n < nvarHyperbolic_; ++n) {
			const double U_0 = U0(i, j, k, nstartHyperbolic_ + n);
			const double U_1 = U1(i, j, k, nstartHyperbolic_ + n);
			const double FxU_0 = (dt / dx) * (x1FluxOld(i, j, k, n) - x1FluxOld(i + 1, j, k, n));
			const double FxU_1 = (dt / dx) * (x1Flux(i, j, k, n) - x1Flux(i + 1, j, k, n));
#if (AMREX_SPACEDIM >= 2)
			const double FyU_0 = (dt / dy) * (x2FluxOld(i, j, k, n) - x2FluxOld(i, j + 1, k, n));
			const double FyU_1 = (dt / dy) * (x2Flux(i, j, k, n) - x2Flux(i, j + 1, k, n));
#endif
#if (AMREX_SPACEDIM == 3)
			const double FzU_0 = (dt / dz) * (x3FluxOld(i, j, k, n) - x3FluxOld(i, j, k + 1, n));
			const double FzU_1 = (dt / dz) * (x3Flux(i, j, k, n) - x3Flux(i, j, k + 1, n));
#endif
			// save results in cons_new
			cons_new[n] = (1.0 - IMEX_a32) * U_0 + IMEX_a32 * U_1 + ((0.5 - IMEX_a32) * (AMREX_D_TERM(FxU_0, +FyU_0, +FzU_0))) +
				      (0.5 * (AMREX_D_TERM(FxU_1, +FyU_1, +FzU_1)));
		}

		if (!isStateValid(cons_new)) {
			amendRadState(cons_new);
		}
		AMREX_ASSERT(isStateValid(cons_new));

		for (int n = 0; n < nvarHyperbolic_; ++n) {
			U_new(i, j, k, nstartHyperbolic_ + n) = cons_new[n];
		}
	});
}

template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeEddingtonFactor(double f_in) -> double
{
	// f is the reduced flux == |F|/cE.
	// compute Levermore (1984) closure [Eq. 25]
	// the is the M1 closure that is derived from Lorentz invariance
	const double f = clamp(f_in, 0., 1.); // restrict f to be within [0, 1]
	const double f_fac = std::sqrt(4.0 - 3.0 * (f * f));
	const double chi = (3.0 + 4.0 * (f * f)) / (5.0 + 2.0 * f_fac);

#if 0 // NOLINT
      // compute Minerbo (1978) closure [piecewise approximation]
      // (For unknown reasons, this closure tends to work better
      // than the Levermore/Lorentz closure on the Su & Olson 1997 test.)
	const double chi = (f < 1. / 3.) ? (1. / 3.) : (0.5 - f + 1.5 * f*f);
#endif

	return chi;
}

template <typename problem_t>
template <typename ArrayType>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeMassScalars(ArrayType const &arr, int i, int j, int k) -> amrex::GpuArray<Real, nmscalars_>
{
	amrex::GpuArray<Real, nmscalars_> massScalars;
	for (int n = 0; n < nmscalars_; ++n) {
		massScalars[n] = arr(i, j, k, scalar0_index + n);
	}
	return massScalars;
}

template <typename problem_t>
template <FluxDir DIR>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeCellOpticalDepth(const quokka::Array4View<const amrex::Real, DIR> &consVar,
								    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, int i, int j,
								    int k) -> quokka::valarray<double, nGroups_>
{
	// compute interface-averaged cell optical depth

	// [By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xleft_(i) is the "left"-side of the interface at
	// the left edge of zone i, and xright_(i) is the "right"-side of the
	// interface at the *left* edge of zone i.]

	// piecewise-constant reconstruction
	const double rho_L = consVar(i - 1, j, k, gasDensity_index);
	const double rho_R = consVar(i, j, k, gasDensity_index);

	const double x1GasMom_L = consVar(i - 1, j, k, x1GasMomentum_index);
	const double x1GasMom_R = consVar(i, j, k, x1GasMomentum_index);

	const double x2GasMom_L = consVar(i - 1, j, k, x2GasMomentum_index);
	const double x2GasMom_R = consVar(i, j, k, x2GasMomentum_index);

	const double x3GasMom_L = consVar(i - 1, j, k, x3GasMomentum_index);
	const double x3GasMom_R = consVar(i, j, k, x3GasMomentum_index);

	const double Egas_L = consVar(i - 1, j, k, gasEnergy_index);
	const double Egas_R = consVar(i, j, k, gasEnergy_index);

	auto massScalars_L = RadSystem<problem_t>::ComputeMassScalars(consVar, i - 1, j, k);
	auto massScalars_R = RadSystem<problem_t>::ComputeMassScalars(consVar, i, j, k);

	double Eint_L = NAN;
	double Eint_R = NAN;
	double Tgas_L = NAN;
	double Tgas_R = NAN;

	if constexpr (gamma_ != 1.0) {
		Eint_L = RadSystem<problem_t>::ComputeEintFromEgas(rho_L, x1GasMom_L, x2GasMom_L, x3GasMom_L, Egas_L);
		Eint_R = RadSystem<problem_t>::ComputeEintFromEgas(rho_R, x1GasMom_R, x2GasMom_R, x3GasMom_R, Egas_R);
		Tgas_L = quokka::EOS<problem_t>::ComputeTgasFromEint(rho_L, Eint_L, massScalars_L);
		Tgas_R = quokka::EOS<problem_t>::ComputeTgasFromEint(rho_R, Eint_R, massScalars_R);
	}

	double dl = NAN;
	if constexpr (DIR == FluxDir::X1) {
		dl = dx[0];
	} else if constexpr (DIR == FluxDir::X2) {
		dl = dx[1];
	} else if constexpr (DIR == FluxDir::X3) {
		dl = dx[2];
	}
	quokka::valarray<double, nGroups_> const tau_L = dl * rho_L * RadSystem<problem_t>::ComputeFluxMeanOpacity(rho_L, Tgas_L);
	quokka::valarray<double, nGroups_> const tau_R = dl * rho_R * RadSystem<problem_t>::ComputeFluxMeanOpacity(rho_R, Tgas_R);

	return (tau_L * tau_R * 2.) / (tau_L + tau_R); // harmonic mean
						       // return 0.5*(tau_L + tau_R); // arithmetic mean
}

template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeEddingtonTensor(const double fx, const double fy, const double fz) -> std::array<std::array<double, 3>, 3>
{
	// Compute the radiation pressure tensor

	// AMREX_ASSERT(f < 1.0); // there is sometimes a small (<1%) flux
	// limiting violation when using P1 AMREX_ASSERT(f_R < 1.0);

	auto f = std::sqrt(fx * fx + fy * fy + fz * fz);
	std::array<amrex::Real, 3> fvec = {fx, fy, fz};

	// angle between interface and radiation flux \hat{n}
	// If direction is undefined, just drop direction-dependent
	// terms.
	std::array<amrex::Real, 3> n{};

	for (int ii = 0; ii < 3; ++ii) {
		n[ii] = (f > 0.) ? (fvec[ii] / f) : 0.;
	}

	// compute radiation pressure tensors
	const double chi = RadSystem<problem_t>::ComputeEddingtonFactor(f);

	AMREX_ASSERT((chi >= 1. / 3.) && (chi <= 1.0)); // NOLINT

	// diagonal term of Eddington tensor
	const double Tdiag = (1.0 - chi) / 2.0;

	// anisotropic term of Eddington tensor (in the direction of the
	// rad. flux)
	const double Tf = (3.0 * chi - 1.0) / 2.0;

	// assemble Eddington tensor
	std::array<std::array<double, 3>, 3> T{};

	for (int ii = 0; ii < 3; ++ii) {
		for (int jj = 0; jj < 3; ++jj) {
			const double delta_ij = (ii == jj) ? 1 : 0;
			T[ii][jj] = Tdiag * delta_ij + Tf * (n[ii] * n[jj]);
		}
	}

	return T;
}

template <typename problem_t>
template <FluxDir DIR>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeRadPressure(const double erad, const double Fx, const double Fy, const double Fz, const double fx,
							       const double fy, const double fz) -> RadPressureResult
{
	// Compute the radiation pressure tensor and the maximum signal speed and return them as a struct.

	// check that states are physically admissible
	AMREX_ASSERT(erad > 0.0);

	// Compute the Eddington tensor
	auto T = ComputeEddingtonTensor(fx, fy, fz);

	// frozen Eddington tensor approximation, following Balsara
	// (1999) [JQSRT Vol. 61, No. 5, pp. 617–627, 1999], Eq. 46.
	double Tnormal = NAN;
	if constexpr (DIR == FluxDir::X1) {
		Tnormal = T[0][0];
	} else if constexpr (DIR == FluxDir::X2) {
		Tnormal = T[1][1];
	} else if constexpr (DIR == FluxDir::X3) {
		Tnormal = T[2][2];
	}

	// compute fluxes F_L, F_R
	// T_nx, T_ny, T_nz indicate components where 'n' is the direction of the
	// face normal. F_n is the radiation flux component in the direction of the
	// face normal
	double Fn = NAN;
	double Tnx = NAN;
	double Tny = NAN;
	double Tnz = NAN;

	if constexpr (DIR == FluxDir::X1) {
		Fn = Fx;

		Tnx = T[0][0];
		Tny = T[0][1];
		Tnz = T[0][2];
	} else if constexpr (DIR == FluxDir::X2) {
		Fn = Fy;

		Tnx = T[1][0];
		Tny = T[1][1];
		Tnz = T[1][2];
	} else if constexpr (DIR == FluxDir::X3) {
		Fn = Fz;

		Tnx = T[2][0];
		Tny = T[2][1];
		Tnz = T[2][2];
	}

	AMREX_ASSERT(Fn != NAN);
	AMREX_ASSERT(Tnx != NAN);
	AMREX_ASSERT(Tny != NAN);
	AMREX_ASSERT(Tnz != NAN);

	RadPressureResult result{};
	result.F = {Fn, Tnx * erad, Tny * erad, Tnz * erad};
	// It might be possible to remove this 0.1 floor without affecting the code. I tried and only the 3D RadForce failed (causing S_L = S_R = 0.0 and F[0] =
	// NAN). Read more on https://github.com/quokka-astro/quokka/pull/582 .
	result.S = std::max(0.1, std::sqrt(Tnormal));

	return result;
}

template <typename problem_t>
template <FluxDir DIR>
void RadSystem<problem_t>::ComputeFluxes(array_t &x1Flux_in, array_t &x1FluxDiffusive_in, amrex::Array4<const amrex::Real> const &x1LeftState_in,
					 amrex::Array4<const amrex::Real> const &x1RightState_in, amrex::Box const &indexRange, arrayconst_t &consVar_in,
					 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx)
{
	quokka::Array4View<const amrex::Real, DIR> x1LeftState(x1LeftState_in);
	quokka::Array4View<const amrex::Real, DIR> x1RightState(x1RightState_in);
	quokka::Array4View<amrex::Real, DIR> x1Flux(x1Flux_in);
	quokka::Array4View<amrex::Real, DIR> x1FluxDiffusive(x1FluxDiffusive_in);
	quokka::Array4View<const amrex::Real, DIR> consVar(consVar_in);

	// By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xinterface_(i) is the solution to the Riemann problem at
	// the left edge of zone i.

	// Indexing note: There are (nx + 1) interfaces for nx zones.

	// interface-centered kernel
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i_in, int j_in, int k_in) {
		auto [i, j, k] = quokka::reorderMultiIndex<DIR>(i_in, j_in, k_in);

		// HLL solver following Toro (1998) and Balsara (2017).
		// Radiation eigenvalues from Skinner & Ostriker (2013).

		// calculate cell optical depth for each photon group
		// Similar to the asymptotic-preserving flux correction in Skinner et al. (2019). Use optionally apply it here to reduce odd-even instability.
		quokka::valarray<double, nGroups_> tau_cell{};
		if (use_wavespeed_correction) {
			tau_cell = ComputeCellOpticalDepth<DIR>(consVar, dx, i, j, k);
		}

		// gather left- and right- state variables
		for (int g = 0; g < nGroups_; ++g) {
			double erad_L = x1LeftState(i, j, k, primRadEnergy_index + numRadVars_ * g);
			double erad_R = x1RightState(i, j, k, primRadEnergy_index + numRadVars_ * g);

			double fx_L = x1LeftState(i, j, k, x1ReducedFlux_index + numRadVars_ * g);
			double fx_R = x1RightState(i, j, k, x1ReducedFlux_index + numRadVars_ * g);

			double fy_L = x1LeftState(i, j, k, x2ReducedFlux_index + numRadVars_ * g);
			double fy_R = x1RightState(i, j, k, x2ReducedFlux_index + numRadVars_ * g);

			double fz_L = x1LeftState(i, j, k, x3ReducedFlux_index + numRadVars_ * g);
			double fz_R = x1RightState(i, j, k, x3ReducedFlux_index + numRadVars_ * g);

			// compute scalar reduced flux f
			double f_L = std::sqrt(fx_L * fx_L + fy_L * fy_L + fz_L * fz_L);
			double f_R = std::sqrt(fx_R * fx_R + fy_R * fy_R + fz_R * fz_R);

			// Compute "un-reduced" Fx, Fy, Fz
			double Fx_L = fx_L * (c_light_ * erad_L);
			double Fx_R = fx_R * (c_light_ * erad_R);

			double Fy_L = fy_L * (c_light_ * erad_L);
			double Fy_R = fy_R * (c_light_ * erad_R);

			double Fz_L = fz_L * (c_light_ * erad_L);
			double Fz_R = fz_R * (c_light_ * erad_R);

			// check that states are physically admissible; if not, use first-order
			// reconstruction
			if ((erad_L <= 0.) || (erad_R <= 0.) || (f_L >= 1.) || (f_R >= 1.)) {
				erad_L = consVar(i - 1, j, k, radEnergy_index + numRadVars_ * g);
				erad_R = consVar(i, j, k, radEnergy_index + numRadVars_ * g);

				Fx_L = consVar(i - 1, j, k, x1RadFlux_index + numRadVars_ * g);
				Fx_R = consVar(i, j, k, x1RadFlux_index + numRadVars_ * g);

				Fy_L = consVar(i - 1, j, k, x2RadFlux_index + numRadVars_ * g);
				Fy_R = consVar(i, j, k, x2RadFlux_index + numRadVars_ * g);

				Fz_L = consVar(i - 1, j, k, x3RadFlux_index + numRadVars_ * g);
				Fz_R = consVar(i, j, k, x3RadFlux_index + numRadVars_ * g);

				// compute primitive variables
				fx_L = Fx_L / (c_light_ * erad_L);
				fx_R = Fx_R / (c_light_ * erad_R);

				fy_L = Fy_L / (c_light_ * erad_L);
				fy_R = Fy_R / (c_light_ * erad_R);

				fz_L = Fz_L / (c_light_ * erad_L);
				fz_R = Fz_R / (c_light_ * erad_R);

				f_L = std::sqrt(fx_L * fx_L + fy_L * fy_L + fz_L * fz_L);
				f_R = std::sqrt(fx_R * fx_R + fy_R * fy_R + fz_R * fz_R);
			}

			// ComputeRadPressure returns F_L_and_S_L or F_R_and_S_R
			auto [F_L, S_L] = ComputeRadPressure<DIR>(erad_L, Fx_L, Fy_L, Fz_L, fx_L, fy_L, fz_L);
			S_L *= -1.; // speed sign is -1
			auto [F_R, S_R] = ComputeRadPressure<DIR>(erad_R, Fx_R, Fy_R, Fz_R, fx_R, fy_R, fz_R);

			// correct for reduced speed of light
			F_L[0] *= c_hat_ / c_light_;
			F_R[0] *= c_hat_ / c_light_;
			for (int n = 1; n < numRadVars_; ++n) {
				F_L[n] *= c_hat_ * c_light_;
				F_R[n] *= c_hat_ * c_light_;
			}
			S_L *= c_hat_;
			S_R *= c_hat_;

			const quokka::valarray<double, numRadVars_> U_L = {erad_L, Fx_L, Fy_L, Fz_L};
			const quokka::valarray<double, numRadVars_> U_R = {erad_R, Fx_R, Fy_R, Fz_R};

			// Adjusting wavespeeds is no longer necessary with the IMEX PD-ARS scheme.
			// Read more in https://github.com/quokka-astro/quokka/pull/582
			// However, we let the user optionally apply it to reduce odd-even instability.
			quokka::valarray<double, numRadVars_> epsilon = {1.0, 1.0, 1.0, 1.0};
			if (use_wavespeed_correction) {
				// no correction for odd zones
				if ((i + j + k) % 2 == 0) {
					const double S_corr = std::min(1.0, 1.0 / tau_cell[g]); // Skinner et al.
					epsilon = {S_corr, 1.0, 1.0, 1.0};			// Skinner et al. (2019)
				}
			}

			AMREX_ASSERT(std::abs(S_L) <= c_hat_); // NOLINT
			AMREX_ASSERT(std::abs(S_R) <= c_hat_); // NOLINT

			// in the frozen Eddington tensor approximation, we are always
			// in the star region, so F = F_star
			const quokka::valarray<double, numRadVars_> F =
			    (S_R / (S_R - S_L)) * F_L - (S_L / (S_R - S_L)) * F_R + epsilon * (S_R * S_L / (S_R - S_L)) * (U_R - U_L);

			// check states are valid
			AMREX_ASSERT(!std::isnan(F[0])); // NOLINT
			AMREX_ASSERT(!std::isnan(F[1])); // NOLINT
			AMREX_ASSERT(!std::isnan(F[2])); // NOLINT
			AMREX_ASSERT(!std::isnan(F[3])); // NOLINT

			x1Flux(i, j, k, radEnergy_index + numRadVars_ * g - nstartHyperbolic_) = F[0];
			x1Flux(i, j, k, x1RadFlux_index + numRadVars_ * g - nstartHyperbolic_) = F[1];
			x1Flux(i, j, k, x2RadFlux_index + numRadVars_ * g - nstartHyperbolic_) = F[2];
			x1Flux(i, j, k, x3RadFlux_index + numRadVars_ * g - nstartHyperbolic_) = F[3];

			const quokka::valarray<double, numRadVars_> diffusiveF =
			    (S_R / (S_R - S_L)) * F_L - (S_L / (S_R - S_L)) * F_R + (S_R * S_L / (S_R - S_L)) * (U_R - U_L);

			x1FluxDiffusive(i, j, k, radEnergy_index + numRadVars_ * g - nstartHyperbolic_) = diffusiveF[0];
			x1FluxDiffusive(i, j, k, x1RadFlux_index + numRadVars_ * g - nstartHyperbolic_) = diffusiveF[1];
			x1FluxDiffusive(i, j, k, x2RadFlux_index + numRadVars_ * g - nstartHyperbolic_) = diffusiveF[2];
			x1FluxDiffusive(i, j, k, x3RadFlux_index + numRadVars_ * g - nstartHyperbolic_) = diffusiveF[3];
		} // end loop over radiation groups
	});
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> quokka::valarray<double, nGroups_>
{
	quokka::valarray<double, nGroups_> kappaPVec{};
	for (int g = 0; g < nGroups_; ++g) {
		kappaPVec[g] = NAN;
	}
	return kappaPVec;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> quokka::valarray<double, nGroups_>
{
	quokka::valarray<double, nGroups_> kappaFVec{};
	for (int g = 0; g < nGroups_; ++g) {
		kappaFVec[g] = NAN;
	}
	return kappaFVec;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeEnergyMeanOpacity(const double rho, const double Tgas) -> quokka::valarray<double, nGroups_>
{
	return ComputePlanckOpacity(rho, Tgas);
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto
RadSystem<problem_t>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double /*rho*/,
							   const double /*Tgas*/) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_>, 2>
{
	amrex::GpuArray<amrex::GpuArray<double, nGroups_>, 2> exponents_and_values{};
	return exponents_and_values;
}

template <typename problem_t>
template <typename ArrayType>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeRadQuantityExponents(ArrayType const &quant, amrex::GpuArray<double, nGroups_ + 1> const &boundaries)
    -> amrex::GpuArray<double, nGroups_>
{
	// Compute the exponents for the radiation energy density, radiation flux, radiation pressure, or Planck function.

	// Note: Could save some memory by using bin_center_previous and bin_center_current
	amrex::GpuArray<double, nGroups_> bin_center{};
	amrex::GpuArray<double, nGroups_> quant_mean{};
	amrex::GpuArray<double, nGroups_ - 1> logslopes{};
	amrex::GpuArray<double, nGroups_> exponents{};
	for (int g = 0; g < nGroups_; ++g) {
		bin_center[g] = std::sqrt(boundaries[g] * boundaries[g + 1]);
		quant_mean[g] = quant[g] / (boundaries[g + 1] - boundaries[g]);
		if (g > 0) {
			AMREX_ASSERT(bin_center[g] > bin_center[g - 1]);
			if (quant_mean[g] == 0.0 && quant_mean[g - 1] == 0.0) {
				logslopes[g - 1] = 0.0;
			} else if (quant_mean[g - 1] * quant_mean[g] <= 0.0) {
				if (quant_mean[g] > quant_mean[g - 1]) {
					logslopes[g - 1] = inf;
				} else {
					logslopes[g - 1] = -inf;
				}
			} else {
				logslopes[g - 1] = std::log(std::abs(quant_mean[g] / quant_mean[g - 1])) / std::log(bin_center[g] / bin_center[g - 1]);
			}
			AMREX_ASSERT(!std::isnan(logslopes[g - 1]));
		}
	}
	for (int g = 0; g < nGroups_; ++g) {
		if (g == 0 || g == nGroups_ - 1) {
			exponents[g] = 0.0;
		} else {
			exponents[g] = minmod_func(logslopes[g - 1], logslopes[g]);
		}
		AMREX_ASSERT(!std::isnan(exponents[g]));
		AMREX_ASSERT(std::abs(exponents[g]) < 100);
	}

	return exponents;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeGroupMeanOpacity(amrex::GpuArray<amrex::GpuArray<double, nGroups_>, 2> kappa_expo_and_lower_value,
									 amrex::GpuArray<double, nGroups_> radBoundaryRatios,
									 amrex::GpuArray<double, nGroups_> alpha_quant) -> quokka::valarray<double, nGroups_>
{
	amrex::GpuArray<double, nGroups_> const &alpha_kappa = kappa_expo_and_lower_value[0];
	amrex::GpuArray<double, nGroups_> const &kappa_lower = kappa_expo_and_lower_value[1];

	quokka::valarray<double, nGroups_> kappa{};
	for (int g = 0; g < nGroups_; ++g) {
		double alpha = alpha_quant[g] + 1.0;
		double part1 = 0.0;
		if (std::abs(alpha) < 1e-8) {
			part1 = std::log(radBoundaryRatios[g]);
		} else {
			part1 = (std::pow(radBoundaryRatios[g], alpha) - 1.0) / alpha;
		}
		alpha = alpha_quant[g] + alpha_kappa[g] + 1.0;
		double part2 = 0.0;
		if (std::abs(alpha) < 1e-8) {
			part2 = std::log(radBoundaryRatios[g]);
		} else {
			part2 = (std::pow(radBoundaryRatios[g], alpha) - 1.0) / alpha;
		}
		kappa[g] = kappa_lower[g] / part1 * part2;
		AMREX_ASSERT(!std::isnan(kappa[g]));
	}
	return kappa;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputePlanckOpacityTempDerivative(const double /* rho */,
										    const double /* Tgas */) -> quokka::valarray<double, nGroups_>
{
	quokka::valarray<double, nGroups_> kappa{};
	kappa.fillin(0.0);
	return kappa;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeEintFromEgas(const double density, const double X1GasMom, const double X2GasMom, const double X3GasMom,
								     const double Etot) -> double
{
	const double p_sq = X1GasMom * X1GasMom + X2GasMom * X2GasMom + X3GasMom * X3GasMom;
	const double Ekin = p_sq / (2.0 * density);
	const double Eint = Etot - Ekin;
	AMREX_ASSERT_WITH_MESSAGE(Eint > 0., "Gas internal energy is not positive!");
	return Eint;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeEgasFromEint(const double density, const double X1GasMom, const double X2GasMom, const double X3GasMom,
								     const double Eint) -> double
{
	const double p_sq = X1GasMom * X1GasMom + X2GasMom * X2GasMom + X3GasMom * X3GasMom;
	const double Ekin = p_sq / (2.0 * density);
	const double Etot = Eint + Ekin;
	return Etot;
}

template <typename problem_t>
void RadSystem<problem_t>::AddSourceTerms(array_t &consVar, arrayconst_t &radEnergySource, amrex::Box const &indexRange, amrex::Real dt_radiation,
					  const int stage)
{
	arrayconst_t &consPrev = consVar; // make read-only
	array_t &consNew = consVar;
	auto dt = dt_radiation;
	if (stage == 2) {
		dt = (1.0 - IMEX_a32) * dt_radiation;
	}

	amrex::GpuArray<amrex::Real, nGroups_ + 1> radBoundaries_g = radBoundaries_;
	amrex::GpuArray<amrex::Real, nGroups_> radBoundaryRatios{};
	if constexpr (nGroups_ > 1) {
		if constexpr (opacity_model_ == OpacityModel::piecewisePowerLaw) {
			for (int g = 0; g < nGroups_; ++g) {
				radBoundaryRatios[g] = radBoundaries_g[g + 1] / radBoundaries_g[g];
			}
		}
	}

	// Add source terms

	// 1. Compute gas energy and radiation energy update following Howell &
	// Greenough [Journal of Computational Physics 184 (2003) 53–78].

	// cell-centered kernel
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double c = c_light_;
		const double chat = c_hat_;

		// load fluid properties
		const double rho = consPrev(i, j, k, gasDensity_index);
		const double x1GasMom0 = consPrev(i, j, k, x1GasMomentum_index);
		const double x2GasMom0 = consPrev(i, j, k, x2GasMomentum_index);
		const double x3GasMom0 = consPrev(i, j, k, x3GasMomentum_index);
		const std::array<double, 3> gasMtm0 = {x1GasMom0, x2GasMom0, x3GasMom0};
		const double Egastot0 = consPrev(i, j, k, gasEnergy_index);
		auto massScalars = RadSystem<problem_t>::ComputeMassScalars(consPrev, i, j, k);

		// load radiation energy
		quokka::valarray<double, nGroups_> Erad0Vec;
		for (int g = 0; g < nGroups_; ++g) {
			Erad0Vec[g] = consPrev(i, j, k, radEnergy_index + numRadVars_ * g);
		}
		AMREX_ASSERT(min(Erad0Vec) > 0.0);
		const double Erad0 = sum(Erad0Vec);

		// load radiation energy source term
		// plus advection source term (for well-balanced/SDC integrators)
		quokka::valarray<double, nGroups_> Src;
		for (int g = 0; g < nGroups_; ++g) {
			Src[g] = dt * (chat * radEnergySource(i, j, k, g));
		}

		double Egas0 = NAN;
		double Ekin0 = NAN;
		double Etot0 = NAN;
		double Egas_guess = NAN;
		double T_gas = NAN;
		double lorentz_factor = NAN;
		double lorentz_factor_v = NAN;
		double lorentz_factor_v_v = NAN;
		quokka::valarray<double, nGroups_> fourPiBoverC{};
		quokka::valarray<double, nGroups_> EradVec_guess{};
		quokka::valarray<double, nGroups_> kappaPVec{};
		quokka::valarray<double, nGroups_> kappaEVec{};
		quokka::valarray<double, nGroups_> kappaFVec{};
		amrex::GpuArray<amrex::GpuArray<double, nGroups_>, 2> kappa_expo_and_lower_value{};
		amrex::GpuArray<double, nGroups_> alpha_B{};
		amrex::GpuArray<double, nGroups_> alpha_E{};
		amrex::GpuArray<double, nGroups_> alpha_F{};
		quokka::valarray<double, nGroups_> kappaPoverE{};
		quokka::valarray<double, nGroups_> tau0{}; // optical depth across c * dt at old state
		quokka::valarray<double, nGroups_> tau{};  // optical depth across c * dt at new state
		quokka::valarray<double, nGroups_> D{};	   // D = S / tau0
		quokka::valarray<double, nGroups_> work{};
		quokka::valarray<double, nGroups_> work_prev{};
		amrex::GpuArray<amrex::GpuArray<amrex::Real, nGroups_>, 3> frad{};
		amrex::GpuArray<amrex::Real, 3> dMomentum{};
		amrex::GpuArray<amrex::GpuArray<amrex::Real, nGroups_>, 3> Frad_t1{};

		work.fillin(0.0);
		work_prev.fillin(0.0);

		EradVec_guess = Erad0Vec;

		if constexpr (gamma_ != 1.0) {
			Egas0 = ComputeEintFromEgas(rho, x1GasMom0, x2GasMom0, x3GasMom0, Egastot0);
			Etot0 = Egas0 + (c / chat) * (Erad0 + sum(Src));
		}

		// make a copy of radBoundaries_g
		amrex::GpuArray<double, nGroups_ + 1> radBoundaries_g_copy{};
		amrex::GpuArray<double, nGroups_> radBoundaryRatios_copy{};
		for (int g = 0; g < nGroups_ + 1; ++g) {
			radBoundaries_g_copy[g] = radBoundaries_g[g];
			if (g < nGroups_) {
				radBoundaryRatios_copy[g] = radBoundaryRatios[g];
			}
		}

		amrex::Real gas_update_factor = 1.0;
		if (stage == 1) {
			gas_update_factor = IMEX_a32;
		}

		const int max_ite = 5;
		int ite = 0;
		for (; ite < max_ite; ++ite) {
			quokka::valarray<double, nGroups_> Rvec{};

			if constexpr (gamma_ != 1.0) {
				Ekin0 = Egastot0 - Egas0;

				AMREX_ASSERT(min(Src) >= 0.0);
				AMREX_ASSERT(Egas0 > 0.0);

				const double betaSqr = (x1GasMom0 * x1GasMom0 + x2GasMom0 * x2GasMom0 + x3GasMom0 * x3GasMom0) / (rho * rho * c * c);

				static_assert(beta_order_ <= 3);
				if constexpr ((beta_order_ == 0) || (beta_order_ == 1)) {
					lorentz_factor = 1.0;
					lorentz_factor_v = 1.0;
				} else if constexpr (beta_order_ == 2) {
					lorentz_factor = 1.0 + 0.5 * betaSqr;
					lorentz_factor_v = 1.0;
					lorentz_factor_v_v = 1.0;
				} else if constexpr (beta_order_ == 3) {
					lorentz_factor = 1.0 + 0.5 * betaSqr;
					lorentz_factor_v = 1.0 + 0.5 * betaSqr;
					lorentz_factor_v_v = 1.0;
				} else {
					lorentz_factor = 1.0 / sqrt(1.0 - betaSqr);
					lorentz_factor_v = lorentz_factor;
					lorentz_factor_v_v = lorentz_factor;
				}

				// 1. Compute energy exchange

				// BEGIN NEWTON-RAPHSON LOOP
				// Define the source term: S = dt chat gamma rho (kappa_P B - kappa_E E) + dt chat c^-2 gamma rho kappa_F v * F_i, where gamma =
				// 1 / sqrt(1 - v^2 / c^2) is the Lorentz factor. Solve for the new radiation energy and gas internal energy using a
				// Newton-Raphson method using the base variables (Egas, D_0, D_1,
				// ...), where D_i = R_i / tau_i^(t) and tau_i^(t) = dt * chat * gamma * rho * kappa_{P,i}^(t) is the optical depth across chat
				// * dt for group i at time t. Compared with the old base (Egas, Erad_0, Erad_1, ...), this new base is more stable and
				// converges faster. Furthermore, the PlanckOpacityTempDerivative term is not needed anymore since we assume d/dT (kappa_P /
				// kappa_E) = 0 in the calculation of the Jacobian. Note that this assumption only affects the convergence rate of the
				// Newton-Raphson iteration and does not affect the result at all once the iteration is converged.
				//
				// The Jacobian of F(E_g, D_i) is
				//
				// dF_G / dE_g = 1
				// dF_G / dD_i = c / chat * tau0_i
				// dF_{D,i} / dE_g = 1 / (chat * C_v) * (kappa_{P,i} / kappa_{E,i}) * d/dT (4 \pi B_i)
				// dF_{D,i} / dD_i = - (1 / (chat * dt * rho * kappa_{E,i}) + 1) * tau0_i = - ((1 / tau_i)(kappa_Pi / kappa_Ei) + 1) * tau0_i

				Egas_guess = Egas0;
				T_gas = quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Egas_guess, massScalars);
				AMREX_ASSERT(T_gas >= 0.);
				fourPiBoverC = ComputeThermalRadiation(T_gas, radBoundaries_g_copy);

				if constexpr (opacity_model_ == OpacityModel::user) {
					kappaPVec = ComputePlanckOpacity(rho, T_gas);
					kappaEVec = ComputeEnergyMeanOpacity(rho, T_gas);
					kappaFVec = ComputeFluxMeanOpacity(rho, T_gas);
				} else if constexpr (opacity_model_ == OpacityModel::piecewisePowerLaw) {
					kappa_expo_and_lower_value = DefineOpacityExponentsAndLowerValues(radBoundaries_g_copy, rho, T_gas);
					alpha_B = ComputeRadQuantityExponents(fourPiBoverC, radBoundaries_g_copy);
					alpha_E = ComputeRadQuantityExponents(Erad0Vec, radBoundaries_g_copy);
					kappaPVec = ComputeGroupMeanOpacity(kappa_expo_and_lower_value, radBoundaryRatios_copy, alpha_B);
					kappaEVec = ComputeGroupMeanOpacity(kappa_expo_and_lower_value, radBoundaryRatios_copy, alpha_E);
				}
				AMREX_ASSERT(!kappaPVec.hasnan());
				AMREX_ASSERT(!kappaEVec.hasnan());
				AMREX_ASSERT(!kappaFVec.hasnan());

				for (int g = 0; g < nGroups_; ++g) {
					if (kappaEVec[g] > 0.0) {
						kappaPoverE[g] = kappaPVec[g] / kappaEVec[g];
					} else {
						kappaPoverE[g] = 1.0;
					}
				}

				if constexpr ((beta_order_ != 0) && (include_work_term_in_source)) {
					// compute the work term at the old state
					// const double gamma = 1.0 / sqrt(1.0 - vsqr / (c * c));
					if (ite == 0) {
						if constexpr (opacity_model_ == OpacityModel::user) {
							for (int g = 0; g < nGroups_; ++g) {
								// work[g] = dt * chat * rho * kappaPVec[g] * (Erad0Vec[g] - fourPiBoverC[g]);
								const double frad0 = consPrev(i, j, k, x1RadFlux_index + numRadVars_ * g);
								const double frad1 = consPrev(i, j, k, x2RadFlux_index + numRadVars_ * g);
								const double frad2 = consPrev(i, j, k, x3RadFlux_index + numRadVars_ * g);
								// work = v * F * chi
								work[g] = (x1GasMom0 * frad0 + x2GasMom0 * frad1 + x3GasMom0 * frad2) *
									  (2.0 * kappaEVec[g] - kappaFVec[g]);
								work[g] *= chat / (c * c) * lorentz_factor_v * dt;
							}
						} else if constexpr (opacity_model_ == OpacityModel::piecewisePowerLaw) {
							for (int g = 0; g < nGroups_; ++g) {
								frad[0][g] = consPrev(i, j, k, x1RadFlux_index + numRadVars_ * g);
								frad[1][g] = consPrev(i, j, k, x2RadFlux_index + numRadVars_ * g);
								frad[2][g] = consPrev(i, j, k, x3RadFlux_index + numRadVars_ * g);
								work[g] = 0.0;
							}
							for (int n = 0; n < 3; ++n) {
								alpha_F = ComputeRadQuantityExponents(frad[n], radBoundaries_g_copy);
								kappaFVec =
								    ComputeGroupMeanOpacity(kappa_expo_and_lower_value, radBoundaryRatios_copy, alpha_F);
								for (int g = 0; g < nGroups_; ++g) {
									work[g] +=
									    (kappa_expo_and_lower_value[0][g] + 1.0) * gasMtm0[n] * kappaFVec[g] * frad[n][g];
								}
							}
							for (int g = 0; g < nGroups_; ++g) {
								work[g] *= chat / (c * c) * dt;
							}
						}
					}
				}

				tau0 = dt * rho * kappaPVec * chat * lorentz_factor;
				Rvec = (fourPiBoverC - Erad0Vec / kappaPoverE) * tau0 + work;
				// tau0 is used as a scaling factor for Rvec
				if constexpr (use_D_as_base) {
					for (int g = 0; g < nGroups_; ++g) {
						if (tau0[g] <= 1.0) {
							tau0[g] = 1.0;
						}
					}
					D = Rvec / tau0;
				}

				double F_G = NAN;
				double dFG_dEgas = NAN;
				double deltaEgas = NAN;
				quokka::valarray<double, nGroups_> dFG_dD{};
				quokka::valarray<double, nGroups_> dFR_dEgas{};
				quokka::valarray<double, nGroups_> dFR_i_dD_i{};
				quokka::valarray<double, nGroups_> deltaD{};
				quokka::valarray<double, nGroups_> F_D{};

				const double resid_tol = 1.0e-11; // 1.0e-15;
				const int maxIter = 400;
				int n = 0;
				for (; n < maxIter; ++n) {
					// compute material temperature
					T_gas = quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Egas_guess, massScalars);
					AMREX_ASSERT(T_gas >= 0.);
					// compute opacity, emissivity
					fourPiBoverC = ComputeThermalRadiation(T_gas, radBoundaries_g_copy);

					if constexpr (opacity_model_ == OpacityModel::user) {
						kappaPVec = ComputePlanckOpacity(rho, T_gas);
						kappaEVec = ComputeEnergyMeanOpacity(rho, T_gas);
					} else if constexpr (opacity_model_ == OpacityModel::piecewisePowerLaw) {
						kappa_expo_and_lower_value = DefineOpacityExponentsAndLowerValues(radBoundaries_g_copy, rho, T_gas);
						alpha_B = ComputeRadQuantityExponents(fourPiBoverC, radBoundaries_g_copy);
						alpha_E = ComputeRadQuantityExponents(Erad0Vec, radBoundaries_g_copy);
						kappaPVec = ComputeGroupMeanOpacity(kappa_expo_and_lower_value, radBoundaryRatios_copy, alpha_B);
						kappaEVec = ComputeGroupMeanOpacity(kappa_expo_and_lower_value, radBoundaryRatios_copy, alpha_E);
					}
					AMREX_ASSERT(!kappaPVec.hasnan());
					AMREX_ASSERT(!kappaEVec.hasnan());

					for (int g = 0; g < nGroups_; ++g) {
						if (kappaEVec[g] > 0.0) {
							kappaPoverE[g] = kappaPVec[g] / kappaEVec[g];
						} else {
							kappaPoverE[g] = 1.0;
						}
					}

					tau = dt * rho * kappaEVec * chat * lorentz_factor;
					if constexpr (use_D_as_base) {
						Rvec = tau0 * D;
					}
					for (int g = 0; g < nGroups_; ++g) {
						// If tau = 0.0, Erad_guess shouldn't change
						if (tau[g] > 0.0) {
							EradVec_guess[g] = kappaPoverE[g] * (fourPiBoverC[g] - (Rvec[g] - work[g]) / tau[g]);
						}
					}
					// F_G = Egas_guess - Egas0 + (c / chat) * sum(Rvec);
					F_G = Egas_guess - Egas0;
					F_D = EradVec_guess - Erad0Vec - (Rvec + Src);
					double F_D_abs_sum = 0.0;
					for (int g = 0; g < nGroups_; ++g) {
						if (tau[g] > 0.0) {
							F_D_abs_sum += std::abs(F_D[g]);
							F_G += (c / chat) * Rvec[g];
						}
					}

					// check relative convergence of the residuals
					if ((std::abs(F_G / Etot0) < resid_tol) && ((c / chat) * F_D_abs_sum / Etot0 < resid_tol)) {
						break;
					}

					const double c_v = quokka::EOS<problem_t>::ComputeEintTempDerivative(rho, T_gas, massScalars); // Egas = c_v * T

					const auto dfourPiB_dTgas = chat * ComputeThermalRadiationTempDerivative(T_gas, radBoundaries_g_copy);
					AMREX_ASSERT(!dfourPiB_dTgas.hasnan());

					// compute Jacobian elements
					// I assume (kappaPVec / kappaEVec) is constant here. This is usually a reasonable assumption. Note that this assumption
					// only affects the convergence rate of the Newton-Raphson iteration and does not affect the converged solution at all.
					dFG_dEgas = 1.0;
					for (int g = 0; g < nGroups_; ++g) {
						if (tau[g] <= 0.0) {
							dFR_i_dD_i[g] = -std::numeric_limits<double>::infinity();
						} else {
							dFR_i_dD_i[g] = -1.0 * (1.0 / tau[g] * kappaPoverE[g] + 1.0);
						}
					}
					if constexpr (use_D_as_base) {
						dFG_dD = (c / chat) * tau0;
						dFR_i_dD_i = dFR_i_dD_i * tau0;
					} else {
						dFG_dD.fillin(c / chat);
					}
					dFR_dEgas = 1.0 / c_v * kappaPoverE * (dfourPiB_dTgas / chat);

					// update variables
					RadSystem<problem_t>::SolveLinearEqs(dFG_dEgas, dFG_dD, dFR_dEgas, dFR_i_dD_i, -F_G, -1. * F_D, deltaEgas, deltaD);
					AMREX_ASSERT(!std::isnan(deltaEgas));
					AMREX_ASSERT(!deltaD.hasnan());

					Egas_guess += deltaEgas;
					if constexpr (use_D_as_base) {
						D += deltaD;
					} else {
						Rvec += deltaD;
					}

					// check relative and absolute convergence of E_r
					// if (std::abs(deltaEgas / Egas_guess) < 1e-7) {
					// 	break;
					// }
				} // END NEWTON-RAPHSON LOOP

				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n < maxIter, "Newton-Raphson iteration failed to converge!");
				// std::cout << "Newton-Raphson converged after " << n << " it." << std::endl;
				AMREX_ALWAYS_ASSERT(Egas_guess > 0.0);
				AMREX_ALWAYS_ASSERT(min(EradVec_guess) >= 0.0);
			} // endif gamma != 1.0

			// Erad_guess is the new radiation energy (excluding work term)
			// Egas_guess is the new gas internal energy

			// 2. Compute radiation flux update

			amrex::GpuArray<amrex::Real, 3> Frad_t0{};

			T_gas = quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Egas_guess, massScalars);
			if constexpr (gamma_ != 1.0) {
				fourPiBoverC = ComputeThermalRadiation(T_gas, radBoundaries_g_copy);
			}

			if constexpr (opacity_model_ == OpacityModel::user) {
				if constexpr (gamma_ != 1.0) {
					kappaPVec = ComputePlanckOpacity(rho, T_gas);
					kappaEVec = ComputeEnergyMeanOpacity(rho, T_gas);
					AMREX_ASSERT(!kappaPVec.hasnan());
					AMREX_ASSERT(!kappaEVec.hasnan());
				}
				kappaFVec = ComputeFluxMeanOpacity(rho, T_gas); // note that kappaFVec is used no matter what the value of gamma is
			} else if constexpr (opacity_model_ == OpacityModel::piecewisePowerLaw) {
				kappa_expo_and_lower_value = DefineOpacityExponentsAndLowerValues(radBoundaries_g_copy, rho, T_gas);
				if constexpr (gamma_ != 1.0) {
					alpha_B = ComputeRadQuantityExponents(fourPiBoverC, radBoundaries_g_copy);
					alpha_E = ComputeRadQuantityExponents(EradVec_guess, radBoundaries_g_copy);
					kappaPVec = ComputeGroupMeanOpacity(kappa_expo_and_lower_value, radBoundaryRatios_copy, alpha_B);
					kappaEVec = ComputeGroupMeanOpacity(kappa_expo_and_lower_value, radBoundaryRatios_copy, alpha_E);
					AMREX_ASSERT(!kappaPVec.hasnan());
					AMREX_ASSERT(!kappaEVec.hasnan());
				}
				// Note that alpha_F has not been changed in the Newton iteration
				kappaFVec = ComputeGroupMeanOpacity(kappa_expo_and_lower_value, radBoundaryRatios_copy, alpha_F);
			}
			AMREX_ASSERT(!kappaFVec.hasnan());

			dMomentum = {0., 0., 0.};

			for (int g = 0; g < nGroups_; ++g) {

				Frad_t0[0] = consPrev(i, j, k, x1RadFlux_index + numRadVars_ * g);
				Frad_t0[1] = consPrev(i, j, k, x2RadFlux_index + numRadVars_ * g);
				Frad_t0[2] = consPrev(i, j, k, x3RadFlux_index + numRadVars_ * g);

				if constexpr ((gamma_ != 1.0) && (beta_order_ != 0)) {
					auto erad = EradVec_guess[g];
					std::array<double, 3> gasVel{};
					std::array<double, 3> v_terms{};

					auto Fx = Frad_t0[0];
					auto Fy = Frad_t0[1];
					auto Fz = Frad_t0[2];
					auto fx = Fx / (c_light_ * erad);
					auto fy = Fy / (c_light_ * erad);
					auto fz = Fz / (c_light_ * erad);
					double F_coeff = chat * rho * kappaFVec[g] * dt * lorentz_factor;
					auto Tedd = ComputeEddingtonTensor(fx, fy, fz);

					for (int n = 0; n < 3; ++n) {
						// compute thermal radiation term
						double v_term = NAN;

						if constexpr (opacity_model_ == OpacityModel::user) {
							v_term = kappaPVec[g] * fourPiBoverC[g] * lorentz_factor_v;
							// compute (kappa_F - kappa_E) term
							if (kappaFVec[g] != kappaEVec[g]) {
								v_term += (kappaFVec[g] - kappaEVec[g]) * erad * std::pow(lorentz_factor_v, 3);
							}
						} else if constexpr (opacity_model_ == OpacityModel::piecewisePowerLaw) {
							v_term = kappaPVec[g] * fourPiBoverC[g] * (2.0 - kappa_expo_and_lower_value[0][g] - alpha_B[g]) / 3.0;
						}

						v_term *= chat * dt * gasMtm0[n];

						// compute radiation pressure
						double pressure_term = 0.0;
						for (int z = 0; z < 3; ++z) {
							pressure_term += gasMtm0[z] * Tedd[n][z] * erad;
						}
						if constexpr (opacity_model_ == OpacityModel::user) {
							pressure_term *= chat * dt * kappaFVec[g] * lorentz_factor_v;
						} else if constexpr (opacity_model_ == OpacityModel::piecewisePowerLaw) {
							pressure_term *= chat * dt * kappaEVec[g] * (kappa_expo_and_lower_value[0][g] + 1.0);
						}

						v_term += pressure_term;
						v_terms[n] = v_term;
					}

					if constexpr (beta_order_ == 1) {
						for (int n = 0; n < 3; ++n) {
							// Compute flux update
							Frad_t1[n][g] = (Frad_t0[n] + v_terms[n]) / (1.0 + F_coeff);

							// Compute conservative gas momentum update
							dMomentum[n] += -(Frad_t1[n][g] - Frad_t0[n]) / (c * chat);
						}
					} else {
						if (kappaFVec[g] == kappaEVec[g]) {
							for (int n = 0; n < 3; ++n) {
								// Compute flux update
								Frad_t1[n][g] = (Frad_t0[n] + v_terms[n]) / (1.0 + F_coeff);

								// Compute conservative gas momentum update
								dMomentum[n] += -(Frad_t1[n][g] - Frad_t0[n]) / (c * chat);
							}
						} else {
							const double K0 =
							    2.0 * rho * chat * dt * (kappaFVec[g] - kappaEVec[g]) / c / c * std::pow(lorentz_factor_v_v, 3);

							// A test to see if this routine reduces to the correct result when ignoring the beta^2 terms
							// const double X0 = 1.0 + rho * chat * dt * (kappaFVec[g]);
							// const double K0 = 0.0;

							// Solve 3x3 matrix equation A * x = B, where A[i][j] = delta_ij * X0 + K0 * v_i * v_j and B[i] =
							// O_beta_tau_terms[i] + Frad_t0[i]
							const double A00 = 1.0 + F_coeff + K0 * gasVel[0] * gasVel[0];
							const double A01 = K0 * gasVel[0] * gasVel[1];
							const double A02 = K0 * gasVel[0] * gasVel[2];

							const double A10 = K0 * gasVel[1] * gasVel[0];
							const double A11 = 1.0 + F_coeff + K0 * gasVel[1] * gasVel[1];
							const double A12 = K0 * gasVel[1] * gasVel[2];

							const double A20 = K0 * gasVel[2] * gasVel[0];
							const double A21 = K0 * gasVel[2] * gasVel[1];
							const double A22 = 1.0 + F_coeff + K0 * gasVel[2] * gasVel[2];

							const double B0 = v_terms[0] + Frad_t0[0];
							const double B1 = v_terms[1] + Frad_t0[1];
							const double B2 = v_terms[2] + Frad_t0[2];

							auto [sol0, sol1, sol2] = Solve3x3matrix(A00, A01, A02, A10, A11, A12, A20, A21, A22, B0, B1, B2);
							Frad_t1[0][g] = sol0;
							Frad_t1[1][g] = sol1;
							Frad_t1[2][g] = sol2;
							for (int n = 0; n < 3; ++n) {
								dMomentum[n] += -(Frad_t1[n][g] - Frad_t0[n]) / (c * chat);
							}
						}
					}
				} else {
					for (int n = 0; n < 3; ++n) {
						Frad_t1[n][g] = Frad_t0[n] / (1.0 + rho * kappaFVec[g] * chat * dt);
						// Compute conservative gas momentum update
						dMomentum[n] += -(Frad_t1[n][g] - Frad_t0[n]) / (c * chat);
					}
				}
			} // end loop over radiation groups for flux update

			amrex::Real const x1GasMom1 = consPrev(i, j, k, x1GasMomentum_index) + dMomentum[0];
			amrex::Real const x2GasMom1 = consPrev(i, j, k, x2GasMomentum_index) + dMomentum[1];
			amrex::Real const x3GasMom1 = consPrev(i, j, k, x3GasMomentum_index) + dMomentum[2];

			// 3. Deal with the work term.
			if constexpr ((gamma_ != 1.0) && (beta_order_ != 0)) {
				// compute difference in gas kinetic energy before and after momentum update
				amrex::Real const Egastot1 = ComputeEgasFromEint(rho, x1GasMom1, x2GasMom1, x3GasMom1, Egas_guess);
				amrex::Real const Ekin1 = Egastot1 - Egas_guess;
				amrex::Real const dEkin_work = Ekin1 - Ekin0;

				if constexpr (include_work_term_in_source) {
					// New scheme: the work term is included in the source terms. The work done by radiation went to internal energy, but it
					// should go to the kinetic energy. Remove the work term from internal energy.
					Egas_guess -= dEkin_work;
				} else {
					// Old scheme: since the source term does not include work term, add the work term to radiation energy.

					// compute loss of radiation energy to gas kinetic energy
					auto dErad_work = -(c_hat_ / c_light_) * dEkin_work;

					// apportion dErad_work according to kappaF_i * (v * F_i)
					quokka::valarray<double, nGroups_> energyLossFractions{};
					if constexpr (nGroups_ == 1) {
						energyLossFractions[0] = 1.0;
					} else {
						// compute energyLossFractions
						for (int g = 0; g < nGroups_; ++g) {
							energyLossFractions[g] =
							    kappaFVec[g] * (x1GasMom1 * Frad_t1[0][g] + x2GasMom1 * Frad_t1[1][g] + x3GasMom1 * Frad_t1[2][g]);
						}
						auto energyLossFractionsTot = sum(energyLossFractions);
						if (energyLossFractionsTot != 0.0) {
							energyLossFractions /= energyLossFractionsTot;
						} else {
							energyLossFractions.fillin(0.0);
						}
					}
					for (int g = 0; g < nGroups_; ++g) {
						auto radEnergyNew = EradVec_guess[g] + dErad_work * energyLossFractions[g];
						// AMREX_ASSERT(radEnergyNew > 0.0);
						if (radEnergyNew < Erad_floor_) {
							// return energy to Egas_guess
							Egas_guess -= (Erad_floor_ - radEnergyNew) * (c / chat);
							radEnergyNew = Erad_floor_;
						}
						EradVec_guess[g] = radEnergyNew;
					}
				}
			} // End of step 3

			if constexpr ((beta_order_ == 0) || (gamma_ == 1.0) || (!include_work_term_in_source)) {
				break;
			}

			// If you are here, then you are using the new scheme. Step 3 is skipped. The work term is included in the source term, but it is
			// lagged. The work term is updated in the next step.
			for (int g = 0; g < nGroups_; ++g) {
				// copy work to work_prev
				work_prev[g] = work[g];
				// compute new work term from the updated radiation flux and velocity
				if constexpr (opacity_model_ == OpacityModel::user) {
					work[g] = (x1GasMom1 * Frad_t1[0][g] + x2GasMom1 * Frad_t1[1][g] + x3GasMom1 * Frad_t1[2][g]) * chat / (c * c) *
						  lorentz_factor_v * (2.0 * kappaEVec[g] - kappaFVec[g]) * dt;
				} else if constexpr (opacity_model_ == OpacityModel::piecewisePowerLaw) {
					for (int n = 0; n < 3; ++n) {
						work[n] = 0.0;
					}
					for (int n = 0; n < 3; ++n) {
						alpha_F = ComputeRadQuantityExponents(Frad_t1[n], radBoundaries_g_copy);
						kappaFVec = ComputeGroupMeanOpacity(kappa_expo_and_lower_value, radBoundaryRatios_copy, alpha_F);
						for (int g = 0; g < nGroups_; ++g) {
							work[g] += (kappa_expo_and_lower_value[0][g] + 1.0) * gasMtm0[n] * kappaFVec[g] * Frad_t1[n][g];
						}
					}
					for (int g = 0; g < nGroups_; ++g) {
						work[g] *= chat * dt / (c * c);
					}
				}
			}

			// Check for convergence of the work term: if the relative change in the work term is less than 1e-13, then break the loop
			const double lag_tol = 1.0e-13;
			if ((sum(abs(work)) == 0.0) || ((c / chat) * sum(abs(work - work_prev)) / Etot0 < lag_tol) ||
			    (sum(abs(work - work_prev)) <= lag_tol * sum(Rvec))) {
				break;
			}
		} // end full-step iteration

		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ite < max_ite, "AddSourceTerms iteration failed to converge!");

		// 4b. Store new radiation energy, gas energy
		// In the first stage of the IMEX scheme, the hydro quantities are updated by a fraction (defined by
		// gas_update_factor) of the time step.
		const auto x1GasMom1 = consPrev(i, j, k, x1GasMomentum_index) + dMomentum[0] * gas_update_factor;
		const auto x2GasMom1 = consPrev(i, j, k, x2GasMomentum_index) + dMomentum[1] * gas_update_factor;
		const auto x3GasMom1 = consPrev(i, j, k, x3GasMomentum_index) + dMomentum[2] * gas_update_factor;
		consNew(i, j, k, x1GasMomentum_index) = x1GasMom1;
		consNew(i, j, k, x2GasMomentum_index) = x2GasMom1;
		consNew(i, j, k, x3GasMomentum_index) = x3GasMom1;
		if constexpr (gamma_ != 1.0) {
			Egas_guess = Egas0 + (Egas_guess - Egas0) * gas_update_factor;
			consNew(i, j, k, gasInternalEnergy_index) = Egas_guess;
			consNew(i, j, k, gasEnergy_index) = ComputeEgasFromEint(rho, x1GasMom1, x2GasMom1, x3GasMom1, Egas_guess);
		} else {
			amrex::ignore_unused(EradVec_guess);
			amrex::ignore_unused(Egas_guess);
		}
		for (int g = 0; g < nGroups_; ++g) {
			if constexpr (gamma_ != 1.0) {
				consNew(i, j, k, radEnergy_index + numRadVars_ * g) = EradVec_guess[g];
			}
			consNew(i, j, k, x1RadFlux_index + numRadVars_ * g) = Frad_t1[0][g];
			consNew(i, j, k, x2RadFlux_index + numRadVars_ * g) = Frad_t1[1][g];
			consNew(i, j, k, x3RadFlux_index + numRadVars_ * g) = Frad_t1[2][g];
		}
	});
}

// CCH: this function ComputeSourceTermsExplicit is never used.
template <typename problem_t>
void RadSystem<problem_t>::ComputeSourceTermsExplicit(arrayconst_t &consPrev, arrayconst_t & /*radEnergySource*/, array_t &src, amrex::Box const &indexRange,
						      amrex::Real dt)
{
	const double chat = c_hat_;
	const double a_rad = radiation_constant_;

	// cell-centered kernel
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// load gas energy
		const auto rho = consPrev(i, j, k, gasDensity_index);
		const auto Egastot0 = consPrev(i, j, k, gasEnergy_index);
		const auto x1GasMom0 = consPrev(i, j, k, x1GasMomentum_index);
		const double x2GasMom0 = consPrev(i, j, k, x2GasMomentum_index);
		const double x3GasMom0 = consPrev(i, j, k, x3GasMomentum_index);
		const auto Egas0 = ComputeEintFromEgas(rho, x1GasMom0, x2GasMom0, x3GasMom0, Egastot0);
		auto massScalars = RadSystem<problem_t>::ComputeMassScalars(consPrev, i, j, k);

		// load radiation energy, momentum
		const auto Erad0 = consPrev(i, j, k, radEnergy_index);
		const auto Frad0_x = consPrev(i, j, k, x1RadFlux_index);

		// compute material temperature
		const auto T_gas = quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Egas0, massScalars);

		// compute opacity, emissivity
		const auto kappa = RadSystem<problem_t>::ComputeOpacity(rho, T_gas);
		const auto fourPiB = chat * a_rad * std::pow(T_gas, 4);

		// compute reaction term
		const auto rhs = dt * (rho * kappa) * (fourPiB - chat * Erad0);
		const auto Fx_rhs = -dt * chat * (rho * kappa) * Frad0_x;

		src(radEnergy_index, i) = rhs;
		src(x1RadFlux_index, i) = Fx_rhs;
	});
}

#endif // RADIATION_SYSTEM_HPP_
