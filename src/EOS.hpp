#ifndef EOS_HPP_
#define EOS_HPP_
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file EOS.hpp
/// \brief A class for equation of state calculations.

#include <cmath>

#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

namespace quokka
{
static constexpr double boltzmann_constant_cgs = 1.380658e-16; // cgs
static constexpr double hydrogen_mass_cgs = 1.6726231e-24;     // cgs

// specify default values for ideal gamma-law EOS
//
template <typename problem_t> struct EOS_Traits {
	static constexpr double gamma = 5. / 3.;     // default value
	static constexpr double cs_isothermal = NAN; // only used when gamma = 1
	static constexpr double mean_molecular_weight = NAN;
	static constexpr double boltzmann_constant = boltzmann_constant_cgs;
};

template <typename problem_t> class EOS
{
      public:
	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto ComputeTgasFromEint(amrex::Real rho, amrex::Real Eint) -> amrex::Real;
	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto ComputeEintFromTgas(amrex::Real rho, amrex::Real Tgas) -> amrex::Real;
	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto ComputeEintTempDerivative(amrex::Real rho, amrex::Real Tgas) -> amrex::Real;

      private:
	static constexpr amrex::Real gamma_ = EOS_Traits<problem_t>::gamma;
	static constexpr amrex::Real boltzmann_constant_ = EOS_Traits<problem_t>::boltzmann_constant;
	static constexpr amrex::Real mean_molecular_weight_ = EOS_Traits<problem_t>::mean_molecular_weight;
};

template <typename problem_t>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto EOS<problem_t>::ComputeTgasFromEint(amrex::Real rho, amrex::Real Eint) -> amrex::Real
{
	// return temperature for an ideal gas
	amrex::Real Tgas = NAN;
	if constexpr (gamma_ != 1.0) {
		const amrex::Real c_v = boltzmann_constant_ / (mean_molecular_weight_ * (gamma_ - 1.0));
		Tgas = Eint / (rho * c_v);
	}
	return Tgas;
}

template <typename problem_t>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto EOS<problem_t>::ComputeEintFromTgas(amrex::Real rho, amrex::Real Tgas) -> amrex::Real
{
	// return internal energy density for a gamma-law ideal gas
	amrex::Real Eint = NAN;
	if constexpr (gamma_ != 1.0) {
		const amrex::Real c_v = boltzmann_constant_ / (mean_molecular_weight_ * (gamma_ - 1.0));
		Eint = rho * c_v * Tgas;
	}
	return Eint;
}

template <typename problem_t>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto EOS<problem_t>::ComputeEintTempDerivative(const amrex::Real rho, const amrex::Real /*Tgas*/) -> amrex::Real
{
	// compute derivative of internal energy w/r/t temperature
	amrex::Real dEint_dT = NAN;
	if constexpr (gamma_ != 1.0) {
		const amrex::Real c_v = boltzmann_constant_ / (mean_molecular_weight_ * (gamma_ - 1.0));
		dEint_dT = rho * c_v;
	}
	return dEint_dT;
}

} // namespace quokka

#endif // EOS_HPP_
