//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file GrackleLikeCooling.cpp
/// \brief Implements methods for interpolating cooling rates from Grackle
/// tables.
///

#include "GrackleLikeCooling.hpp"

namespace quokka::GrackleLikeCooling
{

void readGrackleData(std::string &grackle_hdf5_file, grackle_tables &cloudyTables)
{
	grackle_data cloudy_primordial;
	grackle_data cloudy_metals;
	code_units my_units; // cgs
	my_units.density_units = 1.0;
	my_units.length_units = 1.0;
	my_units.time_units = 1.0;
	my_units.velocity_units = 1.0;

	initialize_cloudy_data(cloudy_primordial, "Primordial", grackle_hdf5_file, my_units);
	initialize_cloudy_data(cloudy_metals, "Metals", grackle_hdf5_file, my_units);

	cloudyTables.log_nH = std::make_unique<amrex::TableData<double, 1>>(copy_1d_table(cloudy_primordial.grid_parameters[0]));
	cloudyTables.log_Tgas = std::make_unique<amrex::TableData<double, 1>>(copy_1d_table(cloudy_primordial.grid_parameters[2]));

	cloudyTables.T_min = cloudy_primordial.T_min;
	cloudyTables.T_max = cloudy_primordial.T_max;

	int z_index = 0; // index along the redshift dimension

	cloudyTables.primCooling = std::make_unique<amrex::TableData<double, 2>>(extract_2d_table(cloudy_primordial.cooling_data, z_index));
	cloudyTables.primHeating = std::make_unique<amrex::TableData<double, 2>>(extract_2d_table(cloudy_primordial.heating_data, z_index));
	cloudyTables.mean_mol_weight = std::make_unique<amrex::TableData<double, 2>>(extract_2d_table(cloudy_primordial.mmw_data, z_index));

	cloudyTables.mmw_min = cloudy_primordial.mmw_min;
	cloudyTables.mmw_max = cloudy_primordial.mmw_max;

	cloudyTables.metalCooling = std::make_unique<amrex::TableData<double, 2>>(extract_2d_table(cloudy_metals.cooling_data, z_index));
	cloudyTables.metalHeating = std::make_unique<amrex::TableData<double, 2>>(extract_2d_table(cloudy_metals.heating_data, z_index));
}

auto grackle_tables::const_tables() const -> grackleGpuConstTables
{
	grackleGpuConstTables tables{log_nH->const_table(),
				     log_Tgas->const_table(),
				     primCooling->const_table(),
				     primHeating->const_table(),
				     metalCooling->const_table(),
				     metalHeating->const_table(),
				     mean_mol_weight->const_table(),
				     T_min,
				     T_max,
				     mmw_min,
				     mmw_max};
	return tables;
}

} // namespace quokka::GrackleLikeCooling