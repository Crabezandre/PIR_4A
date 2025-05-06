#!/usr/bin/python

from snif import *
from snif2tex import *

h_inference_parameters = InferenceParameters(
    data_source = './Generate_PSMC/sim_0403.psmc',
    source_type = SourceType.PSMC,
    IICR_type = IICRType.Exact,
    ms_reference_size = None,
    ms_simulations = None,
    psmc_mutation_rate =1e-08 ,
    psmc_number_of_sequences = None,
    psmc_length_of_sequences = None,
    infer_scale = True,
    data_cutoff_bounds = (1, 2e7),
    data_time_intervals = 1000,
    distance_function = ErrorFunction.ApproximatePDF,
    distance_parameter = 0.5,
    distance_max_allowed = 7e3,
    distance_computation_interval = (1, 2e7),
    rounds_per_test_bounds = (5, 5),
    repetitions_per_test = 5,
    number_of_components = 1,
    bounds_islands = (2, 200),
    bounds_migrations_rates = (0.05, 20),
    bounds_deme_sizes = (1, 1),
    bounds_event_times = (1, 2e7),
    bounds_effective_size = (10, 10000)
)

h_settings = Settings(
    static_library_location = './libs/libsnif.so',
    custom_filename_tag = '1208',
    output_directory = './SNIF_results/1208',
    default_output_dirname = './SNIF_results/1208'
)

basename = infer(
    inf = h_inference_parameters,
    settings = h_settings
)

config = Configuration(
        SNIF_basename = basename,
        plot_width_cm = 13,
        plot_height_cm = 6,
        IICR_plots_style  = OutputStyle.Full,
        PDF_plots_style = OutputStyle.Excluded,
        CDF_plots_style = OutputStyle.Excluded,
        islands_plot_style = OutputStyle.Excluded,
        Nref_plot_style = OutputStyle.Excluded,
        test_numbers = "all",
        one_file_per_test = False,
        versus_plot_style = OutputStyle.Excluded,
        CG_style = OutputStyle.Excluded,
        CG_size_history = False,
        CG_source = '',
        CG_source_legend = "",
        Nref_histograms_bins = 100,
        islands_histograms_bins = 100,
        time_histograms_bins = 100,
        migration_histograms_bins = 100,
        size_histograms_bins = 100,
        scaling_units = TimeScale.Years,
        generation_time = 1 
)
TeXify(config)
