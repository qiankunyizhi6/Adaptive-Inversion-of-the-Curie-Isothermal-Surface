Comparison of the recovered Curie depth with previous models
============================================================

Purpose
-------
These figures place the Curie-depth field produced by this repository next to three
independently published Curie-depth models of the same region, and next to the crustal
structure, so that the differences between the models can be read directly.


1. Latitude profile at 37 N
---------------------------
File: latitude_profile_37N.png

Panels from top to bottom:
  * topography along the profile;
  * Curie depth along the profile for
      - this study            (solid red)
      - Li et al., 2017       (purple dash-dot, global reference model)
      - Gao et al., 2015      (green dash-dot, continual-model inversion of the Ordos)
      - Zhou et al., 2024     (blue dotted, adaptive-window method)
      - CRUST1.0 Moho depth   (solid yellow)
    on the left axis;
  * Bouguer gravity anomaly (brown) on the right axis.

The vertical dashed lines mark the western margin of the Ordos Basin, the eastern margin of
the basin and the Taihang Mountains, so that the same tectonic units can be followed from
panel to panel.

What the profile shows
----------------------
  * Within the basin interior the four Curie-depth models agree on a deep magnetic bottom
    and on the general shape of the curve, while the amplitude of the local variations
    differs between them; the additional lateral detail of the present model reflects the
    block-wise optimization of the spectral parameters.
  * Across the western margin the published models converge towards a much shallower Curie
    depth than the present model, which instead stays close to the Moho depth. This is the
    clearest difference between the models in the study area.
  * The Bouguer anomaly increases eastwards along the profile while the Moho shallows,
    consistent with the crustal thinning of the eastern North China Craton.


2. Heat-flow agreement
----------------------
Files: heat_flow_agreement_metrics.png (RMSE, MAE and bias per model)
       heat_flow_agreement_scatter.png (predicted versus observed heat flow)

Each Curie-depth model is converted to surface heat flow with the same heat-conduction
model and compared with the measured heat flow. The curves are labelled:

  data_jlm                variant of the model of this study
  final_curie_depth       variant of the model of this study
  curie_depth_map         variant of the model of this study
  41598_2017_srep45129    published model
  curie_isotherm_depth_xyz published model

The RMSE and MAE of the models of this study are the lowest of the set
(see ../validation/cross_validation_summary.txt for the five-fold values). The scatter
panels show that the predicted heat flow of the present model follows the observed values
in the range where the heat-flow sites are dense, and that the largest residuals come from
the sites with the highest observed heat flow.


Note on the code
----------------
The scripts that produce these figures compare several Curie-depth models and their
conversion to heat flow; they are not part of the inversion workflow and are therefore not
included in this repository. The figures are provided as supporting material, and the scripts
can be made available on request.
