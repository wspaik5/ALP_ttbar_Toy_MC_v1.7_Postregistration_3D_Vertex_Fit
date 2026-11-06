#!/usr/bin/env bash
# Reproduce all eleven figures. Figures 1-5 take about 5 minutes; the v1.3 mini tracking simulation
# (Figure 6) about 2.5 hours and the v1.4 fits (Figure 7) about 3.5 hours, the v1.5 fits (Figure 8) about 6 hours on one core.
set -e
mkdir -p figures
python3 plot_efficiency.py  | tee figures/fig1_numbers.txt
python3 plot_bfield.py      | tee figures/fig2_numbers.txt
python3 plot_conversions.py | tee figures/fig3_numbers.txt
python3 plot_vertex.py      | tee figures/fig4_numbers.txt
python3 plot_hitpattern.py  | tee figures/fig5_numbers.txt
# v1.3
python3 validate_v13.py     | tee figures/validation_v13.txt
python3 run_v13.py          | tee figures/run_v13_output_A_prereg.txt
python3 run_v13.py parallel | tee figures/run_v13_output_B_parallel.txt
python3 explore_nobrems_v13.py | tee figures/explore_nobrems_v13_output.txt
python3 plot_minitrack.py   | tee figures/fig6_numbers.txt
# v1.4 (needs v13_conv_arrays_B_parallel.npz from the v1.3 step above for Figure 7 and the identity check)
python3 validate_v14.py     | tee figures/validation_v14.txt
python3 run_v14.py sim      | tee figures/run_v14_sim.txt
python3 run_v14.py fit a6 && python3 run_v14.py fit b && python3 run_v14.py fit a4 && python3 run_v14.py fit a8
python3 run_v14.py analyze  | tee figures/run_v14_output.txt
python3 plot_bremsfit.py    | tee figures/fig7_numbers.txt
# v1.5 (uses v14_hits.npz and v14_fit_b.npz from the v1.4 step above)
python3 validate_v15.py     | tee figures/validation_v15.txt
for k in conv sig0.017 sig0.05 sig0.1 sig0.2; do python3 run_v15.py fit $k 2h; python3 run_v15.py fit $k ab; done
python3 run_v15.py analyze  | tee figures/run_v15_output.txt
python3 plot_fig8.py
python3 x1_boundary.py      | tee figures/x1_boundary.txt     # post-hoc X1 diagnosis (Figure 9)
# v1.6 field and material scan (two queues in parallel, about 3 hours on two cores)
for B in 0.2 2.0; do V16_B=$B python3 validate_v16.py y1 | tee -a figures/validation_v16.txt; done
for B in 3.8 2.0 1.0 0.5 0.2; do V16_B=$B python3 validate_v16.py y2 | tee -a figures/validation_v16.txt; done
V16_B=0.2 python3 validate_v16.py y3 | tee -a figures/validation_v16.txt
./runq.sh R:3.8:1.0 B2:2.0:1.0 B05:0.5:1.0 M025:3.8:0.25 & ./runq.sh B02:0.2:1.0 B1:1.0:1.0 M05:3.8:0.5 C:2.0:0.5; wait
python3 run_v16.py analyze | tee figures/run_v16_output.txt     # needs ../v15/v15_fit_2h_*.npz for the identity gate
python3 plot_fig10.py
# v1.7 three-dimensional two-hypothesis fit (needs ../v16/v16_R_*.npz from the v1.6 step for the identity gate)
python3 validate_v17.py z3 z1 z2 | tee figures/validation_v17.txt
python3 run_v17.py fit conv & for k in sig0.017 sig0.05 sig0.1 sig0.2; do python3 run_v17.py fit $k; done; wait
python3 run_v17.py analyze | tee figures/run_v17_output.txt
python3 plot_fig11.py
