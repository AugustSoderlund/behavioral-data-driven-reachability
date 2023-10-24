import warnings

warnings.filterwarnings("ignore")

from utils.visualization import *
from utils.operations import (
    visualize_zonotopes,
    create_M_w,
    input_zonotope,
    minkowski_sum,
    linear_map,
)
from utils.reachability import LTI_reachability
from utils.input_state import *
from utils.data_reader import SinD, LABELS
from utils.data_processor import *
from utils.zonotope import zonotope
import numpy as np
from utils.simulation import (
    calc_d,
    generate_trajectory,
    simulate_forged_traj,
    _simulation,
    run_scenario,
    vis_class_trajs,
    format_volume_calc_for_latex,
    visualize_scenario,
    visualize_state_inclusion_acc,
    intersection_figure,
)
from utils.simulation import reachability_for_specific_position_and_mode as RA
from utils.simulation import reachability_for_all_modes as RA_all
import matplotlib.pyplot as plt


def get_modal_zonotopes_for_scenarios():
    """Code to reproduce the modal zonotopes and baseline zonotopes
    for the 3 scenarios.
    NOTE: You will need the full dataset
    """

    if "sind.pkl" not in os.listdir(ROOT + "/resources"):
        sind = SinD()
        data = sind.data(input_len=90)
        _ = sind.labels(data, input_len=90)
    _dict = {
        1: ["cross_right", "cross_straight"],
        2: ["cross_left", "cross_straight"],
        3: ["crossing_now", "cross_illegal"],
    }
    for i in range(1, 4):
        print(f"Running scenario {i}")
        run_scenario(i)
    for i in range(1, 4):
        try:
            visualize_scenario(
                scenario=str(i), all_modes=False, save=True, overlay_image=True
            )
        except Exception:
            print(f"Not possible to visualize any modal zonotopes for scenario {i}")
            break
        plt.close("all")
        for mode in _dict[i]:
            try:
                fig, ax = plt.subplots()
                fig.subplots_adjust(top=1, left=0, bottom=0, right=1)
                im = plt.imread(os.getcwd() + f"/resources/scenario{i}{mode}.png")
                ax.imshow(im)
                plt.axis("off")
                plt.show()
            except Exception:
                print(f"{mode} not computed for scenario {i}. Solutions:")
                print(" - Get the full dataset")
                print(" - Try running code again")


def get_volumes():
    """Code to reproduce the volumes of the zonotopes"""
    for i in range(1, 4):
        print(f"Running scenario {i}")
        run_scenario(i)
    format_volume_calc_for_latex()


def get_state_inclusion_acc():
    """Code to reproduce the state inclusion accuracy

    NOTE: This might take forever to compute, try one day's
    worth of data under /resources/SinD/Data/<8_02_1> and/or
    change the value of the 'frames' argument.
    """
    _simulation(load_data=False, load_calc_d_data=False, frames=400)
    visualize_state_inclusion_acc()


if __name__ == "__main__":
    # NOTE: Get the full dataset for best results

    # To reproduce the results in Fig. 4
    get_modal_zonotopes_for_scenarios()

    # To reproduce the results in Table II
    get_volumes()

    # To reproduce the results in Fig. 5
    get_state_inclusion_acc()
