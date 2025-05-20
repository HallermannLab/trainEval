# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def trainEval():
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Constants (can be adjusted)
    A_To_pA = 1
    ms_To_s = 0.001
    blank_st = -0.3 * ms_To_s
    blank_end = 1.5 * ms_To_s
    base_st = -0.4 * ms_To_s
    base_end = 0.0 * ms_To_s
    peak_st = 0.5 * ms_To_s
    peak_end = 3.0 * ms_To_s
    trace_base_st = 0  # in seconds
    trace_base_end = 1

    zoomStart1 = 1.73 # in s
    zoomEnd1  = 1.75
    zoomStart2 = 1.7  # in s
    zoomEnd2 = 1.8

    time_deltaT = 0.001     # sampling interval

    # === CONFIGURATION ===
    import_folder = "/Users/stefanhallermann/Desktop/"
    filename = ("25-05-20-60Hz.xlsx")
    export_folder = os.path.join(import_folder, "export")
    os.makedirs(export_folder, exist_ok=True)

    # === IMPORT DATA ===
    df = pd.read_excel(os.path.join(import_folder, filename))
    time = df.iloc[:, 0].values  # first column = time
    time *= time_deltaT
    traces = df.iloc[:, 1:]  # remaining columns = traces

    # === STIMULATION DETECTION ===
    stim_file = "stimTimeMarker.xlsx"  # replace with your stim marker file
    stim_col_name = "stim"  # the column containing mostly 0s and occasional 1s

    stim_df = pd.read_excel(os.path.join(import_folder, stim_file))
    stim_signal = stim_df[stim_col_name].values

    # Assumes the stim file has the same time base as the main data
    # Find onset times: 0 -> 1 transitions
    stim_onsets = np.where((stim_signal[:-1] < 25) & (stim_signal[1:] >= 25))[0] + 1
    time_of_stim = time[stim_onsets]

    plt.figure()
    plt.plot(time, stim_signal, label="Stimulus Signal")
    plt.scatter(time[stim_onsets], stim_signal[stim_onsets], color='red', label="Detected Onsets")
    plt.xlabel("Time (s)")
    plt.ylabel("Stim Value")
    plt.title("Stimulus Detection")
    plt.legend()
    plt.savefig(os.path.join(export_folder, f"stimDetect.pdf"))
    plt.close()

    print("Detected stimulation times (s):", time_of_stim)

    results = []

    for trace_name in traces.columns:
        original_y = traces[trace_name].values
        y = original_y.copy()

        # Calculate trace baseline (across full trace base interval)
        trace_base_mask = (time >= trace_base_st) & (time <= trace_base_end)
        trace_base = np.mean(y[trace_base_mask])

        peak_vals = []
        base_vals = []

        # Stimulus-by-stimulus analysis
        for stim_time in time_of_stim:
            # Remove stimulation artifact by linear interpolation
            idx1 = np.searchsorted(time, stim_time + blank_st)
            idx2 = np.searchsorted(time, stim_time + blank_end)

            i1 = y[idx1]
            i2 = y[idx2]
            interp = np.linspace(0, 1, idx2 - idx1 + 1)
            y[idx1:idx2 + 1] = i1 + interp * (i2 - i1)

            # Calculate base
            base_mask = (time >= stim_time + base_st) & (time <= stim_time + base_end)
            base_val = A_To_pA * (np.mean(y[base_mask]) - trace_base)
            base_vals.append(base_val)

            # Calculate peak (min in peak interval)
            peak_mask = (time >= stim_time + peak_st) & (time <= stim_time + peak_end)
            peak_val = A_To_pA * (np.min(y[peak_mask]) - trace_base)
            peak_vals.append(peak_val)

        # Save all three plots in one vertical layout
        fig, axs = plt.subplots(5, 1, figsize=(8, 10), sharex=False)

        # 1. Original trace
        axs[0].plot(time, original_y, label="Original")
        axs[0].set_title(f"Original Trace: {trace_name}")
        axs[0].set_ylabel("Value")

        # 2. Zoomed version
        zoom_mask = (time >= zoomStart1) & (time <= zoomEnd1)
        axs[1].plot(time[zoom_mask], original_y[zoom_mask], label=f"Zoomed ({zoomStart1}–{zoomEnd1}s)", color='green')
        axs[1].set_title(f"Zoomed Artifact-Removed Trace ({zoomStart1}–{zoomEnd1}s)")
        axs[1].set_ylabel("Value")

        # 3. Artifact-removed trace
        axs[2].plot(time, y, label="Stim Artifact Removed", color='orange')
        axs[2].set_title("After Artifact Removal")
        axs[2].set_ylabel("Value")

        # 4. Zoomed version
        zoom_mask = (time >= zoomStart1) & (time <= zoomEnd1)
        axs[3].plot(time[zoom_mask], y[zoom_mask], label=f"Zoomed ({zoomStart1}–{zoomEnd1}s)", color='green')
        axs[3].set_title(f"Zoomed Artifact-Removed Trace ({zoomStart1}–{zoomEnd1}s)")
        axs[3].set_ylabel("Value")

        # 5. Zoomed version
        zoom_mask = (time >= zoomStart2) & (time <= zoomEnd2)
        axs[4].plot(time[zoom_mask], y[zoom_mask], label=f"Zoomed ({zoomStart2}–{zoomEnd2}s)", color='green')
        axs[4].set_title(f"Zoomed Artifact-Removed Trace ({zoomStart2}–{zoomEnd2}s)")
        axs[4].set_xlabel("Time (s)")
        axs[4].set_ylabel("Value")

        # Tight layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(export_folder, f"{trace_name}.pdf"))
        plt.close()

        # Collect results
        for i, stim_time in enumerate(time_of_stim):
            results.append({
                "Trace": trace_name,
                "Stimulus": i + 1,
                "Time (s)": stim_time,
                "Base (pA)": base_vals[i],
                "Peak (pA)": peak_vals[i]
            })

    # Export analysis results
    results_df = pd.DataFrame(results)
    results_df.to_excel(os.path.join(export_folder, "results.xlsx"), index=False)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    trainEval()

