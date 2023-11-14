import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.stats import poisson
from matplotlib.ticker import FuncFormatter
import tkinter as tk
from tkinter import filedialog, StringVar, OptionMenu, Entry, Listbox, BooleanVar
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings

warnings.simplefilter("error",OptimizeWarning)

# Define the function for a sum of lognormal distributions
def sum_of_lognorms(x, *params):
    result = np.zeros_like(x)
    for i in range(0, len(params), 3):
        mu, sigma, weight = params[i], params[i + 1], params[i + 2]
        result += weight * np.exp(-(np.log(x) - mu) ** 2 / (2 * sigma ** 2)) / (x * sigma * np.sqrt(2 * np.pi))
    return result

# Function to format tick labels in scientific notation
def format_sci_notation(x, pos):
    if x == 0:
        return '0'
    exp = int(np.log10(x))
    coef = x / 10 ** exp
    return str(exp)

# Function to open a CSV file using a dialog
def open_csv_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        try:
            if skip_rows_toggle.get():
                data = pd.read_csv(file_path,skiprows=11,encoding='iso-8859-1')
            else:
                data = pd.read_csv(file_path,encoding='iso-8859-1')
            filename.set(file_path.split('/')[-1][:-4])
            plot_title.set(filename.get())
            if dt_data is not None:
                dwell_time_dict = extract_dwell_times(dt_data)
                set_dwelltime(dwell_time_dict)
            return data
        except FileNotFoundError:
            print(f"File '{file_path}' not found.")
    return None

def open_dt_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        try:
            dt_data = pd.read_csv(file_path)
            return dt_data
        except FileNotFoundError:
            print(f"File '{file_path}' not found.")
    return None

def extract_dwell_times(dt_data):
    dwell_time_dict = {}  # Create a dictionary to store file name -> dwell time mappings
    if 'Filename' not in dt_data:
        errormsg_window("No column 'Filename' found")
        return -1
    if 'DT / ms' not in dt_data:
        errormsg_window("No column 'DT / ms' found")
        return -1
    for index, row in dt_data.iterrows():
        file_name = row['Filename']  # Assuming 'File Name' is the column containing file names
        dwell_time = row['DT / ms']  # Assuming 'Dwell Time' is the column containing dwell times
        dwell_time_dict[file_name] = dwell_time
    return dwell_time_dict

def errormsg_window(message):
    window = tk.Toplevel()
    window.title("Error Message")
    tk.Label(window, text=message).pack()
    tk.Button(window, text="Okay", command=window.destroy).pack()

def estimate_initial_parameters(data):
    if log_scale_toggle.get():
        mu_estimate = np.mean(np.log(data))
        sigma_estimate = np.std(np.log(data))
        mu_estimate = np.exp(mu_estimate)
        sigma_estimate = np.exp(sigma_estimate)
    else:
        mu_estimate = np.mean(data)
        sigma_estimate = np.std(data)
    return mu_estimate, sigma_estimate

# Function to plot a histogram with logarithmic bin sizes and x-scale
# def plot_histogram(data, column_name, initial_params, plot_title, x_label, y_label, save, num_peaks, bin_width):
def plot_histogram(data, column_name, initial_params, plot_title, x_label, y_label, save, num_peaks, bin_width, column_name_bg):
    plt.close()
    mean = 0.0
    hist2 = None
    if(log_scale_toggle.get()):
        # Calculate the logarithmically spaced bin edges
        if plot_background_toggle.get():
            if min(data[column_name].min(),data[column_name_bg].min()) == 0.0: # to avoid problems on the logarithmic scale, the minimum bin edge is set to log(1) if the lowest value in one of the columns is 0.0
                data = data.drop(data[data[column_name_bg] == 0.0].index)
                log_min = np.log(data[column_name_bg].min())
            else:
                log_min = np.log10(min(data[column_name].min(),data[column_name_bg].min()))
        else:
            log_min = np.log10(data[column_name].min())
        log_max = np.log10(data[column_name].max())
        bin_width = float(bin_width_entry.get())
        num_bins = int((log_max-log_min)/bin_width)

        bin_edges = np.logspace(log_min, log_max, num=num_bins)

        # Create the histogram
        hist, _ = np.histogram(data[column_name], bins=bin_edges)
        if plot_background_toggle.get():
            hist2, _ = np.histogram(data[column_name_bg], bins=bin_edges)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        curve_x_vals = np.logspace(log_min, log_max, 1000)
    else:
        if plot_background_toggle.get():
            bin_edges = np.linspace(min(data[column_name].min(),data[column_name_bg].min()),data[column_name].max(),num=int((data[column_name].max()-min(data[column_name].min(),data[column_name_bg].min()))/bin_width))
        else:
            bin_edges = np.linspace(data[column_name].min(),data[column_name].max(),num=int((data[column_name].max()-data[column_name].min())/bin_width))
        if(len(bin_edges) > 10000):
            errormsg_window("Number of Bins (" + num_bins + ") is greater than 10000, please change the bin_width!")
            return
        hist, _ = np.histogram(data[column_name], bins=bin_edges)
        if plot_background_toggle.get():
            hist2, _ = np.histogram(data[column_name_bg], bins=bin_edges)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        curve_x_vals = np.linspace(data[column_name].min(), data[column_name].max(),num=1000)
        mask = ((curve_x_vals < x_min.get())|(curve_x_vals > x_max.get()))
        curve_x_vals = curve_x_vals[mask]
        curve_x_vals = np.append(curve_x_vals,np.linspace(x_min.get(),x_max.get(),num=1000))
        curve_x_vals.sort()

    sigma1_field.config(text = str(np.std(data[column_name])))
    # Clear the previous plot
    ax.clear()
    ax1.clear()
    ax2.clear()

    # Plot the histogram and the fitted curve
    if plot_background_toggle.get():
        ax.hist(data[column_name_bg], bins=bin_edges, color="#1f77b4", edgecolor='k', alpha=0.2, label='Background', zorder=0)
        ax1.hist(data[column_name_bg], bins=bin_edges, color="#1f77b4", edgecolor='k', alpha=0.2, label='Background', zorder=0)
        ax2.hist(data[column_name_bg], bins=bin_edges, color="#1f77b4", edgecolor='k', alpha=0.2, label='Background', zorder=0)
    if plot_histogram_toggle.get():
        ax.hist(data[column_name], bins=bin_edges, color="#1f77b4", edgecolor='k', alpha=0.5, label='Data', zorder=1)
        ax1.hist(data[column_name], bins=bin_edges, color="#1f77b4", edgecolor='k', alpha=0.5, label='Data', zorder=1)
        ax2.hist(data[column_name], bins=bin_edges, color="#1f77b4", edgecolor='k', alpha=0.5, label='Data', zorder=1)
    if fit.get():
        if initial_params is None:
            mu_estimate, sigma_estimate = estimate_initial_parameters(data[column_name])
            initial_params = [mu_estimate, sigma_estimate, 1.0]
            if(num_peaks==2):
                initial_params.append([mu_estimate, sigma_estimate, 1.0])
        try:
            could_not_fit_label.grid_forget()
            params, _ = curve_fit(sum_of_lognorms, bin_centers, hist, p0=initial_params)
            # Create the fitted curve
            fitted_curve = sum_of_lognorms(curve_x_vals, *params)
            if num_peaks == 2:
                if plot_curve_toggle.get():
                    ax.plot(curve_x_vals, fitted_curve, 'r-', label='Fitted Sum of Lognormal Distributions', zorder=4)
                    ax1.plot(curve_x_vals, fitted_curve, 'r-', label='Fitted Sum of Lognormal Distributions', zorder=4)
                    ax2.plot(curve_x_vals, fitted_curve, 'r-', label='Fitted Sum of Lognormal Distributions', zorder=4)
                if plot_peak1_toggle.get():
                    fc1 = sum_of_lognorms(curve_x_vals, *params[:3])
                    ax.plot(curve_x_vals, fc1, '-', color='orange', label="Lognormal Distribution 1", zorder=2)
                    ax1.plot(curve_x_vals, fc1, '-', color='orange', label="Lognormal Distribution 1", zorder=2)
                    ax2.plot(curve_x_vals, fc1, '-', color='orange', label="Lognormal Distribution 1", zorder=2)
                if plot_peak2_toggle.get():
                    fc2 = sum_of_lognorms(curve_x_vals, *params[3:])
                    ax.plot(curve_x_vals, fc2, '-', color='indigo', label="Lognormal Distribution 2", zorder=3)
                    ax1.plot(curve_x_vals, fc2, '-', color='indigo', label="Lognormal Distribution 2", zorder=3)
                    ax2.plot(curve_x_vals, fc2, '-', color='indigo', label="Lognormal Distribution 2", zorder=3)
            else:
                if plot_curve_toggle.get():
                    ax.plot(curve_x_vals, fitted_curve, 'r-', label="Fitted Lognormal Distribution", zorder=5)
                    ax1.plot(curve_x_vals, fitted_curve, 'r-', label="Fitted Lognormal Distribution", zorder=5)
                    ax2.plot(curve_x_vals, fitted_curve, 'r-', label="Fitted Lognormal Distribution", zorder=5)
            if plot_median_toggle.get():
                # mean = np.average(bin_centers,weights=hist)
                mean = data[column_name].mean()
                if num_peaks == 2:
                    med_label = "$\mu_{total}$ = "
                else:
                    med_label = "$\mu$ = "
                ax.axvline(mean,color="lime", linewidth=2, label=med_label + '%.2f' % mean, zorder=7)
                ax1.axvline(mean,color="lime", linewidth=2, label=med_label + '%.2f' % mean, zorder=7)
            if plot_median1_toggle.get():#TODO: Camake callable when plotting only one curve such that both the mean of the distribution and fir can be plotted
                ax.axvline(np.exp(params[0]),color="gold", linewidth=2, label="$\mu_1$ = " + '%.2f' % np.exp(params[0]), zorder=8)
                ax1.axvline(np.exp(params[0]),color="gold", linewidth=2, label="$\mu_1$ = " + '%.2f' % np.exp(params[0]), zorder=8)
            init_params_entry.set('%.2f' % params[0] + "," + '%.2f' % params[1] + "," + '%.2f' % params[2])
            if(num_peaks==2):
                if plot_median2_toggle.get():
                    ax.axvline(np.exp(params[3]),color="blueviolet", linewidth=2, label="$\mu_2$ = " + '%.2f' % np.exp(params[3]), zorder=9)
                    ax1.axvline(np.exp(params[3]),color="blueviolet", linewidth=2, label="$\mu_2$ = " + '%.2f' % np.exp(params[3]), zorder=9)
                init_params2_entry.set('%.2f' % params[3] + "," + '%.2f' % params[4] + "," + '%.2f' % params[5])
        except (RuntimeError, OptimizeWarning) as error:
            errormsg_window("Curve fit unsuccessful.\n Try varying the initial parameters")
            # could_not_fit_label.grid(column=0,row=13)
            # could_not_fit_label.grid(column=10,row=12)
    elif plot_median_toggle.get():
        # mean = np.average(bin_centers,weights=hist)
        mean = data[column_name].mean()
        ax.axvline(mean,color="lime", linewidth=2, label="\u03bc = " + '%.2E' % mean, zorder=7)
        ax1.axvline(mean,color="lime", linewidth=2, label="\u03bc = " + '%.2E' % mean, zorder=7)

    if log_scale_toggle.get():    
        ax.set_xscale('log')  # Set x-axis to a logarithmic scale
        ax1.set_xscale('log')  # Set x-axis to a logarithmic scale
        ax.xaxis.set_major_formatter(FuncFormatter(format_sci_notation)) # Format the x-axis tick labels in scientific notation
        ax1.xaxis.set_major_formatter(FuncFormatter(format_sci_notation)) # Format the x-axis tick labels in scientific notation
        x_label = "log$_{10}$(" + x_label + ")"
    else:
        ax.set_xscale('linear')
        ax1.set_xscale('linear')

    # plot appearance
    ax.set_title(plot_title)
    ax.set_xlabel(x_label)
    ax.set_xlim(x_min.get(),x_max.get())
    ax2.set_title(plot_title)
    ax1.set_xlabel(x_label)
    ax1.set_xlim(x_min.get(),x_max.get())
    # handle different combinations of y-axis limits
    if not np.isnan(y_max.get()):
        if not np.isnan(y_min.get()):
            ax.set_ylim(y_min.get(),y_max.get())
            ax1.set_ylim(y_min.get(),y_max.get())
        else:
            ax.set_ylim(0.0,y_max.get())
            ax1.set_ylim(0.0,y_max.get())
    elif not np.isnan(y_min.get()) and hist is not None:
        ax.set_ylim(y_min.get(),1.05*max(hist.max(),hist2.max()))
    if not np.isnan(y2_max.get()):
        if not np.isnan(y2_min.get()):
            ax2.set_ylim(y2_min.get(),y2_max.get())
        else:
            ax2.set_ylim(0.0,y2_max.get())
    elif not np.isnan(y2_min.get()) and hist2 is not None:
        ax2.set_ylim(y2_min.get(),1.05*max(hist.max(),hist2.max()))
    ax.set_ylabel(y_label)
    ax1.set_ylabel(y_label)
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax2.plot([0, 1], [0, 0], transform=ax2.transAxes, **kwargs)
    ax1.plot([0, 1], [1, 1], transform=ax1.transAxes, **kwargs)
    #
    # legend_handles = []
    handles, labels = ax.get_legend_handles_labels()
    # legend_labels = []
    index = 0
    if plot_curve_toggle.get():
        if curve_legend.get()!='' and fit.get():
            labels[index] = curve_legend.get()
        index+=1
    if plot_peak1_toggle.get():
        if peak1_legend.get()!='' and fit.get() and num_peaks == 2:
            labels[index] = peak1_legend.get()
        index+=1
    if plot_peak2_toggle.get():
        if peak2_legend.get()!='' and fit.get() and num_peaks == 2:
            labels[index] = peak2_legend.get()
        index+=1
    if plot_median_toggle.get():
        if median_legend.get()!='':
            labels[index] = median_legend.get()
        index+=1
    if plot_median1_toggle.get():
        if median1_legend.get()!='' and fit.get():
            labels[index] = median1_legend.get()
        index+=1
    if plot_median2_toggle.get():
        if median2_legend.get()!='' and fit.get() and num_peaks == 2:
            labels[index] = median2_legend.get()
        index+=1
    if plot_background_toggle.get():
        if background_legend.get()!='' and column_name_bg!='':
            labels[index] = background_legend.get()
        index+=1
    if plot_histogram_toggle.get():
        if histogram_legend.get()!='':
            labels[index] = histogram_legend.get()
    
    ax.legend(handles, labels)
    ax1.legend(handles, labels)
    
    # Save the image
    if save:
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if save_path:
            if broken_scale_toggle.get():
                fig2.savefig(save_path)
            else:
                fig.savefig(save_path)
    else:
        if broken_scale_toggle.get():
            canvas2.draw()
        else:
            canvas.draw()

# Main program
root = tk.Tk()
root.title("spPlot")

# Function to update the available columns
def update_columns():
    if data is not None:
        columns = data.columns
        column_dropdown["menu"].delete(0, "end")
        background_dropdown["menu"].delete(0, "end")
        for col in columns:
            column_dropdown["menu"].add_command(label=col, command=tk._setit(selected_column, col))
            background_dropdown["menu"].add_command(label=col, command=tk._setit(background_column, col))

# Function to choose the CSV file
def choose_csv_file():
    global data
    data = open_csv_file()
    if data is not None:
        update_columns()

def choose_dt_file():
    global dt_data
    dt_data = open_dt_file()
    if dt_data is not None:
        dwell_time_dict = extract_dwell_times(dt_data)
        if dwell_time_dict == -1:
            return
        set_dwelltime(dwell_time_dict)

def set_dwelltime(dwell_time_dict):
    global dwelltime
    if filename.get() in dwell_time_dict:
        print(filename.get() + " found")
        dwelltime.set(dwell_time_dict[filename.get()])
    else:
        print(filename.get() + " not found")
        
    
def set_lims(*args):
    set_x_lims()
    set_y_lims()

def set_x_lims():
    global mass
    global factor
    column_name = selected_column.get()
    bg_column_name = background_column.get()
    if data is not None and column_name:
        if mass:
            x_max.set(data[column_name].max()*factor)
            if plot_background_toggle.get() and bg_column_name:
                x_min.set(data[bg_column_name].min()*factor)
            else:
                x_min.set(data[column_name].min()*factor)
        else:
            x_max.set(data[column_name].max())
            if plot_background_toggle.get() and bg_column_name:
                x_min.set(data[bg_column_name].min())
            else:
                x_min.set(data[column_name].min())

def set_y_lims():
    y_min.set(np.nan)
    y_max.set(np.nan)
def set_y2_lims():
    y2_min.set(np.nan)
    y2_max.set(np.nan)

# Function to plot the histogram
def plot_selected_column():
    global mass
    mass = False
    column_name = selected_column.get()
    if data is not None and column_name:
        initial_params_list = [float(param) for param in init_params_entry.get().split(',')]
        if(num_peaks.get()==2):
            initial_params_list.extend([float(param) for param in init_params2_entry.get().split(',')])

        plot_histogram(
            data,
            column_name,
            initial_params_list,
            plot_title_entry.get(),
            x_label_entry.get(),
            y_label_entry.get(),
            False,
            num_peaks.get(),
            bin_width1.get(),
            background_column.get()
        )

# Function to save the histogram
def save_selected_column():
    column_name = selected_column.get()
    if data is not None and column_name:
        initial_params_list = [float(param) for param in init_params_entry.get().split(',')]
        if(num_peaks.get()==2):
            initial_params_list.extend([float(param) for param in init_params2_entry.get().split(',')])
        plot_histogram(
            data,
            column_name,
            initial_params_list,
            plot_title_entry.get(),
            x_label_entry.get(),
            y_label_entry.get(),
            True,
            num_peaks.get(),
            bin_width1.get(),
            background_column.get()
        )

def calculate_me(save):
    global factor
    global mass
    mass = True
    column_name = selected_column.get()
    column_name_bg = background_column.get()
    data_set = data.copy()
    if data_set is not None and column_name:
        pitch_value = float(pitch_entry.get())
        flow_value = float(flow_entry.get())
        dwelltime_value = float(dwelltime_entry.get())
        transport_value = float(transport_entry.get())
        data_set[column_name] = 5/3 * data_set[column_name] / (pitch_value * flow_value * dwelltime_value * transport_value)
        if plot_background_toggle.get():
            data_set[column_name_bg] = 5/3 * data_set[column_name_bg] / (pitch_value * flow_value * dwelltime_value * transport_value)
        factor = 5/3 / (pitch_value * flow_value * dwelltime_value * transport_value)

        # Clear the previous plot
        ax.clear()

        # Convert the comma-separated initial parameters to a list
        initial_params_list = [float(param) for param in init_params_entry.get().split(',')]
        if num_peaks.get() == 2:
            initial_params_list.extend([float(param) for param in init_params2_entry.get().split(',')])

        # Plot the histogram with the new data
        plot_histogram(
            data_set,
            column_name,
            initial_params_list,
            plot_title_entry.get(),
            x_label.get(),
            y_label_entry.get(),
            save,
            num_peaks.get(),
            bin_width2.get(),
            column_name_bg
        )

def plot_me():
    if x_label.get() =='':
        x_label.set("Mass / Event / fg")
    calculate_me(False)

def save_me():
    calculate_me(True)

def toggle_bin_width():
    global bw1_old
    if log_scale_toggle.get():
        tmp = bin_width1.get()
        if bw1_old is None:
            bin_width1.set(0.05)
        else:
            bin_width1.set(bw1_old)
        bw1_old = tmp
    else :
        tmp = bin_width1.get()
        if bw1_old is None:
            bin_width1.set(100)
        else:
            bin_width1.set(bw1_old)
        bw1_old = tmp

def toggle_params():
    toggle_fit()
    if(num_peaks.get()==2):
        initial_params2_label.grid(row=20,column=0)
        initial_params2_entry.grid(row=21,column=0)
        init_params2_entry.set(initial_params_entry.get())
    else:
        initial_params2_label.grid_forget()
        initial_params2_entry.grid_forget()

def toggle_broken_axis():
    if broken_scale_toggle.get():
        y2_min_label.grid(column=3, row=29)
        y2_min_entry.grid(column=4, row=29)
        y2_max_label.grid(column=6, row=29)
        y2_max_entry.grid(column=7, row=29)
        reset_y2_lims.grid(column=9, row=29)
        canvas.get_tk_widget().grid_forget()
        canvas2.get_tk_widget().grid(row=0, column=1, rowspan=25, columnspan=9)
        ax2.spines.bottom.set_visible(False)
        ax1.spines.top.set_visible(False)
        ax2.xaxis.tick_top()
        ax2.tick_params(top=False)
        ax2.tick_params(labeltop=False)  # don't put tick labels at the top
        ax1.xaxis.tick_bottom()
        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax2.plot([0, 1], [0, 0], transform=ax2.transAxes, **kwargs)
        ax1.plot([0, 1], [1, 1], transform=ax1.transAxes, **kwargs)
    else:
        y2_min_label.grid_forget()
        y2_min_entry.grid_forget()
        y2_max_label.grid_forget()
        y2_max_entry.grid_forget()
        reset_y2_lims.grid_forget()
        canvas2.get_tk_widget().grid_forget()
        canvas.get_tk_widget().grid(row=0, column=1, rowspan=25, columnspan=9)

def reset_fit_params():
    init_params_entry.set("1.0,1.0,1.0")
    init_params2_entry.set("1.0,1.0,1.0")

def reset_labels():
    histogram_legend.set('')
    curve_legend.set('')
    peak1_legend.set('')
    peak2_legend.set('')
    median_legend.set('')
    median1_legend.set('')
    median2_legend.set('')
    background_legend.set('')

def toggle_fit():
    if fit.get():
        plot_curve_checkbox.grid(row=1, rowspan=2, column=plot_options_column)
        curve_label.grid(row=1,column=plot_options_column+1)
        curve_title.grid(row=2,column=plot_options_column+1)
        plot_curve_toggle.set(True)
        if num_peaks.get() == 2:
            plot_peak1_checkbox.grid(row=3, rowspan=2, column=plot_options_column)
            peak1_label.grid(row=3,column=plot_options_column+1)
            peak1_title.grid(row=4,column=plot_options_column+1)
            plot_peak1_toggle.set(True)
            plot_peak2_checkbox.grid(row=5, rowspan=2, column=plot_options_column)
            peak2_label.grid(row=5,column=plot_options_column+1)
            peak2_title.grid(row=6,column=plot_options_column+1)
            plot_peak2_toggle.set(True)
            plot_median1_checkbox.grid(row=9, rowspan=2, column=plot_options_column)
            median1_label.grid(row=9,column=plot_options_column+1)
            median1_title.grid(row=10,column=plot_options_column+1)
            plot_median1_toggle.set(True)
            plot_median2_checkbox.grid(row=11, rowspan=2, column=plot_options_column)
            median2_label.grid(row=11,column=plot_options_column+1)
            median2_title.grid(row=12,column=plot_options_column+1)
            plot_median2_toggle.set(True)
        else:    
            plot_peak1_checkbox.grid_forget()
            peak1_label.grid_forget()
            peak1_title.grid_forget()
            plot_peak1_toggle.set(False)
            plot_peak2_checkbox.grid_forget()
            peak2_label.grid_forget()
            peak2_title.grid_forget()
            plot_peak2_toggle.set(False)
            plot_median1_checkbox.grid_forget()
            median1_label.grid_forget()
            median1_title.grid_forget()
            plot_median1_toggle.set(False)
            plot_median2_checkbox.grid_forget()
            median2_label.grid_forget()
            median2_title.grid_forget()
            plot_median2_toggle.set(False)
    else:
        plot_curve_checkbox.grid_forget()
        curve_label.grid_forget()
        curve_title.grid_forget()
        plot_curve_toggle.set(False)
        plot_peak1_checkbox.grid_forget()
        peak1_label.grid_forget()
        peak1_title.grid_forget()
        plot_peak1_toggle.set(False)
        plot_peak2_checkbox.grid_forget()
        peak2_label.grid_forget()
        peak2_title.grid_forget()
        plot_peak2_toggle.set(False)
        plot_median1_checkbox.grid_forget()
        median1_label.grid_forget()
        median1_title.grid_forget()
        plot_median1_toggle.set(False)
        plot_median2_checkbox.grid_forget()
        median2_label.grid_forget()
        median2_title.grid_forget()
        plot_median2_toggle.set(False)


# Create variables
data = None
dt_data = None
filename = StringVar()
plot_title = StringVar()
log_scale_toggle = BooleanVar()
plot_background_toggle = BooleanVar()
fit = BooleanVar()
skip_rows_toggle = BooleanVar(value=True)
broken_scale_toggle = BooleanVar()
dwelltime = tk.DoubleVar()
selected_column = StringVar()
background_column = StringVar()
x_label = StringVar()
y_label = StringVar()
x_min = tk.DoubleVar()
x_max=tk.DoubleVar()
y_min = tk.DoubleVar(value=np.nan)
y_max=tk.DoubleVar(value=np.nan)
y2_min = tk.DoubleVar(value=np.nan)
y2_max=tk.DoubleVar(value=np.nan)
init_params_entry = StringVar(value="1.0, 1.0, 1.0")
init_params2_entry = StringVar()
bin_width1 = tk.DoubleVar(value=100)
bin_width2 = tk.DoubleVar(value=0.05)
bw1_old = None
mass = False
factor = 1.0
num_peaks = tk.IntVar()
histogram_legend=StringVar()
curve_legend=StringVar()
peak1_legend=StringVar()
peak2_legend=StringVar()
median_legend=StringVar()
median1_legend=StringVar()
median2_legend=StringVar()
background_legend=StringVar()
sigma1 = StringVar()

choose_file_button = tk.Button(root, text="Choose CSV File", command=choose_csv_file)
skiprows_checkbox = tk.Checkbutton(root, text="Skip first Rows", variable=skip_rows_toggle, onvalue=True, offvalue=False)
plot_button = tk.Button(root, text="Plot Raw Data", command=plot_selected_column)
save_button = tk.Button(root, text="Save Raw Data", command=save_selected_column)
close_button = tk.Button(root, text="Exit", command=root.destroy)

second_log = tk.Checkbutton(root, text="Second Distribution", variable=num_peaks, onvalue=2, offvalue=1, command=toggle_params)

column_dropdown = OptionMenu(root, selected_column, "")
selected_column.trace("w",set_lims)
background_checkbox = tk.Checkbutton(root, text="Plot Background", variable=plot_background_toggle, onvalue=True, offvalue=False)
background_dropdown = OptionMenu(root, background_column, "")

initial_params_label = tk.Label(root, text="Initial Parameters (mu, sigma, weight; comma-separated):")
initial_params_entry = Entry(root, textvariable=init_params_entry)
initial_params2_label = tk.Label(root, text="Initial Parameters 2 (mu, sigma, weight; comma-separated):")
initial_params2_entry = Entry(root, textvariable=init_params2_entry)
initial_params2_label.pack_forget()
initial_params2_entry.pack_forget()
reset_initial_params = tk.Button(root, text="Reset fit parameters", command=reset_fit_params)

plot_title_label = tk.Label(root, text="Plot Title:")
plot_title_entry = Entry(root, textvariable=plot_title)

x_label_label = tk.Label(root, text="X-Axis Label:")
x_label_entry = Entry(root, textvariable=x_label)

y_label_label = tk.Label(root, text="Y-Axis Label:")
y_label_entry = Entry(root, textvariable=StringVar(value="Frequency per Bin"))

bin_width_label = tk.Label(root, text="Bin Width")
bin_width_entry = Entry(root, textvariable=bin_width1)

bin_width2_label = tk.Label(root, text="Bin Width")
bin_width2_entry = Entry(root, textvariable=bin_width2)

log_scale_toggle_button = tk.Checkbutton(root, text="Logarithmic X-Axis", variable=log_scale_toggle, onvalue=True, offvalue=False, command=toggle_bin_width)
fit_toggle_button = tk.Checkbutton(root, text="Fit Curve", variable=fit, onvalue=True, offvalue=False, command=toggle_fit)

toggle_broken_axis_button = tk.Checkbutton(root, text="Broken Y-Axis", variable=broken_scale_toggle, onvalue=True, offvalue=False, command=toggle_broken_axis)

# Create a Figure and an Axes object
fig = Figure()
ax = fig.add_subplot(111)

fig2, (ax2,ax1) = plt.subplots(2,1,sharex=True, gridspec_kw={'height_ratios':[3,7]})
fig2.subplots_adjust(hspace=0.05)

# Create a canvas to display the Figure
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=0,column=1,rowspan=25, columnspan=9)

canvas2 = FigureCanvasTkAgg(fig2, master=root)
# canvas2.get_tk_widget().grid(row=0,column=1,rowspan=25, columnspan=9)

#Dwelltime entry
choose_dt_file_button = tk.Button(root, text="Choose Dwelltime file", command=choose_dt_file)
dwelltime_label = tk.Label(root, text="Dwelltime / ms:")
dwelltime_entry = Entry(root, textvariable=dwelltime)

#Transport efficiency
transport_label = tk.Label(root, text="Transport Efficiency / %:")
transport_entry = Entry(root, textvariable=StringVar(value="100.00"))

#Volume flow
flow_label = tk.Label(root, text="Volume Flow /\u03bcL/min:")
flow_entry = Entry(root, textvariable=StringVar(value="10.00"))

# cali pitch
pitch_label = tk.Label(root, text="Calibration Pitch:")
pitch_entry = Entry(root, textvariable=StringVar())

#calculate & plot/save mayy/event histogram
plot_me_button = tk.Button(root, text="Plot Mass/Event", command=plot_me)
save_me_button = tk.Button(root, text="Save Mass/Event", command=save_me)

could_not_fit_label = tk.Label(root, text="Curve could not be fitted")

# x axis boundaries
x_min_label = tk.Label(root, text="X min:")
x_min_entry = Entry(root, textvariable=x_min)
x_max_label = tk.Label(root, text="X max:")
x_max_entry = Entry(root, textvariable=x_max)
reset_x_lims = tk.Button(root, text="Reset X Limits", command=set_x_lims)
y_min_label = tk.Label(root, text="Y min:")
y_min_entry = Entry(root, textvariable=y_min)
y_max_label = tk.Label(root, text="Y max:")
y_max_entry = Entry(root, textvariable=y_max)
reset_y_lims = tk.Button(root, text="Reset Y Limits", command=set_y_lims)
y2_min_label = tk.Label(root, text="Y min:")
y2_min_entry = Entry(root, textvariable=y2_min)
y2_max_label = tk.Label(root, text="Y max:")
y2_max_entry = Entry(root, textvariable=y2_max)
reset_y2_lims = tk.Button(root, text="Reset Y2 Limits", command=set_y2_lims)

# Checkboxes for element selection
plot_histogram_toggle = tk.BooleanVar(value=True)
plot_histogram_checkbox = tk.Checkbutton(root, text="Plot Histogram", variable=plot_histogram_toggle, onvalue=True, offvalue=False, state="active")
histogram_title = tk.Label(root, text="Histogram Label")
histogram_label = tk.Entry(root, textvariable=histogram_legend)

plot_curve_toggle = tk.BooleanVar(value=True)
plot_curve_checkbox = tk.Checkbutton(root, text="Plot Fitted Curve", variable=plot_curve_toggle, onvalue=True, offvalue=False)
curve_title = tk.Label(root, text="Curve Label")
curve_label = tk.Entry(root, textvariable=curve_legend)

plot_median_toggle = tk.BooleanVar(value=True)
plot_median_checkbox = tk.Checkbutton(root, text="Plot Median", variable=plot_median_toggle, onvalue=True, offvalue=False)
median_title = tk.Label(root, text="Median Label")
median_label = tk.Entry(root, textvariable=median_legend)

plot_peak1_toggle = tk.BooleanVar(value=True)
plot_peak1_checkbox = tk.Checkbutton(root, text="Plot Peak 1", variable=plot_peak1_toggle, onvalue=True, offvalue=False)
peak1_title = tk.Label(root, text="Curve1 Label")
peak1_label = tk.Entry(root, textvariable=peak1_legend)

plot_median1_toggle = tk.BooleanVar(value=True)
plot_median1_checkbox = tk.Checkbutton(root, text="Plot Median 1", variable=plot_median1_toggle, onvalue=True, offvalue=False)
median1_title = tk.Label(root, text="Median1 Label")
median1_label = tk.Entry(root, textvariable=median1_legend)

plot_peak2_toggle = tk.BooleanVar(value=True)
plot_peak2_checkbox = tk.Checkbutton(root, text="Plot Peak 2", variable=plot_peak2_toggle, onvalue=True, offvalue=False)
peak2_title = tk.Label(root, text="Curve2 Label")
peak2_label = tk.Entry(root, textvariable=peak2_legend)

plot_median2_toggle = tk.BooleanVar(value=True)
plot_median2_checkbox = tk.Checkbutton(root, text="Plot Median 2", variable=plot_median2_toggle, onvalue=True, offvalue=False)
median2_title = tk.Label(root, text="Median2 Label")
median2_label = tk.Entry(root, textvariable=median2_legend)

plot_background_toggle = tk.BooleanVar()
plot_background_checkbox = tk.Checkbutton(root, text="Plot Background", variable=plot_background_toggle, onvalue=True, offvalue=False)
background_title = tk.Label(root, text="Background Label")
background_label = tk.Entry(root, textvariable=background_legend)

reset_labels_button = tk.Button(root, text="Reset Labels", command=reset_labels)

sigma1_label = tk.Label(root, text="$\sigma = $")
sigma1_field = tk.Label(root)

# Place elements in the window
choose_file_button.grid(row=0,column=0)
skiprows_checkbox.grid(row=1, column=0)
column_dropdown.grid(row=2,column=0)
plot_title_label.grid(row=3,column=0)
plot_title_entry.grid(row=4,column=0)
x_label_label.grid(row=5,column=0)
x_label_entry.grid(row=6,column=0)
log_scale_toggle_button.grid(row=7,column=0)
fit_toggle_button.grid(row=8,column=0)
y_label_label.grid(row=9,column=0)
y_label_entry.grid(row=10,column=0)
bin_width_label.grid(row=11,column=0)
bin_width_entry.grid(row=12,column=0)
plot_button.grid(row=13,column=0)
save_button.grid(row=15,column=0)
close_button.grid(row=16,column=0)
initial_params_label.grid(row=17,column=0)
initial_params_entry.grid(row=18,column=0)
second_log.grid(row=19,column=0)
reset_initial_params.grid(row=22,column=0)

choose_dt_file_button.grid(row=0,column=10)
dwelltime_label.grid(row=1,column=10)
dwelltime_entry.grid(row=2,column=10)
transport_label.grid(column=10,row=3)
transport_entry.grid(column=10,row=4)
flow_label.grid(column=10,row=5)
flow_entry.grid(column=10,row=6)
pitch_label.grid(column=10,row=7)
pitch_entry.grid(column=10,row=8)
bin_width2_label.grid(column=10,row=9)
bin_width2_entry.grid(column=10,row=10)
plot_me_button.grid(row=11,column=10)
save_me_button.grid(row=13,column=10)
sigma1_label.grid(row=14,column=10)
sigma1_field.grid(row=15,column=10)

x_min_label.grid(column=3,row=26)
x_min_entry.grid(column=4,row=26)
x_max_label.grid(column=6,row=26)
x_max_entry.grid(column=7,row=26)
reset_x_lims.grid(column=9,row=26)
y_min_label.grid(column=3,row=27)
y_min_entry.grid(column=4,row=27)
y_max_label.grid(column=6,row=27)
y_max_entry.grid(column=7,row=27)
reset_y_lims.grid(column=9,row=27)
toggle_broken_axis_button.grid(column=3, row=28)

plot_options_column = 11  # Adjust the column number as needed, depending on width of the pyplot figure

reset_labels_button.grid(row=0, column=plot_options_column, columnspan=2)
# plot_curve_checkbox.grid(row=1, rowspan=2, column=plot_options_column)
# curve_label.grid(row=1,column=plot_options_column+1)
# curve_title.grid(row=2,column=plot_options_column+1)
# plot_peak1_checkbox.grid(row=3, rowspan=2, column=plot_options_column)
# peak1_label.grid(row=3,column=plot_options_column+1)
# peak1_title.grid(row=4,column=plot_options_column+1)
# plot_peak2_checkbox.grid(row=5, rowspan=2, column=plot_options_column)
# peak2_label.grid(row=5,column=plot_options_column+1)
# peak2_title.grid(row=6,column=plot_options_column+1)
plot_median_checkbox.grid(row=7, rowspan=2, column=plot_options_column)
median_label.grid(row=7,column=plot_options_column+1)
median_title.grid(row=8,column=plot_options_column+1)
# plot_median1_checkbox.grid(row=9, rowspan=2, column=plot_options_column)
# median1_label.grid(row=9,column=plot_options_column+1)
# median1_title.grid(row=10,column=plot_options_column+1)
# plot_median2_checkbox.grid(row=11, rowspan=2, column=plot_options_column)
# median2_label.grid(row=11,column=plot_options_column+1)
# median2_title.grid(row=12,column=plot_options_column+1)
plot_histogram_checkbox.grid(row=13, rowspan=2, column=plot_options_column)
histogram_label.grid(row=13,column=plot_options_column+1)
histogram_title.grid(row=14,column=plot_options_column+1)
plot_background_checkbox.grid(row=15, rowspan=2, column=plot_options_column)
background_label.grid(row=15,column=plot_options_column+1)
background_title.grid(row=16,column=plot_options_column+1)
background_dropdown.grid(row=17, column=plot_options_column, columnspan=2)

root.mainloop()

