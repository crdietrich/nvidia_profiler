# nvidia_profiler
#   Log GPU, CPU and RAM usage to CSV file.
#   Uses psutil and nvdia-smi.
#
# Copyright (c) 2019,  Scott C. Lowe <scott.code.lowe@gmail.com>
# Copyright (c) 2023,  Colin Dietrich
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in
#       the documentation and/or other materials provided with the distribution
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import psutil
import subprocess
import time
import threading

import pandas as pd


def check_nvidia_smi():
    try:
        subprocess.check_output(['nvidia-smi'])
    except FileNotFoundError:
        raise EnvironmentError('The nvidia-smi command could not be found.')


check_nvidia_smi()


def poll_cpu():
    """Get current CPU, RAM and Swap utilisation

    Returns
    -------
    float, CPU utilisation in percent
    float, RAM utilisation in percent
    float, Swap utilisation in percent
    """
    return (
        psutil.cpu_percent(),
        psutil.virtual_memory().percent,
        psutil.swap_memory().percent,
    )


def get_gpu_names():
    res = subprocess.check_output(['nvidia-smi', '-L'])
    return [i_res for i_res in res.decode().split('\n') if i_res != '']


def plot_data(fp, **kwargs):
    """Plot SMI data
    
    Parameters
    ----------
    fp : str, filepath to csv data file
    kwargs : any Pandas plot keyword argument
    """
    
    df = pd.read_csv(fp)
    y_columns = [c for c in df.columns if '(C)' in c]
    df.plot(x='Timestamp', secondary_y=y_columns, **kwargs);


class Logger(threading.Thread):
    def __init__(
            self,
            fp_output_filename=None, fp_output_directory=None,
            fname_timestamp=None, refresh_interval=0.05,
            iter_limit=None, show_header=True,
            show_units=True,
    ):
        """Record CPU and NVIDIA GPU usage
        
        Parameters
        ----------
        fp_output_filename : str, prefix or whole filename to write data to
        fp_output_directory : str, path to write data to. Defaults to current
        fname_timestamp : str, timestamp to include in the output filename. Generated if not set.
        refresh_interval : float, interval in seconds to sample data. Defaults to 0.05.
        iter_limit : int, limit on how many iterations to run. Defaults to None.
        show_header: bool, write header in output file
        show_units : bool, write units to header labels and plot
        """
        super().__init__()

        self.fname = None
        if fp_output_filename is not None:
            if fname_timestamp is None:
                t = time.strftime("%Y_%m_%d-%H_%M_%S", time.localtime())
            else:
                t = fname_timestamp
            suffix = ''
            if '.csv' not in fp_output_filename:
                suffix = suffix + '.csv'

            d = ''
            if fp_output_directory is not None:
                d = fp_output_directory + '/'
            self.fname = f"{d}{fp_output_filename}-{t}-timing{suffix}"

        self.refresh_interval = refresh_interval
        self.iter_limit = iter_limit
        self.show_header = show_header
        self.header_count = 0
        self.show_units = show_units
        self.col_width = 10
        self.time_field_name = 'Timestamp'
        self._header_written = False

        self.cpu_field_names = 'CPU,RAM,Swap'
        if self.show_units:
            self.cpu_field_names = 'CPU (%),RAM (%),Swap (%)'
        self.gpu_field_names = ['GPU', 'Mem', 'Temp']
        if self.show_units:
            self.gpu_field_names = ['GPU (%)', 'Mem (%)', 'Temp (C)']

        self.gpu_queries = [
            'utilization.gpu',
            'utilization.memory',
            'temperature.gpu',
        ]
        self.gpu_query = ','.join(self.gpu_queries)
        self.gpu_names = get_gpu_names()

        self.thread_1 = None
        self._stop = threading.Event()

    def poll_gpus(self, flatten=False):
        """
        Query GPU utilisation, and sanitise results
        Returns
        -------
        list of lists of utilisation stats
            For each GPU (outer list), there is a list of utilisations
            corresponding to each query (inner list), as a string.
        """
        res = subprocess.check_output(
            ['nvidia-smi',
             '--query-gpu=' + self.gpu_query,
             '--format=csv,nounits,noheader']
        )
        lines = [i_res for i_res in res.decode().split('\n') if i_res != '']
        data = [[val.strip() if 'Not Supported' not in val else 'N/A'
                 for val in line.split(',')
                 ] for line in lines]
        if flatten:
            data = [y for row in data for y in row]
        return data

    def write_record(self):
        """Write a data record to csv format"""
        header = self.time_field_name + "," + self.cpu_field_names

        _h_gpu_list = []
        for i_gpu in range(len(self.gpu_names)):
            _h_gpu = ['{}:{}'.format(i_gpu, fn) for fn in self.gpu_field_names]
            _h_gpu_list.append(",".join(_h_gpu))
        header += "," + ",".join(_h_gpu_list) + "\n"

        with open(self.fname, 'a') as f:
            if not self._header_written:
                f.write(header)
                self._header_written = True
            stats = list(poll_cpu())
            t = str(pd.Timestamp("now"))
            stats.insert(0, t)
            stats += self.poll_gpus(flatten=True)
            f.write(','.join([str(stat) for stat in stats]) + '\n')

    def run(self, n_iter=None):
        """Run data collection"""
        while not self._stop.is_set():
            t0 = time.time()
            self.write_record()
            t1 = time.time()
            dt = t1 - t0
            t_sleep = self.refresh_interval - dt
            if t_sleep > 0:
                time.sleep(t_sleep)

    def stop(self):
        self._stop.set()
                
    def run_threaded(self):
        """Run data collection in a separate thread"""
        self.thread_1 = threading.Thread(target=self.run)
        self.thread_1.start()

    def __call__(self, n_iter=None):
        self.run(n_iter=n_iter)
