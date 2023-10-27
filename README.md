# neuriteMT
The package analyzes microtubule (MT) orientation in neurites based on the output from utrack / plustiptracker (Danuser lab; https://github.com/DanuserLab/u-track). It will only work with movies of single neurons including their soma. The soma is needed to get the orientation of neurites and thereby of comet tracks.

Used in Schelski and Bradke, 2022, Science Advances (https://www.science.org/doi/10.1126/sciadv.abo2336)

# Installation

The package was developed and tested on Windows.
<br/>
1. If you don't already have Anaconda installed: Download and install Anaconda from https://www.anaconda.com/.
2. If you don't already have git installed: Download and install git from https://git-scm.com/downloads
3. Open a terminal, navigate to the folder where you want to put neuritemt and clone the neuritemt repository:
> git clone https://github.com/maxschelski/neuritemt.git
4. Navigate into the folder of the repository (neuritemt):
> cd neuritemt
5. Create environment for neuritemt with Anaconda:
> conda env create -f environment.yml
6. Activate environment in Anaconda:
> conda activate neuritemt
6. Install neuritemt locally using pip:
> pip install -e .
7. Follow the installation for pyneurite (https://github.com/maxschelski/pyneurite)
5. Use the package
> import neuritemt.mtanalyzer
> 
> from scipy import io
> 
> comet_data_mat = io.loadmat(file_path_to_mat_file)
> 
> analyzer = neuritemt.mtanalyzer.MTanalyzer(comet_data_mat)
> 
> comet_data = analyzer.analyze_orientation()

The comet_data output calculates distance in pixels and time in frames (and therefore speed as pixels/frame). 

For any questions feel free to contact me via E-Mail to max.schelski@googlemail.com.
