# neuriteMT
The package analyzes microtubule (MT) orientation in neurites based on the output from utrack / plustiptracker (Danuser lab; https://github.com/DanuserLab/u-track). It will only work with movies of single neurons including their soma. The soma is needed to get the orientation of neurites and thereby of comet tracks.

# Installation

1. Open a terminal, navigate to the folder where you want to put the repository and clone it:
> git clone https://github.com/maxschelski/neuritemt.git
2. Navigate into the folder of the repository (neuritemt):
> cd neuritemt
3. Install neuritemt locally using pip:
> pip install -e .
4. Follow the installation for pyneurite (https://github.com/maxschelski/pyneurite)
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
