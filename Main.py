from GA import GA
from DataSetTool import DataSetTool
import pathlib, os
import warnings
warnings.filterwarnings("ignore")

def main_iter(x, y, index):
    # For each experiment, pop up the training set, and use the rest for testing
    print('iter=' + str(index + 1))
    temp_x, temp_y = x, y
    t_x, t_y = x.copy(), y.copy()
    for iter_index in range(len(x)):
        target_x = t_x.pop(iter_index)
        target_y = t_y.pop(iter_index)
        source_x = t_x
        source_y = t_y
        ga = GA(100, max_gen=5)
        ga.fit(source_x, source_y, target_x, target_y)
        t_x, t_y = temp_x.copy(), temp_y.copy()


# Experiment begin
# Import data set
path = os.path.join(pathlib.Path(__file__).parent.absolute(), 'data/')
x_list, y_list = DataSetTool.init_data(path, 20, is_normalized=False)
for i in range(3):
    main_iter(x_list, y_list, i)
