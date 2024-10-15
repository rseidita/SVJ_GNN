import sys
import torch
sys.path.append('/t3home/aaguadob/graph_nets/')
from utils.JetGraphProducer import JetGraphProducer
from utils.LundTreeUtilities import OnTheFlyNormalizer
from data_loading.graphDataLoader import GraphDataLoader




class TrainingDataManager():
    '''
    Class to manage the data loading and splitting.

    Attributes:
    - backgrounds: dictionary containing the information about the background samples
    - signals: dictionary containing the information about the signal samples
    - test_size: fraction of the data to be used for testing
    - validation_size: fraction of the data to be used for validation
    - seed: seed for the random number generator
    - variables: list of variables to be used in the analysis
    - train_data: data used for training
    - test_data: data used for testing
    - train_labels: labels of the training data
    - test_labels: labels of the testing data
    - train_weights: weights of the training data
    - test_weights: weights of the testing data

    Methods:
    - get_train_val_loaders: get the training and validation data loaders
    '''
    def __init__(
        self,
        data_loading_config,
        batch_size
        ):
        '''
        Constructor of the DataManager class.

        Arguments:
        - configuration: path to the configuration
        '''
        
        self.data_loading_config = data_loading_config
        self.backgrounds = data_loading_config.backgrounds
        self.signals = data_loading_config.signals
        self.test_size = data_loading_config.test_size
        self.validation_size=data_loading_config.validation_size
        self.variables = data_loading_config.variables
        self.batch_size=batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_data_from_files(self, sample_dict):
        
        """
        Constructor of JetGraphs

        Returns:
            data: It returns the data as a JetGraph
        """

        data={}
        for sample_name, sample_info in sample_dict["samples"].items():
        
            data[sample_name]=JetGraphProducer(
                sample_info["path"],
                sample_name,
                self.variables[0]["n_store_jets"],
                use_lund_decomp=self.variables[0]["use_lund_decomp"],
                n_lund_vars=self.variables[0]["n_lund_vars"],
                save_energy_fractions=self.variables[0]["save_energy_fractions"],
                save_4vectors_in_lund_tree=self.variables[0]["save_4vectors_in_lund_tree"],
                weights=self.variables[0]["weights"],
                verbose=self.variables[0]["verbose"],
                label=sample_info["class"],
                kt_cut=self.variables[0]["kt_cut"],
                add_self_connections=self.variables[0]["add_self_connections"],
            )
            for i in range(len(data[sample_name])):
                if data[sample_name][i].num_nodes==1:
                    print(i, sample_name)

   

        return data
    


    def Normalizer(self, data):
        
        """Returns a normalizer based on the attributes of background data

        Returns:
            OnTheFlyNormalizer: function that normalizes the data
        """

        means, stds = 0., 0.
        weights = []
        for sample_name, sample_data in data.items():
            for graph in sample_data:
                means = means + graph.x.sum(dim=0)*graph.w
                weights.append(graph.w)
        means=means/sum(weights)
        for sample_name, sample_data in data.items():
            for graph in sample_data:
                stds = stds + ((graph.x - means)**2).sum(dim=0)*graph.w
        stds=stds/sum(weights)
        stds = torch.sqrt(stds)

        return OnTheFlyNormalizer(["x"], means, stds)
    


    def loader(self, data_loading_config, batch_size):
        
        
        """Function that loads the data in a Graph format (careful, the objects are not strictly speaking graphs)

        Returns:
            loader_training: training data (background) in a graph format
            loader_validation: validation data (background) in a graph format
            loader_test_back: test data (background) in a graph format
            loader_svjs: dictionary of test data (signal) in a graph format
            """

        self.data_loading_config=data_loading_config

        data_training_n=self.backgrounds
        data_test_n=self.signals 

        data_training=self._load_data_from_files(data_training_n)
        data_test=self._load_data_from_files(data_test_n)


        sample_names = list(data_training_n["samples"].keys())
        name1, name2 = sample_names[0], sample_names[1]
        
        normalizerr=self.Normalizer(data_training)

        loader_training = GraphDataLoader(
            [
                [
                    data_training_n["samples"][name1]["path"], # path to the dataset
                    name1, # name of the dataset (you can process the same raw data with multiple settings given each a different name)
                ],
                [
                    data_training_n["samples"][name2]["path"], # path to the dataset
                    name2, # name of the dataset (you can process the same raw data with multiple settings given each a different name)
                ],
            ],
            window_to_load=[[0., 1.0-self.test_size-self.validation_size], [0., 1.0-self.test_size-self.validation_size]],
            batch_size=batch_size,
            normalizer=normalizerr,
            device=self.device,
        )
        loader_validation = GraphDataLoader(
            [
                [
                    data_training_n["samples"][name1]["path"], # path to the dataset
                    name1, # name of the dataset (you can process the same raw data with multiple settings given each a different name)
                ],
                [
                    data_training_n["samples"][name2]["path"], # path to the dataset
                    name2, # name of the dataset (you can process the same raw data with multiple settings given each a different name)
                ],
            ],
            window_to_load=[[1.0-self.test_size-self.validation_size, 1.0-self.test_size], [1.0-self.test_size-self.validation_size, 1.0-self.test_size]],
            normalizer=normalizerr,
            device=self.device,
        )
        loader_test_back = GraphDataLoader(
            [
                [
                    data_training_n["samples"][name1]["path"], # path to the dataset
                    name1, # name of the dataset (you can process the same raw data with multiple settings given each a different name)
                ],
                [
                    data_training_n["samples"][name2]["path"], # path to the dataset
                    name2, # name of the dataset (you can process the same raw data with multiple settings given each a different name)
                ],
            ],
            window_to_load=[[1.0-self.test_size, 1.0], [1.0-self.test_size, 1.0]],
            normalizer=normalizerr,
            device=self.device,
        )
        loader_svjs={}
        for signal_names, signal_data in data_test_n["samples"].items():
            loader_svjs[signal_names]= GraphDataLoader(
                [
                    [
                        data_test_n["samples"][signal_names]["path"], # path to the dataset
                        signal_names, # name of the dataset (you can process the same raw data with multiple settings given each a different name)
                    ]
                ],
                window_to_load=[[0., 1.]],
                normalizer=normalizerr,
                device=self.device,
            )
        return loader_training, loader_validation, loader_test_back, loader_svjs


