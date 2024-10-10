import torch
import mdmm
from tqdm import tqdm
import os
import logging
from torch_geometric.utils import to_dense_adj
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from torch_geometric.utils import batched_negative_sampling, negative_sampling
import numpy as np
import time

logger = logging.getLogger(__name__)

#from python.trainingHelpers import EarlyStopping


class GAETrainer():

    def __init__(self, data_manager, training_config, data_loading_config):
        
        self.data_manager = data_manager
        self.training_config = training_config
        self.data_loading_config = data_loading_config
        #self.training_variables = self.training_config.training_variables


        self.loader_train, self.loader_valid, self.loader_test_back, self.loader_signals = self.data_manager.loader(self.data_loading_config, self.training_config.batch_size)

        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.training_config.model.to(self.device)

        self.batch_size = self.training_config.batch_size
        self.epochs = self.training_config.epochs
        self.scale = self.training_config.scale
        self.damping = self.training_config.damping
        self.num_features = self.training_config.num_features

        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    def __make_mdmm_optimizer(self):

        mdmm_module=mdmm.MDMM([self.training_config.nodes_constraint])
        optimizer = mdmm_module.make_optimizer(self.model.parameters(), lr=self.training_config.learning_rate)

        return mdmm_module, optimizer
    
    def __train_batch(self, batch, mdmm_module, optimizer):



        self.model.train()
        
        x = batch.x
        edges = batch.edge_index
        _batch = batch.batch.to(self.device)

        edges_loss = self.model.edges_loss(
        x,
        edges,
        # Negative sampling is important to also show the network
        # examples of non-connected nodes
        batched_negative_sampling(
            edges,
            _batch,
        ).to(self.device),
    )

    #features_loss = model.features_loss(x, edges)

        mdmm_return = mdmm_module(edges_loss, [(x, edges)])
    # mdmm_return = MDMM_module(features_loss, [(x, edges, _batch)])

        optimizer.zero_grad()
        mdmm_return.value.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)      #gradient clipping
    #loss.backward()
        optimizer.step()

        with torch.no_grad():
             features_loss = self.model.features_loss(x, edges)

    # with torch.no_grad():
    #         edges_loss = model.edges_loss(
    #             x,
    #             edges,
    #             torch.tensor(batched_negative_sampling(edges, _batch), dtype=torch.int64),
    #         )

        return float(edges_loss.detach()), float(features_loss.detach())

    def validation(self, graph):

        self.model.eval()
        x = graph.x
        edges = graph.edge_index
        negative_edge_index=negative_sampling(edges).to(self.device)


        if negative_edge_index.numel() == 0:
            print(to_dense_adj(edges))
            print(edges)
            assert negative_edge_index.numel() > 0, "Negative edge index is empty"

        edges_loss = self.model.edges_loss(
            x,
            edges,
            #Negative sampling is important to also show the network
            # examples of non-connected nodes
            negative_sampling(
                edges,
            ).to(self.device),
        )

        features_loss = self.model.features_loss(x, edges)

        #loss = edges_loss + features_loss

        return float(edges_loss.detach()), float(features_loss.detach())
    

    def __save_model(self, output_path) -> None:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        torch.save(self.model.state_dict(), output_path + "/model.pt")

    def train_and_validate(self, output_path):


        mdmm_module, optimizer =self.__make_mdmm_optimizer()


        train_loss_batch_def=[]
        train_loss_batch_edges_def=[]
        train_loss_batch_nodes_def=[]
        train_loss_batch=[]
        train_loss_batch_edges=[]
        train_loss_batch_nodes=[]

        train_loss_epoch=[]
        train_loss_epoch_edges=[]
        train_loss_epoch_nodes=[]

        validation_loss_batch_def=[]
        validation_loss_batch_edges_def=[]
        validation_loss_batch_nodes_def=[]
        validation_loss_batch=[]
        validation_loss_batch_edges=[]
        validation_loss_batch_nodes=[]

        validation_loss_epoch=[]
        validation_loss_epoch_edges=[]
        validation_loss_epoch_nodes=[]



        epochsx=self.epochs
        best_loss=0.0

#torch.autograd.set_detect_anomaly(True)
        start= time.time()
        for epoch in range(epochsx):
            for batch in tqdm(self.loader_train, desc=f"Epoch{epoch}", leave=False):
                loss_edges, loss_nodes = self.__train_batch( batch, mdmm_module, optimizer)
                train_loss_batch.append((loss_edges+loss_nodes))
                train_loss_batch_edges.append(loss_edges)
                train_loss_batch_nodes.append((loss_nodes))
                train_loss_batch_def.append((loss_edges+loss_nodes))
                train_loss_batch_edges_def.append(loss_edges)
                train_loss_batch_nodes_def.append((loss_nodes))
            train_loss_epoch.append(np.mean(train_loss_batch))
            train_loss_epoch_edges.append(np.mean(train_loss_batch_edges))
            train_loss_epoch_nodes.append(np.mean(train_loss_batch_nodes))
            train_loss_batch=[]
            train_loss_batch_edges=[]
            train_loss_batch_nodes=[]
            print(f"{epoch}-th training completed!")


            for graph in tqdm(self.loader_valid,  leave=False):
          
                if to_dense_adj(graph.edge_index).shape == torch.Size([1,1,1]):
                    print("Single-node graph detected. Skipping this graph...")
                    continue

                loss_edges, loss_nodes = self.validation( graph)
                validation_loss_batch.append((loss_edges+loss_nodes))
                validation_loss_batch_edges.append(loss_edges)
                validation_loss_batch_nodes.append((loss_nodes))
                validation_loss_batch_def.append((loss_edges+loss_nodes))
                validation_loss_batch_edges_def.append(loss_edges)
                validation_loss_batch_nodes_def.append((loss_nodes))
            validation_loss_epoch.append(np.mean(validation_loss_batch))
            validation_loss_epoch_edges.append(np.mean(validation_loss_batch_edges))
            validation_loss_epoch_nodes.append(np.mean(validation_loss_batch_nodes))
            validation_loss_batch=[]
            validation_loss_batch_edges=[]
            validation_loss_batch_nodes=[]
            '''experiment.log_metrics(
                {"loss_train": train_loss_epoch[-1], "loss_val": validation_loss_epoch[-1]}, 
                    step=epoch, 
                    epoch=epoch
                )
            scheduler.step()'''
            print(f"{epoch}-th epoch completed!")
            '''print(validation_loss_epoch_edges[-1])
            print(validation_loss_epoch_nodes[-1])'''



        print(f"--> Training completed in {time.time()-start:.2f} seconds")

        return train_loss_epoch, validation_loss_epoch
    
    def test(self) -> None:

        self.model.eval()


        losses_testing_qcd = []

        for graph in tqdm(self.loader_test_back, desc="Testing QCD", leave=True):
            x = graph.x
            edges = graph.edge_index

            edges_loss = self.model.edges_loss(
                x,
                edges,
                negative_sampling(
                    edges,
                ).to(self.device),
            )

            features_loss = self.model.features_loss(x, edges)
            loss = edges_loss + features_loss

            losses_testing_qcd.append(float(loss.detach()))

        losses_testing_signal = {}
        for name, loader_signal in self.loader_signals.items():
            losses_testing_signal[name] = []
            for graph in tqdm(loader_signal, desc="Testing SVJ", leave=True):
                x = graph.x
                edges = graph.edge_index

                edges_loss = self.model.edges_loss(
                    x,
                    edges,
                    negative_sampling(
                        edges,
                    ).to(self.device),
                )

                features_loss = self.model.features_loss(x, edges)
                loss = edges_loss + features_loss

                losses_testing_signal[name].append(float(loss.detach()))

        y_true={}
        y_score={}
        fpr={}
        tpr={}
        aucx={}
        for signal_name, signal_losses in losses_testing_signal.items():
            y_true[signal_name] = [0]*len(losses_testing_qcd) + [1]*len(signal_losses)
            y_score[signal_name] = losses_testing_qcd + signal_losses
            fpr[signal_name], tpr[signal_name], _ = roc_curve(y_true[signal_name], y_score[signal_name]) 
            aucx[signal_name] = roc_auc_score(y_true[signal_name], y_score[signal_name])

       

        return np.mean(list(aucx.values())), fpr, tpr, aucx






        
        