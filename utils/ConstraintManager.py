import mdmm
from torch_geometric.utils import batched_negative_sampling, negative_sampling


class MDMMConstraint():
    def __init__(
        self,
        constraint_name:str,
        #type:str,
        #target:float,
        scale:float,
        damping:float,
        model,
    ):
        self.constraint_name=constraint_name
        self.scale=scale
        self.damping=damping
        self.model=model

    def create_constraint(self):
        if self.constraint_name=="nodes":
            constraint_node = mdmm.MaxConstraint(
                lambda x, edges: self.model.features_loss(x, edges),
                max=20.0,
                scale=self.scale,
                damping=self.damping,
            )
            return constraint_node
        if self.constraint_name=="edges":
            constraint_edge = mdmm.MaxConstraint(
                lambda x, edges, batch: model.edges_loss(x, edges, torch.tensor(batched_negative_sampling(edges, batch), dtype=torch.int64)),
                    max=20.0,
                    scale=self.scale,
                    damping=self.damping,
                )
            return constraint_edge
        


class ConstraintsManager():
    def __init__(self, constraint_obs:list[str]):
        self.constraint_obs = constraint_obs
        self.constraint_names = []
        self.constraints = []
        self.arguments = []

    def add_constraint(self, constraint):
        self.constraints.append(constraint)
        '''self.constraint_names.append(constraint.get_name())
        self.arguments.append(constraint.argument_names)'''

    def get_argument_names(self):
        return self.arguments

    def get_arguments_for_batch(self, data, weights, xs_weights, dnn1, dnn2=None):
        
        # Get the arguments
        args = []
        for l in self.arguments:
            tmp = []
            for arg in l:
                if arg == "dnn1":
                    tmp.append(dnn1)
                elif arg == "dnn2":
                    tmp.append(dnn2)
                elif arg == "weights":
                    tmp.append(weights)
                elif arg == "xs_weights":
                    tmp.append(xs_weights)
                else:
                    tmp.append(data[:,self.constraint_obs.index(arg)])

            args.append(tmp)

        return args

    def get_constraints(self):
        return self.constraints
    
    def get_constraint_names(self):
        return self.constraint_names



