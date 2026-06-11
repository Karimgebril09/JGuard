# experiment 1

## data

batchsize:32

## model
model arch:
```
self.Fxu = nn.Sequential(
    nn.Linear(state_dim + input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, state_dim)
)

# G(x_t,u_t)=>z_t (Observation model)
self.Gxu = nn.Sequential(
    nn.Linear(state_dim + input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, output_dim)
)     
```

state_dim(xt) = 768    # Latent state dimension<br>
input_dim(ut) = 768    # Input embedding dimension<br>
output_dim(zt) = 768   # Observation embedding dimension<br>
hidden_dim_ssm = 512  # Hidden layer size of dialogue dynamics

## training loop
num_epochs=200,<br> 
weight_decay=0,<br>
ssm_learning_rate=1e-4

# final result

Models saved as models_best_ssm, ssm loss on validation: 0.0002651262960474317

==============================

# experiment 2


## data

batchsize:32

## model
model arch:
```
self.Fxu = nn.Sequential(
    nn.Linear(state_dim + input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, state_dim)
)

# G(x_t,u_t)=>z_t (Observation model)
self.Gxu = nn.Sequential(
    nn.Linear(state_dim + input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, output_dim)
)     
```

state_dim(xt) = 768    # Latent state dimension<br>
input_dim(ut) = 768    # Input embedding dimension<br>
output_dim(zt) = 768   # Observation embedding dimension<br>
hidden_dim_ssm = 512  # Hidden layer size of dialogue dynamics

## training loop
num_epochs=200,<br> 
weight_decay=0,<br>
ssm_learning_rate=5e-5

# final result

Models saved as models_best_ssm, ssm loss on validation: 0.00026592366500861116



==============================

# experiment 5


## data
batchsize:32

## model
model arch:
```
self.Fxu = nn.Sequential(
    nn.Linear(state_dim + input_dim, hidden_dim1),
    nn.ReLU(),
    nn.Linear(hidden_dim1, hidden_dim2),
    nn.ReLU(),
    nn.Linear(hidden_dim2, state_dim)
)

# G(x_t,u_t)=>z_t (Observation model)
self.Gxu = nn.Sequential(
    nn.Linear(state_dim + input_dim, hidden_dim1),
    nn.ReLU(),
    nn.Linear(hidden_dim1, hidden_dim2),
    nn.ReLU(),
    nn.Linear(hidden_dim2, output_dim)
)
```

state_dim(xt) = 768    # Latent state dimension<br>
input_dim(ut) = 768    # Input embedding dimension<br>
output_dim(zt) = 768   # Observation embedding dimension<br>
hidden_dim_ssm1 = 1200 <br> 
hidden_dim_ssm2 = 900 
 

## training loop
num_epochs=200,<br> 
weight_decay=0,<br>
ssm_learning_rate=1e-4
# final result

Models saved as models_best_ssm, ssm loss on validation: 0.00024878944613091234



==============================

# experiment 4


## data
batchsize:32

## model
model arch:
```
self.Fxu = nn.Sequential(
    nn.Linear(state_dim + input_dim, hidden_dim1),
    nn.ReLU(),
    nn.Linear(hidden_dim1, hidden_dim2),
    nn.ReLU(),
    nn.Linear(hidden_dim2, state_dim)
)

# G(x_t,u_t)=>z_t (Observation model)
self.Gxu = nn.Sequential(
    nn.Linear(state_dim + input_dim, hidden_dim1),
    nn.ReLU(),
    nn.Linear(hidden_dim1, hidden_dim2),
    nn.ReLU(),
    nn.Linear(hidden_dim2, output_dim)
)
```

state_dim = 332    # x_t<br>
input_dim = 768    # u_t<br>
output_dim = 768   # z_t<br>
hidden_dim_ssm1 = 1000 <br> 
hidden_dim_ssm2 = 850 
 

## training loop
num_epochs=200,<br> 
weight_decay=0,<br>
ssm_learning_rate=1e-4
# final result

Models saved as models_best_ssm, ssm loss on validation: Models saved as models_best_ssm, ssm loss: 0.00025501811372426647


# Notes

1-collect data that is non attacking
