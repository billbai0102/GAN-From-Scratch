# gen input output neurons
gen_in = 1
gen_out = 30

# dis input output neurons
dis_in = gen_out
dis_out = gen_in

# generator hidden neurons
gen_hidden = 20
# generator step size
gen_step = 1e-3

# discriminator hidden neurons
dis_hidden = 20
# discriminator step size
dis_step = .01

# gradient clipping - prevents exploding gradients
grad_clip = .2
weight_clip = .25

# train epochs
epochs = 100000
