1. Check the LearnedPendulum-v0 dynamics match the ddp-pendulum-v0 dynamics (rebecca)
2. Try ddp with learnedpendulum-v0. See if DDP loss is reducing. (rebecca)
3. Move timesteps to mpc-params, refactor ddp functions to take timesteps as an argument instead of being from env.timesteps
4. Consolidate mpc-pendulum stuff into ddp-pendulum (harleen)
5. refactor mpc to use learned for control, and then have simulation with "ground truth"
6. try mpc with learnedpendulum-v0
9. QUAD STUFF:
    * Construct quadrotor SympodeN network. Make a network that takes the right number of inputs and outputs with embeded angles (walker)
    * new experiment-simple-quad folder, with train.py and data.py (rebecca) Status: Made, not tested
    * new analyze-simple-quad.py (rebecca) Status: Made, not tested
    * make learnquad environment (rebecca) Status: Made, not tested
9. plots???
    * what we have: energy comparison, test loss during training time,
    * what we need:
        * Plot showing how MPCDDP performs with: ground truth, naive baseline model, symp model
        * Plot showing long rollout of ground truth, naive model, symp model
        * Plot showing "just run the entire 1st ddp traj" with ground_truth, naive baseline, sympmodel         
10. Reach: new loss functions?  