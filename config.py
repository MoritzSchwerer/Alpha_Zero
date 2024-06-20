class AlphaZeroConfig:
    def __init__(self):

        self.self_play_batch_size = 128
        self.max_num_threads = 8
        self.num_processes = 32

        # gumbel
        self.num_sampled_actions = 4
        self.c_visit = 50
        self.c_scale = 1.0

        # game settings
        self.max_moves = 100
        self.action_space_size = 4672

        # exploration
        self.exploration_fraction = 0.25
        self.dirichlet_alpha = 0.3
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # other
        self.num_simulations = 16

    def softmax_temperature_fn(self, num_moves=30):
        return 1 if num_moves < 30 else 0
