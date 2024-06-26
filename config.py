class AlphaZeroConfig:
    def __init__(self,
        self_play_batch_size: int = 128,
        max_num_threads: int = 8,
        num_processes:int = 40,
        game_length: int = 100,
    ):

        self.self_play_batch_size = self_play_batch_size
        self.max_num_threads = max_num_threads
        self.num_processes = num_processes

        # gumbel
        self.num_sampled_actions = 2
        self.c_visit = 50
        self.c_scale = 1.0

        # game settings
        self.max_moves = game_length
        self.action_space_size = 4672

        # exploration
        self.exploration_fraction = 0.25
        self.dirichlet_alpha = 0.3
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # other
        self.num_simulations = 2

    def softmax_temperature_fn(self, num_moves=30):
        return 1 if num_moves < 30 else 0
