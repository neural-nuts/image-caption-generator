class Configuration():

    def __init__(self, args):
        self.word_threshold = 2
        self.max_len = 20
        self.dim_imgft = 1536
        self.embedding_size = 256
        self.num_hidden = 256
        self.batch_size = 100
        self.num_timesteps = 22
        self.learning_rate = 0.002
        self.nb_epochs = 10000
        self.bias_init = True
        self.xavier_init = False
        self.dropout = False
        self.lstm_keep_prob = 0.7
        self.beta_l2 = 0.001
        self.batch_decode = False
        self.mode = args["mode"]
        self.resume = args["resume"]
        self.load_image = bool(args.get("load_image"))
        self.data_is_coco = bool(args.get("data_is_coco"))
        self.inception_path = args.get("inception_path", "ConvNets/inception_v4.pb")
        self.saveencoder = bool(args.get("saveencoder"))
        self.savedecoder = bool(args.get("savedecoder"))
