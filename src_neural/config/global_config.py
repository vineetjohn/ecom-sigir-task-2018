class GlobalConfig:

    def __init__(self):
        self.logger_name = "ecom"
        self.log_level = "INFO"
        self.vocab_filename = "products.vocab"
        self.labels_filename = "labels.vocab"
        self.model_filename = "classifier.mdl"


gconf = GlobalConfig()
